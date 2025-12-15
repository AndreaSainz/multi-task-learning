import tomosipo as ts
import torch
from torch.nn import MSELoss, CrossEntropyLoss, Module
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from code.dataloader.data_loader import MnistDataset
from code.utils.metrics import update_confusion_matrix, metrics_from_confusion_matrix, soft_argmax_2d
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import logging
import os
import json
import random
import numpy as np
import gc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



class ModelBase(Module):
    
    def __init__(self, model_path, dimension, cut, seed, batch_size, epochs, optimizer_type, learning_rate, debug,  accelerator, task,  scheduler = True):
        super().__init__()

        # model parameters
        self.task = task.strip().lower()
        if self.task not in {"classify", "centre_finder"}:
            raise ValueError("This class only supports classification and centre finding.")
        self.num_classes = 10

        # dataset parameters
        self.model = None
        self.max_len_train = None
        self.max_len_val = None
        self.max_len_test = None
    

        # model parameters
        self.model_path = model_path
        self.batch_size = batch_size
        self.dimension = dimension
        self.H, self.W = self.dimension
        self.num_angles = 10
        self.sigma = 1
        self.standard_dose = True
        self.cut = cut
        self.seed = seed
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.debug = debug
        self.logger = self._setup_logger() if self.debug else None
        self.optimizer_type = optimizer_type
        self.optimizer_ = None
        self.loss_fn = None
        self.scheduler = scheduler
        

        # accelerator for faster code
        self.accelerator = accelerator
        self.trainable_params = None

        # set the device once for the whole class
        self.device = self.accelerator.device

        # Create tomosipo volume and projection geometry
        self.vg = ts.volume(shape=(1,self.H,self.W))                            # Volumen
        self.angles = np.linspace(0, np.pi, self.num_angles, endpoint=True)     # Angles
        self.pg = ts.parallel(angles=self.angles, shape=(1, self.H))            # Parallel beam structure
        self.A = ts.operator(self.vg,self.pg)         
   


    def _setup_logger(self):
        if not self.debug:
            return None

        logger = logging.getLogger(f"ModelBase::{self.model_path}")
        logger.setLevel(logging.INFO)

        if logger.handlers:
            return logger

        log_path = f"{self.model_path}.log"
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

        fh = logging.FileHandler(log_path, mode="a")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        logger.propagate = False
        return logger


    def _log(self, msg: str):
        if self.logger is not None:
            self.logger.info(msg)

    def _set_seed(self):

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    

    def _get_dataloaders(self):
        
        # load training and validation datasets together from the training dataset
        data = MnistDataset(training = True,
                testing= False, 
                A = self.A, 
                dimension = self.dimension, 
                num_angles = self.num_angles, 
                sigma = self.sigma,
                standard_dose = self.standard_dose,
                cut = self.cut,
                seed = self.seed,
                max_len = None,
                device = self.device)

        # len of training and validation based in maximun length given
        n_total = len(data)
        if self.max_len_train is not None:
            n_train = min(self.max_len_train, n_total)
        else:
            #por ejemplo, 90% train / 10% val
            n_train = int(0.9 * n_total)


        n_val_t = n_total - n_train

        if self.max_len_val is not None:
            n_val = min(self.max_len_val, n_val_t)
        else:
            #por ejemplo, 90% train / 10% val
            n_val = n_val_t

        #split in training and validation
        n_rest = n_total - (n_train + n_val)

        lengths = [n_train, n_val]
        if n_rest > 0:
            lengths.append(n_rest)

        g_split = torch.Generator()
        g_split.manual_seed(self.seed)
        splits = random_split(data, lengths, generator=g_split)

        train_data = splits[0]
        val_data = splits[1]

        # create dataloader for both and a generator for reproducibility
        g = torch.Generator() 
        g.manual_seed(self.seed)

        train_dataloader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            generator=g)

        val_dataloader = DataLoader(
            val_data,
            batch_size=self.batch_size,
            shuffle=False)

        return (train_dataloader, val_dataloader)
            

    def _get_test_dataloaders(self):

        test_data = MnistDataset(training = False,
                testing= True, 
                A = self.A, 
                dimension = self.dimension, 
                num_angles = self.num_angles, 
                sigma = self.sigma,
                standard_dose = self.standard_dose,
                cut = self.cut,
                seed = self.seed,
                max_len = self.max_len_test,
                device = self.device)
        
        test_dataloader = DataLoader(
                test_data,
                batch_size=self.batch_size,
                shuffle=False)

        return test_dataloader


    def train_one_epoch(self, train_dataloader, opt, loss, e):

        # set the model in training mode
        self.model.train()
        

        # initialize the total training loss
        total_train_loss = 0
        n_samples = 0

        # loop over the training set
        for batch_id, batch in enumerate(train_dataloader):

            # take data from batch
            gt_image = batch["ground_truth_image"]
            gt_label = batch["label"]
            gt_centre = batch["centre"]

            if self.task == "classify":
                gt = gt_label
            else: 
                gt = gt_centre
            
            # perform a forward pass and calculate the training loss
            pred = self.model(gt_image)
            if self.task != "classify":
                    pred, _ = soft_argmax_2d(pred, beta=50.0)
            loss_value = loss(pred, gt)

        

            # zero out the gradients, perform the backpropagation step, and update the weights
            opt.zero_grad()
            self.accelerator.backward(loss_value)
            opt.step()


            # add the loss to the total training loss so far
            B = gt.shape[0]
            total_train_loss += loss_value.item() * B
            n_samples += B

            #Clean memory
            del gt_image, gt_label, gt_centre, gt, pred, loss_value
            if batch_id % 10 == 0:
                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        return total_train_loss, n_samples


    def validation(self, val_dataloader, loss, e):
        
        # switch off autograd for evaluation
        with torch.no_grad():

            # set the model in evaluation mode
            self.model.eval()

            # initialize validation metrics
            total_val_loss = 0
            n_samples = 0
            
            #initlize metrics
            state, metrics_cfg = self._init_metric_state()

            # loop over the validation set
            for batch_id, batch in enumerate(tqdm(val_dataloader, desc=f"Validation Epoch {e+1}", leave=False)):

                # take data from batch
                gt_image = batch["ground_truth_image"]
                gt_label = batch["label"]
                gt_centre = batch["centre"]

                if self.task == "classify":
                    gt = gt_label
                else: 
                    gt = gt_centre
                
                # make the predictions and calculate the validation loss
                pred = self.model(gt_image)
                if self.task != "classify":
                    pred, _ = soft_argmax_2d(pred, beta=50.0)
                loss_value = loss(pred, gt)


                #update metrics
                self._update_metrics_state(state, metrics_cfg, pred, gt)
                
                # add the loss to the total validation loss so far
                B = gt.shape[0]
                total_val_loss+= loss_value.item()*B
                n_samples += B
                
                del gt_image, gt_label, gt_centre, gt, pred, loss_value
                #Clean memory
                if batch_id % 10 == 0:
                    gc.collect()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()

                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()
        
        val_metrics = self._finalize_metrics_state(state, metrics_cfg)
        return total_val_loss, n_samples, val_metrics

        

    def train(self, max_len_train = None, max_len_val=None, epochs=None):

        #changing paths parameters
        epochs = epochs or self.epochs
        self.max_len_train = max_len_train
        self.max_len_val = max_len_val

        # fix seed
        self._set_seed()

        # load training and validation datasets
        train_dataloader, val_dataloader = self._get_dataloaders()

        # send models to device
        self.model.to(self.device)
        
        
        # Get only parameters that have requires_grad=True
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        # Optimizer
        if self.optimizer_type == "Adam":
            self.optimizer = Adam(trainable_params, lr=self.learning_rate)
        elif self.optimizer_type == "AdamW":
            self.optimizer = AdamW(trainable_params, lr=self.learning_rate)
        elif self.optimizer_type == "SGD":
            self.optimizer = SGD(trainable_params, lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

        # Loss function
        if self.task == "classify":
            self.loss_fn =CrossEntropyLoss()
            loss = self.loss_fn
        else:
            self.loss_fn = MSELoss()
            loss = self.loss_fn



        # accelerates training
        self.model, opt, train_dataloader, val_dataloader = self.accelerator.prepare(self.model, self.optimizer, train_dataloader, val_dataloader)
        
        
        # initialize scheduler for learning rate
        if self.scheduler:
            scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
        
        
        # initialize a dictionary to store training history
        history = {
            "train_loss": [],
            "val_loss": [],
        }

        
        t = tqdm(range(epochs), desc="Epochs")

        self._log(f"=== TRAIN START | task={self.task} | epochs={epochs} | lr={self.learning_rate} | bs={self.batch_size} ===")

        # loop over our epochs
        for e in t:

            # call the training function
            total_train_loss, train_n = self.train_one_epoch(train_dataloader, opt, loss, e)

            # call the validation function
            total_val_loss, val_n, val_metrics = self.validation(val_dataloader, loss, e)
            history.setdefault("val_metrics", []).append(val_metrics)

            # update our training history
            avg_train_loss = total_train_loss / max(1, train_n)
            avg_val_loss = total_val_loss/ max(1, val_n)
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            # save checkpoint every 5 epochs
            if e  % 5 == 1:
                self.save_checkpoint(epoch=e, history=history)
                self._log(f"Checkpoint saved at epoch {e}")

            # update the learning rate scheduler (when the validation loss is not improving)
            if self.scheduler:
                scheduler.step(avg_val_loss)
            
        
            # print the model training and validation information
            t.set_postfix({
                "train_loss": f"{avg_train_loss:.6f}",
                "val_loss": f"{avg_val_loss:.6f}"
            })

            #logger
            lr_now = opt.param_groups[0]["lr"]
            self._log(
                f"[Epoch {e+1}/{epochs}] "
                f"lr={lr_now:.3e} | "
                f"train_loss={avg_train_loss:.6f} | "
                f"val_loss={avg_val_loss:.6f} | "
                f"val_metrics={val_metrics}"
            )
            
            #Clean memory
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()


        # Save final metrics and model
        
        with open(f'{self.model_path}_metrics.json', 'w') as f:
            json.dump(history, f)
        self._log(f"Final training history saved at {self.model_path}_metrics.json")
        self.save_checkpoint(epoch=epochs, history=history)
        self._log("Final checkpoint saved")
        self._log("=== TRAIN END ===")
        return history


    def evaluate(self, test_dataloader, loss):

        # save ground truth and predictions
        gt_images = []
        gts = []
        preds = []
        per_sample_mse = []

        # switch off autograd for evaluation
        with torch.no_grad():

            # set the model in evaluation mode
            self.model.eval()
           
            # initialize metrics
            total_test_loss = 0
            n_samples = 0


            #initilize metrics
            state, metrics_cfg = self._init_metric_state()

            # loop over the validation set
            for batch_id, batch in enumerate(tqdm(test_dataloader)):
                # take data from batch
                gt_image = batch["ground_truth_image"]
                gt_label = batch["label"]
                gt_centre = batch["centre"]

                if self.task == "classify":
                    gt = gt_label
                else: 
                    gt = gt_centre

                

                # make the predictions and calculate the validation loss
                pred = self.model(gt_image)
                if self.task != "classify":
                    pred, _ = soft_argmax_2d(pred, beta=50.0)
                loss_value = loss(pred, gt)

                #update metrics
                self._update_metrics_state(state, metrics_cfg, pred, gt)

                if self.task != "classify":
                    mse_each = ((pred - gt) ** 2).sum(dim=1)
                    per_sample_mse.append(mse_each.detach().cpu())
                
                # add the loss to the total validation loss so far
                B = gt.shape[0]
                total_test_loss += loss_value.item() * B
                n_samples += B
                


                # save gound truth and prediction
                gt_images.append(gt_image.cpu())
                gts.append(gt.cpu())
                preds.append(pred.cpu())
                
                del gt_image, gt_label, gt_centre, gt, pred, loss_value
                #Clean memory
                if batch_id % 10 == 0:
                    gc.collect()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()

                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()
        
        test_metrics = self._finalize_metrics_state(state, metrics_cfg)
        per_sample_mse = torch.cat(per_sample_mse).numpy() if len(per_sample_mse) else None
        return gt_images, gts, preds, total_test_loss, n_samples, test_metrics, per_sample_mse

    def test(self, max_len_test=None,):

        #changing paths parameters
        self.max_len_test = max_len_test


        # fix seed for reproducibility
        self._set_seed()

        # get test data loader
        test_dataloader = self._get_test_dataloaders()
        

        #initialice accelerator
        self.model, test_dataloader = self.accelerator.prepare(self.model, test_dataloader)
        loss = self.loss_fn

        self._log("=== TEST START ===")
        
        # call evaliation function
        gt_images, gts, preds, total_test_loss, test_n, test_metrics, per_sample_mse = self.evaluate(test_dataloader, loss)
    
        # training history
        avg_test_loss = total_test_loss / max(1, test_n)
        
        
        results = {
            "test_loss": avg_test_loss,
            "metrics": test_metrics
        }
        if per_sample_mse is not None:
            results["per_sample_mse"] = per_sample_mse.tolist()
        
        # save predictions and ground truth
        torch.save(torch.cat(gt_images), f"{self.model_path}_ground_truth_images.pt")
        if self.task == "classify":
            torch.save(torch.cat(gts), f"{self.model_path}_ground_truth_labels.pt")
            torch.save(torch.cat(preds), f"{self.model_path}_predicted_labels.pt")
        else: 
            torch.save(torch.cat(gts), f"{self.model_path}_ground_truth_centres.pt")
            torch.save(torch.cat(preds), f"{self.model_path}_predicted_centres.pt")


        # save metrics to file
        with open(f'{self.model_path}_test_metrics.json', 'w') as f:
            json.dump(results, f)

        self._log(f"test_loss={avg_test_loss:.6f} | test_metrics={test_metrics}")
        self._log(f"Final test metrics saved at {self.model_path}_test_metrics.json")
        self._log("=== TEST END ===")

        return results
    

    ############################################################################ Metrics for batch ############################################################################

    def _init_metric_state(self):
        """
        Creates per-epoch metric state (running sums / confusion matrices).
        Returns:
            state, metrics_cfg
        """
        if self.task == "classify":
            metrics_cfg = {"model": ["classifier_metrics"]}
            state = {"model": {"n_batches": 0, "cm": torch.zeros(self.num_classes, self.num_classes, device=self.device, dtype=torch.int64)}}

        else: 
            metrics_cfg = {"model": ["mse"]}
            state = {"model": {"mse_sum": 0.0, "n_samples": 0}}
        return state, metrics_cfg

    def _update_metrics_state(self, state, metrics_cfg, pred, gt):
        """
        Updates metric accumulators for a batch.
        """
        # model1 metrics (CT recon)
        if self.task == "classify":
            for m in metrics_cfg["model"]:
                if m == "classifier_metrics":
                    # accumulate confusion matrix on-device
                    state["model"]["cm"] = update_confusion_matrix(state["model"]["cm"], pred, gt, num_classes=self.num_classes)
            state["model"]["n_batches"] += 1

        else:
            err2 = ((pred - gt) ** 2).sum(dim=1)   # MSE por muestra
            state["model"]["mse_sum"] += err2.sum().item()
            state["model"]["n_samples"] += gt.shape[0]



    def _finalize_metrics_state(self, state, metrics_cfg):
        """
        Converts accumulators into final metrics dict (means + classifier metrics).
        """
        results = {}

        if self.task == "classify":
            cm_cpu = state["model"]["cm"].detach().cpu()
            results["model"] = {}
            results["model"]["confusion_matrix"] = cm_cpu.tolist()
            for m in metrics_cfg["model"]:
                if m == "classifier_metrics":
                    results["model"].update(metrics_from_confusion_matrix(cm_cpu))


        else:
            n = max(1, state["model"]["n_samples"])
            results["model"] = {}
            results["model"]["mse"] = state["model"]["mse_sum"] / n

        return results
    

    ############################################################################ Checkpoints and save model ############################################################################

    def _get_config(self):
        return {
            "task": self.task,
            "num_classes": self.num_classes,
            "model_path": self.model_path,
            "dimension": list(self.dimension),
            "num_angles": self.num_angles,
            "sigma": self.sigma,
            "standard_dose": self.standard_dose,
            "cut": self.cut,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "optimizer_type": self.optimizer_type,
            "learning_rate": self.learning_rate,
            "scheduler": self.scheduler,
            "max_len_train": self.max_len_train,
            "max_len_val": self.max_len_val,
            "max_len_test": self.max_len_test,
            # útil para recrear exactamente el modelo
            "model_class": self.model.__class__.__name__ if self.model is not None else None,
        }


    def save_checkpoint(self, epoch, history):

        # 1) config JSON
        cfg_path = f"{self.model_path}_config.json"
        with open(cfg_path, "w") as f:
            json.dump(self._get_config(), f, indent=2)

        # 2) checkpoint torch
        ckpt = {
            "epoch": epoch,
            "model_state": self.accelerator.unwrap_model(self.model).state_dict(),
            "optimizer_state": self.optimizer.state_dict() if hasattr(self, "optimizer") and self.optimizer is not None else None,
            "loss_fn": self.loss_fn.__class__.__name__ if self.loss_fn is not None else None,
            "history": history,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        torch.save(ckpt, f"{self.model_path}_checkpoint.pt")


    def load_checkpoint(self, load_optimizer: bool = True):
        ckpt_path = f"{self.model_path}_checkpoint.pt"
        ckpt = torch.load(ckpt_path, map_location=self.device)

        # carga pesos
        self.model.to(self.device)
        self.model.load_state_dict(ckpt["model_state"])

        # carga optimizer si procede
        if load_optimizer and ckpt.get("optimizer_state") is not None:
            if not hasattr(self, "optimizer") or self.optimizer is None:
                raise RuntimeError("Optimizer not initialised. Call train() or create self.optimizer before loading it.")
            self.optimizer.load_state_dict(ckpt["optimizer_state"])

        # restaura RNG (opcional, para reproducibilidad)
        rs = ckpt.get("rng_state", None)
        if rs is not None:
            random.setstate(rs["python"])
            np.random.set_state(rs["numpy"])
            torch.set_rng_state(rs["torch"])
            if torch.cuda.is_available() and rs["cuda"] is not None:
                torch.cuda.set_rng_state_all(rs["cuda"])

        return ckpt.get("epoch", None), ckpt.get("history", None)
    

    ############################################################################ Plots ############################################################################

    def plot_training(self, history, out_path):
        if out_path is None:
            out_path = f"{self.model_path}_train_curves.png"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        epochs = np.arange(1, len(train_loss) + 1)

        plt.figure()
        plt.plot(epochs, train_loss, label="train_loss")
        plt.plot(epochs, val_loss, label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Training / Validation loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

    def plot_validation_metrics(self, history, out_path):
        val_metrics = history.get("val_metrics", [])
        if not val_metrics:
            return
        
        if out_path is None:
            out_path = f"{self.model_path}_val_metrics.png"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        # get the metrics based on the model trained
        model_metrics = [vm.get("model", {}) for vm in val_metrics]
        keys = sorted({k for d in model_metrics for k in d.keys()})

        if not keys:
            return

        epochs = np.arange(1, len(model_metrics) + 1)

        plt.figure()
        for k in keys:
            y = [d.get(k, np.nan) for d in model_metrics]
            plt.plot(epochs, y, label=k)

        plt.xlabel("epoch")
        plt.ylabel("metric")
        plt.title("Validation metrics")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

    
    def plot_test(self, out_path=None, normalize_cm=True):
        if out_path is None:
            out_path = os.path.dirname(self.model_path) or "."
        os.makedirs(out_path, exist_ok=True)

        with open(f"{self.model_path}_test_metrics.json", "r") as f:
            results = json.load(f)

        if self.task == "classify":
            cm = np.array(results["metrics"]["model"]["confusion_matrix"], dtype=np.float32)

            cm_plot = cm.copy()
            if normalize_cm:
                row_sums = cm_plot.sum(axis=1, keepdims=True)
                cm_plot = cm_plot / np.maximum(row_sums, 1e-12)
                fname = f"{os.path.basename(self.model_path)}_test_confusion_matrix_normalized.png"
            else:
                fname = f"{os.path.basename(self.model_path)}_test_confusion_matrix.png"

            out_png = os.path.join(out_path, fname)

            plt.figure()
            plt.imshow(cm_plot, aspect="auto")
            plt.colorbar()
            plt.xlabel("pred")
            plt.ylabel("gt")
            plt.title("Confusion matrix" + (" (normalized)" if normalize_cm else ""))
            plt.tight_layout()
            plt.savefig(out_png, dpi=200)
            plt.close()

        else:
            per_sample_mse = results.get("per_sample_mse", None)
            if per_sample_mse is None:
                raise RuntimeError("No per_sample_mse found. You must compute it in evaluate() and save it to JSON.")

            x = np.asarray(per_sample_mse, dtype=np.float32)
            out_png = os.path.join(out_path, f"{os.path.basename(self.model_path)}_test_mse_violin.png")

            plt.figure()
            plt.violinplot(x, showmeans=True, showextrema=True)
            plt.ylabel("MSE per sample")
            plt.title("Test per-sample MSE distribution")
            plt.tight_layout()
            plt.savefig(out_png, dpi=200)
            plt.close()
