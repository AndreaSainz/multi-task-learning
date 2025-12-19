import tomosipo as ts
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.nn import MSELoss, CrossEntropyLoss, Module
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from code.dataloader.data_loader import MnistDataset
from code.utils.metrics import compute_psnr_results, compute_ssim, compute_mse, update_confusion_matrix, metrics_from_confusion_matrix
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import logging
from torch.nn import SmoothL1Loss
import json
import random
import numpy as np
import gc
import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



class ProjectBase(Module):
    
    def __init__(self, model_path, dimension, num_angles, sigma, standard_dose, cut, seed, batch_size, epochs, optimizer_type, learning_rate, debug,  accelerator,  scheduler = (True,True)):
        super().__init__()

                                                                             
        # dataset parameters
        self.model1 = None
        self.model2 = None
        self.model3 = None
        self.training_path = None
        self.validation_path = None
        self.test_path = None
        
        self.sigma = sigma
        self.max_len_train = None
        self.max_len_val = None
        self.max_len_test = None
    

        # model parameters
        self.num_classes = 10
        self.model_path = model_path
        self.batch_size = batch_size
        self.dimension = dimension
        self.H, self.W = self.dimension
        self.num_angles = num_angles
        self.sigma = sigma
        self.standard_dose = standard_dose
        self.cut = cut
        self.seed = seed
        self.epochs = epochs
        self.learning_rate_2, self.learning_rate_3 = learning_rate
        self.seed = seed
        self.debug = debug
        self.logger = self._setup_logger() if self.debug else None
        self.optimizer_type_2, self.optimizer_type_3 = optimizer_type
        self.optimizer_2 = None
        self.optimizer_3 = None
        self.loss_fn_1 = None
        self.loss_fn_2 = None
        self.loss_fn_3 = None
        self.scheduler_2, self.scheduler_3 = scheduler
        

        # accelerator for faster code
        self.accelerator = accelerator
        self.trainable_params_2 = None
        self.trainable_params_3 = None

        # set the device once for the whole class
        self.device = self.accelerator.device

        # Create tomosipo volume and projection geometry
        self.vg = ts.volume(shape=(1,self.H,self.W))                            # Volumen
        self.angles = np.linspace(0, np.pi, self.num_angles, endpoint=True)     # Angles
        self.pg = ts.parallel(angles=self.angles, shape=(1, self.H))            # Parallel beam structure
        self.A = ts.operator(self.vg,self.pg)         
   
    @staticmethod
    def show_samples(dataloader, save_path, n=5):
        batch = next(iter(dataloader))
        imgs = batch["ground_truth_image"][:n]
        labels = batch["label"][:n]

        plt.figure(figsize=(n * 2, 2))
        for i in range(n):
            plt.subplot(1, n, i + 1)
            plt.imshow(imgs[i, 0].detach().cpu(), cmap="gray")
            plt.title(f"label={labels[i].item()}")
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()



    def show_samples_train(self, ground_truth_image, reconstructed_image, gt_label, pred_label, gt_centre, pred_centre, n=5):
        n = min(n, ground_truth_image.shape[0])

        imgs = ground_truth_image[:n]
        rec_imgs = reconstructed_image[:n]
        labels = gt_label[:n]
        pred_labels = pred_label[:n]
       
        centres = gt_centre[:n]
        pred_centres = pred_centre[:n]

        fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))

        if n == 1:
            axes = np.expand_dims(axes, axis=0)

        for i in range(n):
            # --- Ground truth ---
            axes[i, 0].imshow(imgs[i, 0].detach().cpu(), cmap="gray")
            axes[i, 0].set_title("Ground truth")
            axes[i, 0].axis("off")

            # --- Reconstruction ---
            axes[i, 1].imshow(rec_imgs[i, 0].detach().cpu(), cmap="gray")
            axes[i, 1].set_title("Reconstruction")
            axes[i, 1].axis("off")

            # --- Row title ---
            if pred_labels.ndim == 2:
                pred_cls = int(pred_labels[i].argmax().item())
            else:
                pred_cls = int(pred_labels[i].item())
            txt = (
            f"GT: {int(labels[i].item())} | Pred: {pred_cls}\n"
            f"GT c: ({centres[i,0].item():.1f}, {centres[i,1].item():.1f}) | "
            f"Pred c: ({pred_centres[i,0].item():.1f}, {pred_centres[i,1].item():.1f})"
            )

            # texto dentro del panel (arriba-izquierda)
            axes[i, 0].text(
                0.02, 0.98, txt,
                transform=axes[i, 0].transAxes,
                va="top", ha="left",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2)
            )


        plt.tight_layout()
        plt.savefig(f"{self.model_path}_train_pred_examples.png", dpi=200)
        plt.close()



    def debug_loader(self, dataloader, name="train", n_show=5, save_path=None):
        mins, maxs = [], []
        counts = torch.zeros(self.num_classes, dtype=torch.long)

        # stats + counts
        for batch in dataloader:
            x = batch["ground_truth_image"]
            y = batch["label"]

            mins.append(x.min().item())
            maxs.append(x.max().item())
            counts += torch.bincount(y.cpu(), minlength=self.num_classes)

        self._log(f"[{name}] min={min(mins):.4f}, max={max(maxs):.4f}")
        total = counts.sum().item()
        for c in range(self.num_classes):
            self._log(f"[{name}] class {c}: {counts[c].item()} ({100*counts[c].item()/max(1,total):.2f}%)")

        # 2) show samples
        if save_path is not None:
            self.show_samples(dataloader, save_path, n=n_show)


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


    def train_one_epoch(self, train_dataloader, opt2, opt3, loss1, loss2, loss3, e):

        # set the model in training mode
        self.model1.eval()
        self.model2.train()
        self.model3.train()

        # initilize metrics
        state, metrics_cfg = self._init_metric_state()

        # initialize the total training loss
        total_train_loss_1 = 0
        total_train_loss_2 = 0
        total_train_loss_3 = 0
        n_samples = 0

        # loop over the training set
        for batch_id, batch in enumerate(train_dataloader):

            # send the input to the device
            ground_truth_1 = batch["ground_truth_image"]
            ground_truth_2 = batch["label"]
            ground_truth_3 = batch["centre"]
            input_data = batch["noisy_sinogram"]


            # perform a forward pass and calculate the training loss
            pred_1 = self.model1(input_data)
            loss_value_1 = loss1(pred_1, ground_truth_1)


            # doing simultanious tasks
            pred_1_detached = pred_1.detach()
            # classifier
            pred_2 = self.model2(pred_1_detached)
            loss_value_2 = loss2(pred_2, ground_truth_2)

            # find the center
            pred_3 = self.model3(pred_1_detached)
            loss_value_3 = loss3(pred_3, ground_truth_3)

            

            # printing examples to see how bad the reconstruction is, ONLY in the first epoch
            if e+1 == min(self.epochs, 5) and batch_id == 0:
                self.show_samples_train(ground_truth_1, pred_1_detached, ground_truth_2, pred_2, ground_truth_3, pred_3, n=5)

            # zero out the gradients, perform the backpropagation step, and update the weights
            # Model 1 is just a SIRT so it does not train
                                                                #opt1.zero_grad()
                                                                #self.accelerator.backward(loss_value_1)
                                                                #opt1.step()

            opt2.zero_grad()
            self.accelerator.backward(loss_value_2)
            opt2.step()


            opt3.zero_grad()
            self.accelerator.backward(loss_value_3)
            opt3.step()

            # update metrics
            self._update_metrics_state(
                state, metrics_cfg,
                pred_1, ground_truth_1,
                pred_2, ground_truth_2,
                pred_3, ground_truth_3,
                data_range=1.0,
                compute_train_mse=True)

            # add the loss to the total training loss so far
            B = ground_truth_1.shape[0]
            total_train_loss_1 += loss_value_1.item()*B
            total_train_loss_2 += loss_value_2.item()*B
            total_train_loss_3 += loss_value_3.item()*B
            n_samples += B

            #Clean memory
            del input_data, ground_truth_1, ground_truth_2, ground_truth_3, pred_1, pred_2, pred_3, loss_value_1, loss_value_2, loss_value_3
            if batch_id % 10 == 0:
                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        train_metrics = self._finalize_metrics_state(state, metrics_cfg)
        return total_train_loss_1, total_train_loss_2, total_train_loss_3, train_metrics, n_samples


    def validation(self, val_dataloader, loss1, loss2, loss3, e):
        
        # switch off autograd for evaluation
        with torch.no_grad():

            # set the model in evaluation mode
            self.model1.eval()
            self.model2.eval()
            self.model3.eval()

            # initialize validation metrics
            total_val_loss_1 = 0
            total_val_loss_2 = 0
            total_val_loss_3 = 0
            n_samples = 0
            
            #initlize metrics
            state, metrics_cfg = self._init_metric_state()

            # loop over the validation set
            for batch_id, batch in enumerate(tqdm(val_dataloader, desc=f"Validation Epoch {e+1}", leave=False)):

                # send the input to the device
                ground_truth_1 = batch["ground_truth_image"]
                ground_truth_2 = batch["label"]
                ground_truth_3 = batch["centre"]
                input_data = batch["noisy_sinogram"]
                
                # make the predictions and calculate the validation loss
                pred_1 = self.model1(input_data)
                loss_value_1 = loss1(pred_1, ground_truth_1)
                pred_1_detached = pred_1.detach()

                # doing simultanious tasks
                # classifier
                pred_2 = self.model2(pred_1_detached)
                loss_value_2 = loss2(pred_2, ground_truth_2)

                # find the centre
                pred_3 = self.model3(pred_1_detached)
                loss_value_3 = loss3(pred_3, ground_truth_3)

                #update metrics
                self._update_metrics_state(
                    state, metrics_cfg,
                    pred_1, ground_truth_1,
                    pred_2, ground_truth_2,
                    pred_3, ground_truth_3,
                    data_range=1.0,
                    compute_train_mse=False)
                
                # add the loss to the total validation loss so far 
                B = ground_truth_1.shape[0]
                total_val_loss_1 += loss_value_1.item()*B
                total_val_loss_2 += loss_value_2.item()*B
                total_val_loss_3 += loss_value_3.item()*B
                n_samples += B

                # checking that predictions and ground truth images have same shape
                assert ground_truth_1.shape == pred_1.shape, f"[ERROR] Shape mismatch: predicted {pred_1.shape}, ground truth {ground_truth_1.shape}"

                
                 #Clean memory
                if batch_id % 10 == 0:
                    gc.collect()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()

                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()
        
        val_metrics = self._finalize_metrics_state(state, metrics_cfg)
        return total_val_loss_1, total_val_loss_2, total_val_loss_3, val_metrics, n_samples

        

    def train(self, training_path=None, validation_path=None, max_len_train = None, max_len_val=None, epochs=None):

        #changing paths parameters
        epochs = epochs or self.epochs
        self.training_path = training_path
        self.validation_path = validation_path
        self.max_len_train = max_len_train
        self.max_len_val = max_len_val

        # fix seed
        self._set_seed()


        # load training and validation datasets
        train_dataloader, val_dataloader = self._get_dataloaders()

        debug_train_loader = DataLoader(train_dataloader.dataset, batch_size=self.batch_size, shuffle=False)
        debug_val_loader   = DataLoader(val_dataloader.dataset,   batch_size=self.batch_size, shuffle=False)
        if self.debug:
            self.debug_loader(debug_train_loader, name="train", save_path="training_examples.png")
            self.debug_loader(debug_val_loader, name="val", save_path="validation_examples.png")

        # send models to device
        self.model1.to(self.device)
        self.model2.to(self.device)
        self.model3.to(self.device)
        
        ######################################## INVERSE PROBLEM ALGORITHM ########################################
        
        # Loss function
        self.loss_fn_1 = MSELoss()
        loss1 = self.loss_fn_1


        ######################################## CLASSIFIER ALGORITHM ########################################
        # Get only parameters that have requires_grad=True
        trainable_params_2 = [p for p in self.model2.parameters() if p.requires_grad]

        # Optimizer
        if self.optimizer_type_2 == "Adam":
            self.optimizer_2 = Adam(trainable_params_2, lr=self.learning_rate_2)
        elif self.optimizer_type_2 == "AdamW":
            self.optimizer_2 = AdamW(trainable_params_2, lr=self.learning_rate_2)
        elif self.optimizer_type_2 == "SGD":
            self.optimizer_2 = SGD(trainable_params_2, lr=self.learning_rate_2)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type_2}")

        # Loss function
        self.loss_fn_2 = CrossEntropyLoss()
        loss2 = self.loss_fn_2



        ######################################## FIND CENTER ALGORITHM ########################################
        # Get only parameters that have requires_grad=True
        trainable_params_3 = [p for p in self.model3.parameters() if p.requires_grad]

        # Optimizer
        if self.optimizer_type_3 == "Adam":
            self.optimizer_3 = Adam(trainable_params_3, lr=self.learning_rate_3)
        elif self.optimizer_type_3 == "AdamW":
            self.optimizer_3 = AdamW(trainable_params_3, lr=self.learning_rate_3)
        elif self.optimizer_type_3 == "SGD":
            self.optimizer_3 = SGD(trainable_params_3, lr=self.learning_rate_3)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type_3}")

        # Loss function
        self.loss_fn_3 = SmoothL1Loss(reduction="mean")
        loss3= self.loss_fn_3


        # accelerates training
        self.model1, train_dataloader, val_dataloader = self.accelerator.prepare(self.model1, train_dataloader, val_dataloader)
        self.model2, opt2 = self.accelerator.prepare(self.model2, self.optimizer_2)
        self.model3, opt3 = self.accelerator.prepare(self.model3, self.optimizer_3)
        
        # initialize scheduler for learning rate
        if self.scheduler_2:
            scheduler2 = ReduceLROnPlateau(opt2, mode='min', factor=0.5, patience=5)

        if self.scheduler_3:
            scheduler3 = ReduceLROnPlateau(opt3, mode='min', factor=0.5, patience=5)

        # initialize a dictionary to store training history
        history = {
            "train_loss_1": [],
            "val_loss_1": [],
            "train_loss_2": [],
            "val_loss_2": [],
            "train_loss_3": [],
            "val_loss_3": []
        }

        self._log(f"=== TRAIN START | epochs={epochs} | learning rates=({self.learning_rate_2},{self.learning_rate_3}) | batch size={self.batch_size} ===")

        t = tqdm(range(epochs), desc="Epochs")

        # loop over our epochs
        for e in t:

            # call the training function
            total_train_loss_1, total_train_loss_2, total_train_loss_3, train_metrics, n_train = self.train_one_epoch(train_dataloader, opt2, opt3, loss1, loss2, loss3, e)
            history.setdefault("train_metrics", []).append(train_metrics)

            # call the validation function
            total_val_loss_1, total_val_loss_2, total_val_loss_3, val_metrics, n_val = self.validation(val_dataloader, loss1, loss2, loss3, e)
            history.setdefault("val_metrics", []).append(val_metrics)

            # update our training history
            avg_train_loss_1 = total_train_loss_1 / n_train
            avg_train_loss_2 = total_train_loss_2 / n_train
            avg_train_loss_3 = total_train_loss_3 / n_train
            avg_val_loss_1 = total_val_loss_1 / n_val
            avg_val_loss_2 = total_val_loss_2 / n_val
            avg_val_loss_3 = total_val_loss_3 / n_val
            history["train_loss_1"].append(avg_train_loss_1)
            history["train_loss_2"].append(avg_train_loss_2)
            history["train_loss_3"].append(avg_train_loss_3)
            history["val_loss_1"].append(avg_val_loss_1)
            history["val_loss_2"].append(avg_val_loss_2)
            history["val_loss_3"].append(avg_val_loss_3)

            # save checkpoint every 5 epochs
            if (e+1)  % 5 == 0:
                self.save_checkpoint(epoch=e+1, history=history)
                self._log(f"Checkpoint saved at epoch {e+1}")

            # update the learning rate scheduler (when the validation loss is not improving)
        
            if self.scheduler_2:
                scheduler2.step(avg_val_loss_2)
            
            if self.scheduler_3:
                scheduler3.step(avg_val_loss_3)

            
        
            # print the model training and validation information
            t.set_postfix({
                "INVERSE MODEL": "",
                "train_loss_1": f"{avg_train_loss_1:.6f}",
                "val_loss_1": f"{avg_val_loss_1:.6f}",
                "CLASSIFIER MODEL": "",
                "train_loss_2": f"{avg_train_loss_2:.6f}",
                "val_loss_2": f"{avg_val_loss_2:.6f}",
                "CENTRE FINDER MODEL": "",
                "train_loss_3": f"{avg_train_loss_3:.6f}",
                "val_loss_3": f"{avg_val_loss_3:.6f}"
            })

            #logger
            lr2_now = opt2.param_groups[0]["lr"]
            lr3_now = opt3.param_groups[0]["lr"]
            self._log(
                f"[Epoch {e+1}/{epochs}] lr=({lr2_now:.3e},{lr3_now:.3e}) | "
                f"train_loss_1={avg_train_loss_1:.6f}, val_loss_1={avg_val_loss_1:.6f} | "
                f"train_loss_2={avg_train_loss_2:.6f}, val_loss_2={avg_val_loss_2:.6f} | "
                f"train_loss_3={avg_train_loss_3:.6f}, val_loss_3={avg_val_loss_3:.6f} | "
                f"val_metrics={val_metrics}"
            )

            #Clean memory
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()


        # Save final metrics
        with open(f'{self.model_path}_metrics.json', 'w') as f:
            json.dump(history, f)
        
        self._log(f"Final training history saved at {self.model_path}_metrics.json")
        self.save_checkpoint(epoch=epochs, history=history)
        self._log("Final checkpoint saved")
        self._log("=== TRAIN END ===")

        return history


    def evaluate(self, test_dataloader, loss1, loss2, loss3):

        # save ground truth and predictions
        gt_images, gt_labels, gt_centres = [],[],[]
        img_pred = []
        label_pred = []
        centre_pred = []
        sinograms = []

        # switch off autograd for evaluation
        with torch.no_grad():

            # set the model in evaluation mode
            self.model1.eval()
            self.model2.eval()
            self.model3.eval()
            
            # initialize metrics
            total_test_loss_1 = 0
            total_test_loss_2 = 0
            total_test_loss_3 = 0
            n_samples = 0

            #initilize metrics
            state, metrics_cfg = self._init_metric_state()

            # loop over the validation set
            for batch_id, batch in enumerate(tqdm(test_dataloader)):
                # send the input to the device
                ground_truth_1 = batch["ground_truth_image"]
                ground_truth_2 = batch["label"]
                ground_truth_3 = batch["centre"]
                input_data = batch["noisy_sinogram"]

                

                # make the predictions and calculate the validation loss
                pred_1 = self.model1(input_data)
                loss_value_1 = loss1(pred_1, ground_truth_1)

                # doing simultanious tasks
                # classifier
                pred_1_detached = pred_1.detach()
                pred_2 = self.model2(pred_1_detached)
                loss_value_2 = loss2(pred_2, ground_truth_2)

                # find the center
                pred_3 = self.model3(pred_1_detached)
                loss_value_3 = loss3(pred_3, ground_truth_3)

                #update metrics
                self._update_metrics_state(
                    state, metrics_cfg,
                    pred_1, ground_truth_1,
                    pred_2, ground_truth_2,
                    pred_3, ground_truth_3,
                    data_range=1.0,
                    compute_train_mse=False
                )
                
                # add the loss to the total validation loss so far
                B = ground_truth_1.shape[0]
                total_test_loss_1 += loss_value_1.item()*B
                total_test_loss_2 += loss_value_2.item()*B
                total_test_loss_3 += loss_value_3.item()*B
                n_samples += B

                # checking that predictions and ground truth images have same shape
                assert ground_truth_1.shape == pred_1.shape, f"[ERROR] Shape mismatch: predicted {pred_1.shape}, ground truth {ground_truth_1.shape}"

                # save gound truth and prediction
                img_pred.append(pred_1.cpu())
                label_pred.append(pred_2.cpu())
                centre_pred.append(pred_3.cpu())
                gt_images.append(ground_truth_1.cpu())
                gt_labels.append(ground_truth_2.cpu())
                gt_centres.append(ground_truth_3.cpu())
                sinograms.append(input_data.cpu())

                #Clean memory
                if batch_id % 10 == 0:
                    gc.collect()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()

                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()
        
        test_metrics = self._finalize_metrics_state(state, metrics_cfg)
        return img_pred, label_pred, centre_pred, gt_images, gt_labels, gt_centres, sinograms, total_test_loss_1, total_test_loss_2, total_test_loss_3, test_metrics, n_samples


    def test(self, test_path=None, max_len_test=None):

        self._log("=== TEST START ===")
        #changing paths parameters
        self.test_path = test_path
        self.max_len_test = max_len_test


        # fix seed for reproducibility
        self._set_seed()

        # get test data loader
        test_dataloader = self._get_test_dataloaders()

        debug_test_loader   = DataLoader(test_dataloader.dataset,   batch_size=self.batch_size, shuffle=False)
        if self.debug:
            self.debug_loader(debug_test_loader, name="test", n_show=5, save_path="test_examples.png")

        
        #initialice accelerator
        self.model1, test_dataloader = self.accelerator.prepare(self.model1, test_dataloader)
        self.model2 = self.accelerator.prepare(self.model2)
        self.model3 = self.accelerator.prepare(self.model3)

        loss1 = self.loss_fn_1
        loss2 = self.loss_fn_2
        loss3 = self.loss_fn_3
        

        # call evaliation function
        img_pred, label_pred, centre_pred, gt_images, gt_labels, gt_centres, sinograms, total_test_loss_1, total_test_loss_2, total_test_loss_3, test_metrics, n_test = self.evaluate(test_dataloader, loss1, loss2, loss3)
    
        # training history
        avg_test_loss_1 = total_test_loss_1 / n_test
        avg_test_loss_2 = total_test_loss_2 / n_test
        avg_test_loss_3 = total_test_loss_3 / n_test
        

        results = {
            "test_loss_1": avg_test_loss_1,
            "test_loss_2": avg_test_loss_2,
            "test_loss_3": avg_test_loss_3,
            "metrics": test_metrics
        }

        # save predictions and ground truth
        torch.save(torch.cat(img_pred), f"{self.model_path}_predictions_images.pt")
        torch.save(torch.cat(label_pred), f"{self.model_path}_predictions_labels.pt")
        torch.save(torch.cat(gt_labels), f"{self.model_path}_ground_truth_labels.pt")
        torch.save(torch.cat(gt_centres), f"{self.model_path}_ground_truth_centers.pt")
        torch.save(torch.cat(centre_pred), f"{self.model_path}_predictions_centers.pt")
        torch.save(torch.cat(gt_images), f"{self.model_path}_ground_truth_images.pt")
        torch.save(torch.cat(sinograms), f"{self.model_path}_noisy_sinograms.pt")

        # save metrics to file
        with open(f'{self.model_path}_test_metrics.json', 'w') as f:
            json.dump(results, f)

        self._log(
            f"test_loss_1={avg_test_loss_1:.6f} | "
            f"test_loss_2={avg_test_loss_2:.6f} | "
            f"test_loss_3={avg_test_loss_3:.6f} | "
            f"test_metrics={test_metrics}"
        )
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
        metrics_cfg = {
            "model1": ["mse", "psnr", "ssim", ],          # CT recon
            "model2": ["classifier_metrics"],    # classifier
            "model3": ["l2", "smoothl1"],        # center regression
        }

        state = {
            "model1": {"n_batches": 0,"mse_sum": 0.0, "psnr_sum": 0.0, "ssim_sum": 0.0, "n_samples": 0},
            "model2": {"n_batches": 0, "cm": torch.zeros(self.num_classes, self.num_classes, device=self.device, dtype=torch.int64)},
            "model3": {"n_batches": 0, "l2_sum": 0.0, "smoothl1_sum": 0.0, "n_samples": 0},
        }

        return state, metrics_cfg

    def _update_metrics_state(self, state, metrics_cfg, pred_1, gt_1, pred_2, gt_2, pred_3, gt_3, data_range=1.0, compute_train_mse=False):
        """
        Updates metric accumulators for a batch.
        """
        B = gt_1.shape[0]

        # model1 metrics (CT recon)
        if "model1" in metrics_cfg:
            for m in metrics_cfg["model1"]:
                if m == "psnr":
                    state["model1"]["psnr_sum"] += compute_psnr_results(pred_1, gt_1, max_val=data_range) * B
                elif m == "ssim":
                    state["model1"]["ssim_sum"] += compute_ssim(pred_1, gt_1, data_range=data_range) * B
                elif m == "mse":
                    state["model1"]["mse_sum"] += compute_mse(pred_1, gt_1) * B
            state["model1"]["n_samples"] += B

        # model2 metrics (classifier)
        if "model2" in metrics_cfg:
            for m in metrics_cfg["model2"]:
                if m == "classifier_metrics":
                    # accumulate confusion matrix on-device
                    state["model2"]["cm"] = update_confusion_matrix(state["model2"]["cm"], pred_2, gt_2, num_classes=self.num_classes)
            state["model2"]["n_batches"] += 1

        # model3 metrics (center regression)
        if "model3" in metrics_cfg:
            for m in metrics_cfg["model3"]:
                if m == "l2":
                    state["model3"]["l2_sum"] += ((pred_3-gt_3)**2).sum(dim=1).sqrt().sum().item()
                elif m == "smoothl1":
                    state["model3"]["smoothl1_sum"] += self.loss_fn_3(pred_3, gt_3).item() * B
            state["model3"]["n_samples"] += B


    def _finalize_metrics_state(self, state, metrics_cfg):
        """
        Converts accumulators into final metrics dict (means + classifier metrics).
        """
        results = {}

        # model1
        n1 = max(1, state["model1"]["n_samples"])
        results["model1"] = {}
        if "mse" in metrics_cfg["model1"]:
            results["model1"]["mse"] = state["model1"]["mse_sum"] / n1
        if "psnr" in metrics_cfg["model1"]:
            results["model1"]["psnr"] = state["model1"]["psnr_sum"] / n1
        if "ssim" in metrics_cfg["model1"]:
            results["model1"]["ssim"] = state["model1"]["ssim_sum"] / n1

        # model2
        results["model2"] = {}
        if "classifier_metrics" in metrics_cfg["model2"]:
            cm_cpu = state["model2"]["cm"].detach().cpu()
            results["model2"]["confusion_matrix"] = cm_cpu.numpy().tolist()  
            results["model2"].update(metrics_from_confusion_matrix(cm_cpu))

        # model3
        n3 = max(1, state["model3"]["n_samples"])
        results["model3"] = {}
        if "l2" in metrics_cfg["model3"]:
            results["model3"]["l2"] = state["model3"]["l2_sum"] / n3
        if "smoothl1" in metrics_cfg["model3"]:
            results["model3"]["smoothl1"] = state["model3"]["smoothl1_sum"] / n3

        return results

    ############################################################################ Checkpoints and save model ############################################################################

    def _get_config(self):
        return {
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
            "optimizer_type_2": self.optimizer_type_2,
            "optimizer_type_3": self.optimizer_type_3,
            "learning_rate_2": self.learning_rate_2,
            "learning_rate_3": self.learning_rate_3,
            "scheduler_2": self.scheduler_2,
            "scheduler_3": self.scheduler_3,
            "max_len_train": self.max_len_train,
            "max_len_val": self.max_len_val,
            "max_len_test": self.max_len_test,
            # útil para recrear exactamente el modelo
            "model_class_1": self.model1.__class__.__name__ if self.model1 is not None else None,
            "model_class_2": self.model2.__class__.__name__ if self.model2 is not None else None,
            "model_class_3": self.model3.__class__.__name__ if self.model3 is not None else None,
        }


    def save_checkpoint(self, epoch, history):

        # 1) config JSON
        cfg_path = f"{self.model_path}_config.json"
        with open(cfg_path, "w") as f:
            json.dump(self._get_config(), f, indent=2)

        # 2) checkpoint torch
        ckpt = {
            "epoch": epoch,
            "model1_state": self.accelerator.unwrap_model(self.model1).state_dict(),
            "model2_state": self.accelerator.unwrap_model(self.model2).state_dict(),
            "model3_state": self.accelerator.unwrap_model(self.model3).state_dict(),
            "optimizer2_state": self.optimizer_2.state_dict(),
            "optimizer3_state": self.optimizer_3.state_dict(),
            "history": history,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            }
        torch.save(ckpt, f"{self.model_path}_checkpoint.pt")


    def load_checkpoint(self, load_optimizer: bool = True):
        ckpt_path = f"{self.model_path}_checkpoint.pt"
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.model1.to(self.device); self.model2.to(self.device); self.model3.to(self.device)
        self.model1.load_state_dict(ckpt["model1_state"])
        self.model2.load_state_dict(ckpt["model2_state"])
        self.model3.load_state_dict(ckpt["model3_state"])

        if load_optimizer:
            if self.optimizer_2 is not None and ckpt.get("optimizer2_state") is not None:
                self.optimizer_2.load_state_dict(ckpt["optimizer2_state"])
            if self.optimizer_3 is not None and ckpt.get("optimizer3_state") is not None:
                self.optimizer_3.load_state_dict(ckpt["optimizer3_state"])

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

    def _extract_metric_series(self, metrics_list, model_key, metric_key, n_epochs):
        """metrics_list: history['train_metrics'] or history['val_metrics']"""
        out = []
        for i in range(n_epochs):
            if i < len(metrics_list):
                out.append(metrics_list[i].get(model_key, {}).get(metric_key, np.nan))
            else:
                out.append(np.nan)
        return out


    def plot_per_model_metrics(self, history, out_dir=None):
        """
        Crea:
        - 3 plots separados (model1/model2/model3) con métricas train/val por epoch
        - 1 plot conjunto (resumen) para comparar a la vez
        """
        if out_dir is None:
            out_dir = os.path.dirname(self.model_path) or "."
        os.makedirs(out_dir, exist_ok=True)

        train_m = history.get("train_metrics", [])
        val_m   = history.get("val_metrics", [])

        # número de epochs según lo que haya guardado
        n_epochs = max(len(train_m), len(val_m),
                    len(history.get("train_loss_1", [])),
                    len(history.get("val_loss_1", [])))
        if n_epochs == 0:
            return

        epochs = np.arange(1, n_epochs + 1)

        # Define qué quieres plotear por modelo
        model_cfg = {
            "model1": ["mse", "psnr", "ssim"],
            "model2": ["accuracy", "f1_macro"],   # ajusta a lo que devuelva metrics_from_confusion_matrix
            "model3": ["l2", "smoothl1"],
        }

        # --------- 1) Plots separados por modelo ----------
        for model_key, metric_keys in model_cfg.items():
            plt.figure()
            for mk in metric_keys:
                y_tr = self._extract_metric_series(train_m, model_key, mk, n_epochs)
                y_va = self._extract_metric_series(val_m,   model_key, mk, n_epochs)

                plt.plot(epochs, y_tr, label=f"train_{mk}")
                plt.plot(epochs, y_va, label=f"val_{mk}")

            plt.xlabel("epoch")
            plt.ylabel("metric")
            plt.title(f"{model_key}: train vs val metrics")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{os.path.basename(self.model_path)}_{model_key}_metrics.png"), dpi=200)
            plt.close()

        # --------- 2) Plot conjunto (resumen) ----------
        # Para que no sea ilegible: elige 1-2 métricas “top” por modelo
        summary = [
            ("model1", "psnr"),
            ("model1", "ssim"),
            ("model2", "accuracy"),
            ("model3", "l2"),
        ]

        plt.figure()
        for model_key, mk in summary:
            y_tr = self._extract_metric_series(train_m, model_key, mk, n_epochs)
            y_va = self._extract_metric_series(val_m,   model_key, mk, n_epochs)
            plt.plot(epochs, y_tr, label=f"{model_key}_train_{mk}")
            plt.plot(epochs, y_va, label=f"{model_key}_val_{mk}")

        plt.xlabel("epoch")
        plt.ylabel("metric")
        plt.title("Summary metrics (train vs val)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{os.path.basename(self.model_path)}_ALL_models_summary.png"), dpi=200)
        plt.close()
    
    def plot_test_per_model(self, out_dir=None, normalize_cm=True):
        if out_dir is None:
            out_dir = os.path.dirname(self.model_path) or "."
        os.makedirs(out_dir, exist_ok=True)

        with open(f"{self.model_path}_test_metrics.json", "r") as f:
            results = json.load(f)


        # Carga tensores guardados en test()
        pred_img = torch.load(f"{self.model_path}_predictions_images.pt", map_location="cpu")   # (N,1,H,W)
        gt_img   = torch.load(f"{self.model_path}_ground_truth_images.pt", map_location="cpu")

        pred_logits = torch.load(f"{self.model_path}_predictions_labels.pt", map_location="cpu")  # (N,C)
        gt_labels   = torch.load(f"{self.model_path}_ground_truth_labels.pt", map_location="cpu") # (N,)

        pred_cent = torch.load(f"{self.model_path}_predictions_centers.pt", map_location="cpu")   # (N,2)
        gt_cent   = torch.load(f"{self.model_path}_ground_truth_centers.pt", map_location="cpu")  # (N,2)

        

        # ---------------------------
        # (1) Model2: UNA confusion matrix
        # ---------------------------
        cm = np.asarray(results["metrics"]["model2"]["confusion_matrix"], dtype=np.float32)
        if normalize_cm:
            cm_plot = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1e-12)
            title = "Test confusion matrix (row-normalized)"
        else:
            cm_plot = cm
            title = "Test confusion matrix"

        plt.figure(figsize=(6, 5))
        plt.imshow(cm_plot, aspect="auto")
        plt.colorbar()
        plt.xlabel("pred")
        plt.ylabel("gt")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{os.path.basename(self.model_path)}_test_model2_cm.png"), dpi=200)
        plt.close()

        # ---------------------------
        # (2) Model1: violin de métricas por muestra (MSE + PSNR)
        # ---------------------------
        # MSE por muestra
        mse_i = ((pred_img - gt_img)**2).mean(dim=(1,2,3))
        eps = 1e-12
        psnr_i = 10.0 * torch.log10((1.0 ** 2) / mse_i.clamp_min(eps))
        ssim_out = []
        for i in range(len(pred_img)):
            ssim_out_i = ssim(pred_img[i], gt_img[i], data_range=1.0)
            ssim_out.append(ssim_out_i)
        ssim_i = torch.Tensor(ssim_out)
        if isinstance(ssim_out, torch.Tensor) and ssim_out.ndim == 0:
            # es escalar -> NO hay violin real
            ssim_i = None
        else:
            ssim_i = ssim_out

        plt.figure(figsize=(9, 8))
        data = [mse_i.numpy(), psnr_i.numpy()]
        labels = ["mse", "psnr"]
        if ssim_i is not None:
            data.insert(1, ssim_i.numpy())
            labels.insert(1, "ssim")
        plt.figure(figsize=(7, 4))
        plt.violinplot(data, showmeans=True, showmedians=True)
        plt.xticks(np.arange(1, len(labels)+1), labels)
        plt.title("Model1 test metrics (per-sample)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{os.path.basename(self.model_path)}_test_model1_violin.png"), dpi=200)
        plt.close()

        # ---------------------------
        # (3) Model3: violin de errores por muestra (L2 + SmoothL1)
        # ---------------------------
        l2_i = torch.norm(pred_cent - gt_cent, dim=1).numpy()

        # SmoothL1 por muestra (por coordenada y luego media)
        smoothl1_i = F.smooth_l1_loss(pred_cent, gt_cent, reduction="none").sum(dim=1).numpy()

        plt.figure(figsize=(6, 4))
        data = [l2_i, smoothl1_i]
        plt.violinplot(data, showmeans=True, showmedians=True)
        plt.xticks([1,2], ["l2", "smoothl1"])
        plt.title("Model3 test metrics (per-sample)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{os.path.basename(self.model_path)}_test_model3_violin.png"), dpi=200)
        plt.close()