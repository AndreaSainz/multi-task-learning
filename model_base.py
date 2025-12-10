import tomosipo as ts
import torch
from torch.nn import MSELoss, L1Loss, CrossEntropyLoss, Module
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_loader import MnistDataset
from torch.utils.data import DataLoader, random_split
import time
from tqdm import tqdm
import json
import random
import numpy as np
import os
import gc
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



class ModelBase(Module):
    
    def __init__(self, model_path, dimension, num_angles, sigma, standard_dose, cut, seed, batch_size, epochs, optimizer_type, learning_rate, debug,  accelerator,  scheduler = (True,True,True)):
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
        self.learning_rate_1, self.learning_rate_2, self.learning_rate_3 = learning_rate
        self.seed = seed
        self.debug = debug
        self.optimizer_type_1, self.optimizer_type_2, self.optimizer_type_3 = optimizer_type
        self.optimizer_1 = None
        self.optimizer_2 = None
        self.optimizer_3 = None
        self.loss_fn_1 = None
        self.loss_fn_2 = None
        self.loss_fn_3 = None
        self.scheduler_1, self.scheduler_2, self.scheduler_3 = scheduler
        

        # accelerator for faster code
        self.accelerator = accelerator
        self.trainable_params_1 = None
        self.trainable_params_2 = None
        self.trainable_params_3 = None

        # set the device once for the whole class
        self.device = self.accelerator.device

        # Create tomosipo volume and projection geometry
        self.vg = ts.volume(shape=(1,self.H,self.W))                            # Volumen
        self.angles = np.linspace(0, np.pi, self.num_angles, endpoint=True)     # Angles
        self.pg = ts.parallel(angles=self.angles, shape=(1, self.H))            # Parallel beam structure
        self.A = ts.operator(self.vg,self.pg)         
   


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
            n_train = n_val_t

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





    def train_one_epoch(self, train_dataloader, opt1, opt2, opt3, loss1, loss2, loss3, e):

        # set the model in training mode
        self.model1.train()
        self.model2.train()
        self.model3.train()

        # initialize the total training loss
        total_train_loss_1 = 0
        total_train_loss_2 = 0
        total_train_loss_3 = 0


        # loop over the training set
        for batch_id, batch in enumerate(train_dataloader):

            # send the input to the device
            ground_truth_1 = batch["ground_truth_image"]
            print(f"Ground Truth Image Shape: {ground_truth_1.shape}")
            ground_truth_2 = batch["label"]
            ground_truth_3 = batch["center"]
            input_data = batch["noisy_sinogram"]


            # perform a forward pass and calculate the training loss
            #pred_1 = ground_truth_1.clone()
            pred_1 = self.model1(input_data)
            loss_value_1 = loss1(pred_1, ground_truth_1)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(input_data[0, 0].detach().cpu().numpy(), cmap="gray")
            axes[0].set_title("Sinograma")

            axes[1].imshow(pred_1[0, 0].detach().cpu().numpy(), cmap="gray")
            axes[1].set_title("Reconstrucción pred_1")

            axes[2].imshow(ground_truth_1[0, 0].detach().cpu().numpy(), cmap="gray")
            axes[2].set_title("Ground truth")

            for ax in axes:
                ax.axis("off")

            plt.tight_layout()
            plt.savefig(f"debug_epoch{e}_batch{batch_id}.png")
            plt.close(fig)


            # doing simultanious tasks
            pred_1_detached = pred_1.detach() ###### ESTO SOLO DE MOMENTO PORQUE QUEREMOS QUE LOS MODELOS SEAN INDEPENDIENTES, ESTO CAMBIARÁ
            # classifier
            pred_2 = self.model2(pred_1_detached)
            loss_value_2 = loss2(pred_2, ground_truth_2)

            print(f"Original label: {ground_truth_2}, Predicted label = {pred_2}")

            # find the center
            pred_3 = self.model3(pred_1_detached)
            pred_center = pred_3.round().long()  # cause they are pixel positions
            loss_value_3 = loss3(pred_center, ground_truth_3)

            print(f"Original center: {ground_truth_3}, Predicted center = {pred_center}")

            # checking that predictions and ground truth images have same shape
            assert ground_truth_1.shape == pred_1_detached.shape, f"[ERROR] Shape mismatch: predicted {pred_1_detached.shape}, ground truth {ground_truth_1.shape}"

            # zero out the gradients, perform the backpropagation step, and update the weights
            opt1.zero_grad()
            self.accelerator.backward(loss_value_1)
            opt1.step()

            opt2.zero_grad()
            self.accelerator.backward(loss_value_2)
            opt2.step()


            opt3.zero_grad()
            self.accelerator.backward(loss_value_3)
            opt3.step()

            # add the loss to the total training loss so far
            total_train_loss_1 += loss_value_1.item()
            total_train_loss_2 += loss_value_2.item()
            total_train_loss_3 += loss_value_3.item()

            #Clean memory
            del input_data, ground_truth_1, ground_truth_2, ground_truth_3, pred_1, pred_2, pred_3, loss_value_1, loss_value_2, loss_value_3
            if batch_id % 10 == 0:
                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        return total_train_loss_1, total_train_loss_2, total_train_loss_3


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
            

            # loop over the validation set
            for batch_id, batch in enumerate(tqdm(val_dataloader, desc=f"Validation Epoch {e+1}", leave=False)):

                # send the input to the device
                ground_truth_1 = batch["ground_truth_image"]
                ground_truth_2 = batch["label"]
                ground_truth_3 = batch["center"]
                input_data = batch["noisy_sinogram"]
                
                # make the predictions and calculate the validation loss
                pred_1 = self.model1(input_data)
                loss_value_1 = loss1(pred_1, ground_truth_1)

                # doing simultanious tasks
                # classifier
                pred_2 = self.model2(pred_1)
                loss_value_2 = loss2(pred_2, ground_truth_2)

                # find the center
                pred_3 = self.model3(pred_1)
                loss_value_3 = loss3(pred_3, ground_truth_3)


                # add the loss to the total validation loss so far
                total_val_loss_1 += loss_value_1.item()
                total_val_loss_2 += loss_value_2.item()
                total_val_loss_3 += loss_value_3.item()

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
        
        return total_val_loss_1, total_val_loss_2, total_val_loss_3

        

    def train(self, training_path, validation_path, max_len_train = None, max_len_val=None, epochs=None):

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

        # show model summary
        self.model1.to(self.device)
        self.model2.to(self.device)
        self.model3.to(self.device)
        
        ######################################## INVERSE PROBLEM ALGORITHM ########################################
        # Get only parameters that have requires_grad=True
        trainable_params_1 = [p for p in self.model1.parameters() if p.requires_grad]

        # Optimizer
        if self.optimizer_type_1 == "Adam":
            self.optimizer_1 = Adam(trainable_params_1, lr=self.learning_rate_1)
        elif self.optimizer_type_1 == "AdamW":
            self.optimizer_1 = AdamW(trainable_params_1, lr=self.learning_rate_1)
        elif self.optimizer_type_1 == "SGD":
            self.optimizer_1 = SGD(trainable_params_1, lr=self.learning_rate_1)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type_1}")

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
        self.loss_fn_3 = MSELoss()
        loss3= self.loss_fn_3


        # accelerates training
        self.model1, opt1, train_dataloader, val_dataloader = self.accelerator.prepare(self.model1, self.optimizer_1, train_dataloader, val_dataloader)
        self.model2, opt2 = self.accelerator.prepare(self.model2, self.optimizer_2)
        self.model3, opt3 = self.accelerator.prepare(self.model3, self.optimizer_3)
        
        # initialize scheduler for learning rate
        if self.scheduler_1:
            scheduler1 = ReduceLROnPlateau(opt1, mode='min', factor=0.5, patience=5)
        
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

        
        t = tqdm(range(epochs), desc="Epochs")

        # loop over our epochs
        for e in t:

            # call the training function
            total_train_loss_1, total_train_loss_2, total_train_loss_3 = self.train_one_epoch(train_dataloader, opt1, opt2, opt3, loss1, loss2, loss3, e)

            # call the validation function
            total_val_loss_1, total_val_loss_2, total_val_loss_3 = self.validation(val_dataloader, loss1, loss2, loss3, e)

            # update our training history
            avg_train_loss_1 = total_train_loss_1 / len(train_dataloader)
            avg_train_loss_2 = total_train_loss_2 / len(train_dataloader)
            avg_train_loss_3 = total_train_loss_3 / len(train_dataloader)
            avg_val_loss_1 = total_val_loss_1 / len(val_dataloader)
            avg_val_loss_2 = total_val_loss_2 / len(val_dataloader)
            avg_val_loss_3 = total_val_loss_3 / len(val_dataloader)
            history["train_loss_1"].append(avg_train_loss_1)
            history["train_loss_2"].append(avg_train_loss_2)
            history["train_loss_3"].append(avg_train_loss_3)
            history["val_loss_1"].append(avg_val_loss_1)
            history["val_loss_2"].append(avg_val_loss_2)
            history["val_loss_3"].append(avg_val_loss_3)


            # update the learning rate scheduler (when the validation loss is not improving)
            if self.scheduler_1:
                scheduler1.step(avg_val_loss_1)
            
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
                "CENTER FINDER MODEL": "",
                "train_loss_3": f"{avg_train_loss_3:.6f}",
                "val_loss_3": f"{avg_val_loss_3:.6f}"
            })

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

        return history


    def evaluate(self, test_dataloader, loss1, loss2, loss3):

        # save ground truth and predictions
        gt_images = []
        img_pred = []
        label_pred = []
        center_pred = []
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

            # loop over the validation set
            for batch_id, batch in enumerate(tqdm(test_dataloader)):
                # send the input to the device
                ground_truth_1 = batch["ground_truth_image"]
                ground_truth_2 = batch["label"]
                ground_truth_3 = batch["center"]
                input_data = batch["noisy_sinogram"]

                

                # make the predictions and calculate the validation loss
                pred_1 = self.model1(input_data)
                loss_value_1 = loss1(pred_1, ground_truth_1)

                # doing simultanious tasks
                # classifier
                pred_2 = self.model2(pred_1)
                loss_value_2 = loss2(pred_2, ground_truth_2)

                # find the center
                pred_3 = self.model3(pred_1)
                loss_value_3 = loss3(pred_3, ground_truth_3)


                # add the loss to the total validation loss so far
                total_test_loss_1 += loss_value_1.item()
                total_test_loss_2 += loss_value_2.item()
                total_test_loss_3 += loss_value_3.item()

                # checking that predictions and ground truth images have same shape
                assert ground_truth_1.shape == pred_1.shape, f"[ERROR] Shape mismatch: predicted {pred_1.shape}, ground truth {ground_truth_1.shape}"

                

                # save gound truth and prediction
                img_pred.append(pred_1.cpu())
                label_pred.append(pred_2.cpu())
                center_pred.append(pred_3.cpu())
                gt_images.append(ground_truth_1.cpu())
                sinograms.append(input_data.cpu())

                #Clean memory
                if batch_id % 10 == 0:
                    gc.collect()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()

                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()
        
        return img_pred, label_pred, center_pred, gt_images, sinograms, total_test_loss_1, total_test_loss_2, total_test_loss_3


    def test(self, test_path, max_len_test=None,):

        #changing paths parameters
        self.test_path = test_path
        self.max_len_test = max_len_test


        # fix seed for reproducibility
        self._set_seed()

        # get test data loader
        test_dataloader = self._get_test_dataloaders()
        

        #initialice accelerator
        self.model1, test_dataloader = self.accelerator.prepare(self.model1, test_dataloader)
        self.model2 = self.accelerator.prepare(self.model2)
        self.model3 = self.accelerator.prepare(self.model3)

        loss1 = self.loss_fn_1
        loss2 = self.loss_fn_2
        loss3 = self.loss_fn_3
        

        # call evaliation function
        img_pred, label_pred, center_pred, gt_images, sinograms, total_test_loss_1, total_test_loss_2, total_test_loss_3 = self.evaluate(test_dataloader, loss1, loss2, loss3)
    
        # training history
        avg_test_loss_1 = total_test_loss_1 / len(test_dataloader)
        avg_test_loss_2 = total_test_loss_2 / len(test_dataloader)
        avg_test_loss_3 = total_test_loss_3 / len(test_dataloader)
        

        results = {
            "test_loss_1": avg_test_loss_1,
            "test_loss_2": avg_test_loss_2,
            "test_loss_3": avg_test_loss_3
        }

        # save predictions and ground truth
        torch.save(torch.cat(img_pred), f"{self.model_path}_predictions_images.pt")
        torch.save(torch.cat(label_pred), f"{self.model_path}_predictions_labels.pt")
        torch.save(torch.cat(center_pred), f"{self.model_path}_predictions_centers.pt")
        torch.save(torch.cat(gt_images), f"{self.model_path}_ground_truth_images.pt")
        torch.save(torch.cat(sinograms), f"{self.model_path}_noisy_sinograms.pt")

        # save metrics to file
        with open(f'{self.model_path}_test_metrics.json', 'w') as f:
            json.dump(results, f)

        return results