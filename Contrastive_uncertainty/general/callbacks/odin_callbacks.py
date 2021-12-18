from numpy.core.numeric import indices
from numpy.lib.financial import _ipmt_dispatcher
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import os
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wandb
import sklearn.metrics as skm
import faiss
import statistics 


import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score

from Contrastive_uncertainty.general.utils.ood_utils import get_measures # Used to calculate the AUROC, FPR and AUPR
from Contrastive_uncertainty.general.utils.hybrid_utils import OOD_conf_matrix
#from Contrastive_uncertainty.Contrastive.models.loss_functions import class_discrimination
from Contrastive_uncertainty.general.callbacks.general_callbacks import quickloading
from Contrastive_uncertainty.general.utils.pl_metrics import precision_at_k, mean
from Contrastive_uncertainty.general.run.model_names import model_names_dict
from Contrastive_uncertainty.general.callbacks.ood_callbacks import get_roc_sklearn

# Implementation based on these resources
# https://github.com/facebookresearch/odin/blob/main/code/calData.py
#https://github.com/guyera/Generalized-ODIN-Implementation/blob/master/code/cal.py
class ODIN(pl.Callback):
    def __init__(self, Datamodule,OOD_Datamodule,
        quick_callback:bool = True,Temperature:float = 1000, noise_magnitude:float = 0.0014):

        super().__init__()
        self.Datamodule = Datamodule
        self.OOD_Datamodule = OOD_Datamodule
        self.OOD_Datamodule.test_transforms = self.Datamodule.test_transforms #  Make the transform of the OOD data the same as the actual data
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly
        
        
        self.OOD_dataname = self.OOD_Datamodule.name
        self.summary_key = f'ODIN AUROC OOD {self.OOD_dataname}'
        self.summary_aupr = self.summary_key.replace("AUROC", "AUPR")
        self.summary_fpr = self.summary_key.replace("AUROC", "FPR") 
        
        # https://github.com/facebookresearch/odin/blob/main/code/main.py - default parameters
        self.temperature = 1000
        #self.epsilon = 0.0014

        self.minimum_FPR = 1.0 # Highest possible FPR
        self.optimal_episilon = 0.0

    
    def on_test_epoch_end(self, trainer, pl_module):
        # Perform callback only for the situation
        torch.set_grad_enabled(True) # need to set grad enabled true during the test step
        pl_module.eval()
        self.hyperparameter_tune(trainer,pl_module)
        self.forward_callback(trainer,pl_module)
    
    def hyperparameter_tune(self,trainer,pl_module):
        self.OOD_Datamodule.setup()
        test_loader = self.Datamodule.test_dataloader()
        val_ood_loader = self.OOD_Datamodule.val_dataloader()
        if isinstance(val_ood_loader, tuple) or isinstance(val_ood_loader, list):
            _, val_ood_loader = val_ood_loader # separate for the case of the train val loader and the test val loader
        
        epsilon_values = np.linspace(0,0.004,5) # otain 5 values to examine between the intervals
        for epsilon in epsilon_values: # iterate through the epsilon values
            dtest = self.get_perturbed_scores(trainer,pl_module,test_loader,epsilon)
            dood_val = self.get_perturbed_scores(trainer,pl_module, val_ood_loader,epsilon)
            self.get_hyperparameter_eval_results(dtest, dood_val,epsilon)

    # Performs all the computation in the callback
    def forward_callback(self,trainer,pl_module):
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA

        train_loader = self.Datamodule.deterministic_train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()
        
        # Obtain representations of the data
            
        #self.get_perturbed_scores(trainer,pl_module, train_loader)
        dtest = self.get_perturbed_scores(trainer, pl_module, test_loader,self.optimal_episilon)
        dood = self.get_perturbed_scores(trainer, pl_module, ood_loader,self.optimal_episilon)
        self.get_eval_results(dtest,dood)

    # Scores when the inputs are not perturbed by a gradient    
    def get_unperturbed_scores(self,trainer,pl_module,dataloader):
        collated_scores = []
        loader = quickloading(self.quick_callback, dataloader)
        for index, (img, *label, indices) in enumerate(loader):
            assert len(loader)>0, 'loader is empty'
            if isinstance(img, tuple) or isinstance(img, list):
                    img, *aug_img = img # Used to take into accoutn whether the data is a tuple of the different augmentations

            # Selects the correct label based on the desired label level
            if len(label) > 1:
                label_index = 0
                label = label[label_index]
            else: # Used for the case of the OOD data
                label = label[0]
            img = img.to(pl_module.device)
            
            ##### MAY NEED TO SUBTRACT LARGEST VALUE TO STABILISE IT 
            logits = pl_module.class_forward(img)
            max_logits,_ = torch.max(logits,dim=1)
            max_logits = max_logits.unsqueeze(dim=1)
            logits = logits - max_logits
            #######
            
            probs = F.softmax(logits,dim=1)

            scores, _ = torch.max(probs,dim=1) # use the maximum logit value as the scores
            
        return np.array(collated_scores)


    def get_perturbed_scores(self,trainer,pl_module,dataloader,epsilon):
        collated_scores = []
        loader = quickloading(self.quick_callback, dataloader)
        for index, (img, *label, indices) in enumerate(loader):
            assert len(loader)>0, 'loader is empty'
            if isinstance(img, tuple) or isinstance(img, list):
                    img, *aug_img = img # Used to take into accoutn whether the data is a tuple of the different augmentations

            # Selects the correct label based on the desired label level
            if len(label) > 1:
                label_index = 0
                label = label[label_index]
            else: # Used for the case of the OOD data
                label = label[0]
            img = img.to(pl_module.device)

            scores = self.odin_scoring(trainer, pl_module, img,epsilon)
            collated_scores += list(scores.data.cpu().numpy())
        return collated_scores
    
    # Second working version of calculating the gradients, able to perturb the input and calculate all the different values present
    def odin_scoring(self, trainer, pl_module, x,epsilon):
        
        grad = self.get_grads(trainer,pl_module,x)
        # Perturb input
        perturbed_inputs = torch.add(x, grad, alpha=epsilon)
        perturbed_logits = pl_module.class_forward(perturbed_inputs)
        perturbed_logits = perturbed_logits/self.temperature
        
        ####
        # MAY NEED TO SUBTRACT LARGEST VALUE TO STABILISE IT 
        max_logits,_ = torch.max(perturbed_logits,dim=1)
        max_logits = max_logits.unsqueeze(dim=1) # need to change shape from (b) to (b,1)
        
        
        perturbed_logits = perturbed_logits - max_logits # shape (b,num classes ) and (b,1)
        #####
        
        probs = F.softmax(perturbed_logits,dim=1)
        scores, _ = torch.max(probs,dim=1)
        return scores

    # Calculates gradient for perturbation
    def get_grads(self,trainer,pl_module,x):
        # Calculate gradient
        x_params = nn.Parameter(x)
        #x_params, y = x_params.to(pl_module.device), y.to(pl_module.device)
        x_params.retain_grad() # required to obtain the gradient otherwise the UserWarning : UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the gradient for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations.

        logits = pl_module.class_forward(x_params)
        logits = logits/self.temperature     
        #probs = F.softmax(logits,dim=1)

        labels = torch.argmax(logits,dim=1) # use the maximum probability indices as the labels 
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        
        # normalize grad
        grad = self.normalize_grad(x_params.grad)
        return grad

    # Used to normalize the gradient to the space of images
    def normalize_grad(self,grad):
        # Can use normalization of the ID dataset for both the ID and OOD data since the OOD data has the same transform
        normalization = self.Datamodule.test_transforms.normalization
        normalization_mean = normalization.mean
        # corresponds to the toy dataset case
        grad = torch.ge(grad,0)
        grad = (grad.float() - 0.5) * 2
        if len(normalization_mean) ==0:
            grad[::, 0] = (grad[::, 0] )/1
            grad[::, 1] = (grad[::, 1] )/1
        # corresponds to the grayscale datasets    
        elif len(normalization_mean) == 1:
            grad[::, 0] = (grad[::, 0] )/normalization_mean[0]
            #grad[::, 1] = (grad[::, 1] )/normalization_mean[0]
            #grad[::, 2] = (grad[::, 2] )/normalization_mean[0]
        # correspond to rgb datasets
        else:
            grad[::, 0] = (grad[::, 0] )/normalization_mean[0]
            grad[::, 1] = (grad[::, 1] )/normalization_mean[1]
            grad[::, 2] = (grad[::, 2] )/normalization_mean[2]
        
        return grad

    def get_eval_results(self,dtest, dood):
        """
            None.
        """
        # Nawid- get false postive rate and asweel as AUROC and aupr
        auroc, aupr, fpr = get_measures(dood,dtest)
        wandb.run.summary[self.summary_key] = auroc
        wandb.run.summary[self.summary_aupr] = aupr
        wandb.run.summary[self.summary_fpr] = fpr

    # updates the minimum fpr and the optimal epsilon (hyperparameter tuning)
    def get_hyperparameter_eval_results(self,dtest, dood_val,epsilon):
        """
            None.
        """
        # Nawid- get false postive rate and asweel as AUROC and aupr
        auroc, aupr, fpr = get_measures(dood_val,dtest)
        if fpr < self.minimum_FPR:
            self.minimum_FPR = fpr
            self.optimal_episilon = epsilon