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


from Contrastive_uncertainty.general.utils.hybrid_utils import OOD_conf_matrix
#from Contrastive_uncertainty.Contrastive.models.loss_functions import class_discrimination
from Contrastive_uncertainty.general.callbacks.general_callbacks import quickloading
from Contrastive_uncertainty.general.utils.pl_metrics import precision_at_k, mean
from Contrastive_uncertainty.general.run.model_names import model_names_dict

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
    
    '''   
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int, dataloader_idx: int):
        (img_1, img_2), *labels, indices = batch = batch
        if isinstance(labels, tuple) or isinstance(labels, list):
            labels, *coarse_labels = labels
        pl_module.eval()
        self.loss_calculation(trainer, pl_module,img_1, labels)
    
    # Second working version of calculating the gradients, able to perturb the input and calculate all the different values present
    def loss_calculation(self,trainer, pl_module,x,y):
        
        x_params = nn.Parameter(x)
        
        
        x_params, y = x_params.to(pl_module.device), y.to(pl_module.device)
        x_params.retain_grad() # required to obtain the gradient otherwise the UserWarning : UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the gradient for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations.

        logits = pl_module.class_forward(x_params)
        
        probs = F.softmax(logits,dim=1)
        labels = torch.argmax(probs,dim=1) # use the maximum probability indices as the labels 
        loss = nn.CrossEntropyLoss()(probs, labels)
        loss.backward()
        print(x_params.grad)
        x = x.to(pl_module.device)
        perturbed_x = torch.add(x,x_params.grad ,alpha=0.01) # adding x with x grad in conjuction with an alpha term to get the different values
        perturbed_outputs = pl_module.class_forward(perturbed_x)
        
       
    '''

    ''' First working version of calculating the gradients
    def loss_calculation(self,trainer, pl_module,x,y):
        
        x_params = nn.Parameter(x)
        
        
        x_params, y = x_params.to(pl_module.device), y.to(pl_module.device)
        x_params.retain_grad() # required to obtain the gradient otherwise the UserWarning : UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the gradient for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations.

        logits = pl_module.class_forward(x_params)

        probs = F.softmax(logits,dim=1)
        loss = nn.CrossEntropyLoss()(probs, y)
        loss.backward()
        print(x_params.grad)
    '''

    '''
    def on_test_epoch_end(self, trainer, pl_module):
        # Perform callback only for the situation
        if pl_module.name == model_names_dict['CE']:
            pl_module.automatic_optimization= False
            torch.set_grad_enabled(True)
            self.forward_callback(trainer=trainer, pl_module=pl_module) 
        else:
            pass
    '''


    '''       
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int, dataloader_idx: int):
        (img_1, img_2), *labels, indices = batch = batch
        if isinstance(labels, tuple) or isinstance(labels, list):
            labels, *coarse_labels = labels
        torch.set_grad_enabled(True) # need to set grad enabled true during the test step
        pl_module.eval()
        self.loss_calculation(trainer, pl_module,img_1, labels)
    '''

    def on_test_epoch_end(self, trainer, pl_module):
        # Perform callback only for the situation
        torch.set_grad_enabled(True) # need to set grad enabled true during the test step
        pl_module.eval()
        self.forward_callback(trainer,pl_module)
    
    # Performs all the computation in the callback
    def forward_callback(self,trainer,pl_module):
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA

        train_loader = self.Datamodule.deterministic_train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()
        
        # Obtain representations of the data
            
        self.get_grads(trainer,pl_module, train_loader)
        self.get_grads(trainer, pl_module, test_loader)
        self.get_grads(trainer, pl_module, ood_loader)
        # Number of classes obtained from the max label value + 1 ( to take into account counting from zero)
        
    def get_grads(self,trainer,pl_module,dataloader):
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
            
            self.loss_calculation(trainer, pl_module, img, label)


        return img
    
    # Second working version of calculating the gradients, able to perturb the input and calculate all the different values present
    def loss_calculation(self,trainer, pl_module,x,y):
        x_params = nn.Parameter(x)
        
        x_params, y = x_params.to(pl_module.device), y.to(pl_module.device)
        x_params.retain_grad() # required to obtain the gradient otherwise the UserWarning : UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the gradient for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations.

        logits = pl_module.class_forward(x_params)
        
        probs = F.softmax(logits,dim=1)


        labels = torch.argmax(probs,dim=1) # use the maximum probability indices as the labels 
        loss = nn.CrossEntropyLoss()(probs, labels)
        loss.backward()
        '''
        ############ Normalize gradient ##############
        
        gradient = torch.ge(x_params.grad,0)
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[::, 0] = (gradient[::, 0] )/(63.0/255.0)
        gradient[::, 1] = (gradient[::, 1] )/(62.1/255.0)
        #gradient[::, 2] = (gradient[::, 2] )/(66.7/255.0)

        x = x.to(pl_module.device)
        tempInputs = torch.add(x, gradient, alpha=0.01)

        import ipdb; ipdb.set_trace()


        ############
        ''' 
        print(x_params.grad)
        x = x.to(pl_module.device)

        perturbed_x = torch.add(x,x_params.grad ,alpha=0.01) # adding x with x grad in conjuction with an alpha term to get the different values
        perturbed_outputs = pl_module.class_forward(perturbed_x)

       
    '''
    def get_scores(self,ftrain, ftest, food, ypred):
        # Nawid - get all the features which belong to each of the different classes
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        
        din = [
            np.sum(
                (ftest - np.mean(x, axis=0, keepdims=True)) # Nawid - distance between the data point and the mean
                * (
                    np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                        (ftest - np.mean(x, axis=0, keepdims=True)).T
                    ) # Nawid - calculating the covariance matrix of the data belonging to a particular class and dot product by the distance of the data point from the mean (distance calculation)
                ).T,
                axis=-1,
            )
            for x in xc # Nawid - done for all the different classes
        ]
        
        dood = [
            np.sum(
                (food - np.mean(x, axis=0, keepdims=True))
                * (
                    np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                        (food - np.mean(x, axis=0, keepdims=True)).T
                    )
                ).T,
                axis=-1,
            )
            for x in xc # Nawid- this calculates the score for all the OOD examples 
        ]
        # Calculate the indices corresponding to the values
        indices_din = np.argmin(din,axis = 0)
        indices_dood = np.argmin(dood, axis=0)

        din = np.min(din, axis=0) # Nawid - calculate the minimum distance 
        dood = np.min(dood, axis=0)

        return din, dood, indices_din, indices_dood
    

        
    def get_eval_results(self,ftrain, ftest, food, labelstrain):
        """
            None.
        """
        ftrain_norm,ftest_norm,food_norm = self.normalise(ftrain,ftest,food)
        # Nawid - obtain the scores for the test data and the OOD data
        
        dtest, dood, indices_dtest, indices_dood = self.get_scores(ftrain_norm, ftest_norm, food_norm, labelstrain)

        # Nawid- get false postive rate and asweel as AUROC and aupr
        auroc= get_roc_sklearn(dtest, dood)
        wandb.run.summary[self.summary_key] = auroc
    '''  
