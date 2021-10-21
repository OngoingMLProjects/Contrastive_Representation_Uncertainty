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


# NEED TO BE ABLE TO PERFORM BACKPROPAGATION EFFECTIVELY
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
        import ipdb; ipdb.set_trace()
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


       
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int, dataloader_idx: int):
        (img_1, img_2), *labels, indices = batch = batch
        if isinstance(labels, tuple) or isinstance(labels, list):
            labels, *coarse_labels = labels
        torch.set_grad_enabled(True) # need to set grad enabled true during the test step
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

        #import ipdb; ipdb.set_trace()
       
    



    # Performs all the computation in the callback
    def forward_callback(self,trainer,pl_module):
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA

        train_loader = self.Datamodule.deterministic_train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()
        
        # Obtain representations of the data
            
        features_train, labels_train = self.get_features(pl_module, train_loader)
        features_test, labels_test = self.get_features(pl_module, test_loader)
        features_ood, labels_ood = self.get_features(pl_module, ood_loader)
        # Number of classes obtained from the max label value + 1 ( to take into account counting from zero)
        '''
        self.get_eval_results(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood),
            np.copy(labels_train))
        '''

    def get_features(self, pl_module, dataloader):
        features, labels = [], []
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
            
            # Compute feature vector and place in list
            feature_vector = pl_module.callback_vector(img) # Performs the callback for the desired level
            logits = pl_module.encoder.class_fc2(feature_vector)
            prediction_values, prediction_labels= torch.max(logits,dim=1)
            
            loss = torch.nn.CrossEntropyLoss()(logits,prediction_labels)
            #import ipdb; ipdb.set_trace()
            pl_module.manual_backward(loss)
            
            
            features += list(feature_vector.data.cpu().numpy())
            labels += list(label.data.cpu().numpy())            

        return np.array(features), np.array(labels)
    
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
    
    def normalise(self,ftrain,ftest,food):
        # Nawid -normalise the featues for the training, test and ood data
        # standardize data
        
        ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10
        ftest /= np.linalg.norm(ftest, axis=-1, keepdims=True) + 1e-10
        food /= np.linalg.norm(food, axis=-1, keepdims=True) + 1e-10
        # Nawid - calculate the mean and std of the traiing features
        m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)
        # Nawid - normalise data using the mean and std
        ftrain = (ftrain - m) / (s + 1e-10)
        ftest = (ftest - m) / (s + 1e-10)
        food = (food - m) / (s + 1e-10)
        
        return ftrain, ftest,food