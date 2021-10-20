from numpy.core.numeric import indices
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
from Contrastive_uncertainty.general.callbacks.ood_callbacks import get_roc_sklearn
from Contrastive_uncertainty.general.run.model_names import model_names_dict


class Max_Softmax_Probability(pl.Callback):
    def __init__(self, Datamodule,OOD_Datamodule,
        quick_callback:bool = True):

        super().__init__()
        self.Datamodule = Datamodule
        self.OOD_Datamodule = OOD_Datamodule
        self.OOD_Datamodule.test_transforms = self.Datamodule.test_transforms #  Make the transform of the OOD data the same as the actual data
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly
        
      
        self.OOD_dataname = self.OOD_Datamodule.name
        self.summary_key = f'Maximum Softmax Probability AUROC OOD {self.OOD_dataname}'
    
    '''
    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip if fast testing as this can lead to issues with the code
        self.forward_callback(trainer=trainer, pl_module=pl_module) 
    '''
    
    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer=trainer, pl_module=pl_module) 

    # Performs all the computation in the callback
    def forward_callback(self,trainer,pl_module):
        assert pl_module.name == model_names_dict['CE'], 'Incorrect Model, not CE Model'
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA

        train_loader = self.Datamodule.deterministic_train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()
        
        # Obtain representations of the data
        
        #prob_train, labels_train = self.get_probs(pl_module, train_loader)
        prob_test, labels_test = self.get_probs(pl_module, test_loader)
        prob_ood, labels_ood = self.get_probs(pl_module, ood_loader)
        
        # Number of classes obtained from the max label value + 1 ( to take into account counting from zero)
        
        self.get_eval_results(
            np.copy(prob_test),
            np.copy(prob_ood))
        

    def get_probs(self, pl_module, dataloader):
        prob_scores, labels = [], []
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
            
            # Compute probablities for the different classes and place in a list
            logits = pl_module.class_forward(img)
            probs = F.softmax(logits,dim=1)
            prob_scores += list(probs.data.cpu().numpy())
            labels += list(label.data.cpu().numpy())            

        return np.array(prob_scores), np.array(labels)
    


    def get_scores(self,prob_test, prob_ood):
        # Nawid - get all the features which belong to each of the different classes
        
        
        din = np.max(prob_test,axis=1)
        dood = np.max(prob_ood,axis=1)

        indices_din = np.argmax(prob_test,axis=1)
        indices_dood = np.argmax(prob_ood,axis=1)

        return din, dood, indices_din, indices_dood
    

        
    def get_eval_results(self, prob_test, prob_ood):
        """
            None.
        """
        # Nawid - obtain the scores for the test data and the OOD data
        
        dtest, dood, indices_dtest, indices_dood = self.get_scores(prob_test, prob_ood)

        # Nawid- get false postive rate and asweel as AUROC and aupr
        auroc= get_roc_sklearn(dtest, dood)
        wandb.run.summary[self.summary_key] = auroc
        
    