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
from sklearn.neighbors import KernelDensity

from Contrastive_uncertainty.general.utils.ood_utils import get_measures # Used to calculate the AUROC, FPR and AUPR
from Contrastive_uncertainty.general.utils.hybrid_utils import OOD_conf_matrix
#from Contrastive_uncertainty.Contrastive.models.loss_functions import class_discrimination
from Contrastive_uncertainty.general.callbacks.general_callbacks import quickloading
from Contrastive_uncertainty.general.utils.pl_metrics import precision_at_k, mean


# Based on this implementation - https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
class KDE_OOD(pl.Callback):
    def __init__(self, Datamodule,OOD_Datamodule,
        quick_callback:bool = True):

        super().__init__()
        self.Datamodule = Datamodule
        self.OOD_Datamodule = OOD_Datamodule
        self.OOD_Datamodule.test_transforms = self.Datamodule.test_transforms #  Make the transform of the OOD data the same as the actual data
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly
        
      
        self.OOD_dataname = self.OOD_Datamodule.name
        self.summary_key = f'KDE AUROC OOD {self.OOD_dataname}'
        self.summary_aupr = self.summary_key.replace("AUROC", "AUPR")
        self.summary_fpr = self.summary_key.replace("AUROC", "FPR")

    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer=trainer, pl_module=pl_module)


    def forward_callback(self,trainer,pl_module):
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA

        train_loader = self.Datamodule.deterministic_train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()
        
        # Obtain representations of the data
        
        features_train, labels_train = self.get_features(pl_module, train_loader)
        features_test, labels_test = self.get_features(pl_module, test_loader)
        features_ood, labels_ood = self.get_features(pl_module, ood_loader)
        #self.visualize_data(trainer,pl_module, ood_loader)
        
        # Number of classes obtained from the max label value + 1 ( to take into account counting from zero)

        self.get_eval_results(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood))
    
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
            
            features += list(feature_vector.data.cpu().numpy())
            labels += list(label.data.cpu().numpy())            

        return np.array(features), np.array(labels)

    
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

    def get_eval_results(self,ftrain, ftest, food):
        """
            None.
        """
        ftrain_norm,ftest_norm,food_norm = self.normalise(ftrain,ftest,food)
        # Nawid - obtain the scores for the test data and the OOD data
        
        kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
        kde.fit(ftrain_norm)

        # score_samples returns the log of the probability density
        dtest = kde.score_samples(ftest_norm)
        dood = kde.score_samples(food_norm)

        # Nawid- get false postive rate and asweel as AUROC and aupr
        # Nawid - Use the negative scores to make it similar to the mahalanobis distance as the higher the KDE scores, the better it is, whilst higher the mahalanobis scores,the worse it is        
        auroc, aupr, fpr = get_measures(-dood,-dtest)

        #auroc= get_roc_sklearn(dtest, dood)
        wandb.run.summary[self.summary_key] = auroc
        wandb.run.summary[self.summary_aupr] = aupr
        wandb.run.summary[self.summary_fpr] = fpr