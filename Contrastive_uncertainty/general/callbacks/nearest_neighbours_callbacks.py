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
import scipy
import math
import random

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score


from Contrastive_uncertainty.general.utils.hybrid_utils import OOD_conf_matrix
#from Contrastive_uncertainty.Contrastive.models.loss_functions import class_discrimination
from Contrastive_uncertainty.general.callbacks.general_callbacks import quickloading
from Contrastive_uncertainty.general.utils.pl_metrics import precision_at_k


class NearestNeighbours(pl.Callback):
    def __init__(self, Datamodule,OOD_Datamodule,
        quick_callback:bool = True):

        super().__init__()
        self.Datamodule = Datamodule
        self.OOD_Datamodule = OOD_Datamodule
        self.OOD_Datamodule.test_transforms = self.Datamodule.test_transforms #  Make the transform of the OOD data the same as the actual data
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly
        
      
        self.OOD_dataname = self.OOD_Datamodule.name
        self.K = 5

    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer=trainer, pl_module=pl_module) 


    def forward_callback(self,trainer,pl_module):
        train_loader = self.Datamodule.deterministic_train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()

        features_train, labels_train = self.get_features(pl_module, train_loader)
        features_test, labels_test = self.get_features(pl_module, test_loader)
        features_ood, labels_ood = self.get_features(pl_module, ood_loader)

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
        
        return ftrain, ftest, food    

    def get_nearest_neighbours(self,ftest, food):
        num_ID = len(ftest)
        collated_features = np.concatenate((ftest,food))
        # Makes a distance matrix for the data
        distance_matrix = scipy.spatial.distance.cdist(collated_features, collated_features)
        
        bottom_k_indices = np.argsort(distance_matrix,axis=1)[:,1:self.K+1] # shape (num samples,k) , each row has the k indices with the smallest values, use 1 :k as 0th value is the distance to itself which is zero
        outlier_neighbours = np.where(bottom_k_indices <num_ID,0,1) # if the value is below num ID, then the value becomes zero which shows the kth neighbour is an inlier, if the value is above num ID, then it is an outlier
        outlier_percentage = np.mean(outlier_neighbours,axis=1)

        ID_outlier_percentage = outlier_percentage[:num_ID]
        OOD_outlier_percentage = outlier_percentage[num_ID:]
        
        ID_outlier_percentage = np.expand_dims(ID_outlier_percentage,axis=1)
        OOD_outlier_percentage = np.expand_dims(OOD_outlier_percentage,axis=1)
        
        #collated_features = torch.from_numpy(collated_features)
        #class_dist = torch.pdist(collated_features, p=2).pow(2).mean()
        return ID_outlier_percentage, OOD_outlier_percentage
        
    def datasaving(self,ID_outlier_percentage, OOD_outlier_percentage,wandb_name):
        ID_outlier_percentage_df = pd.DataFrame(ID_outlier_percentage)
        OOD_outlier_percentage_df = pd.DataFrame(OOD_outlier_percentage)
        concatenated_outlier_percentage_df = pd.concat((ID_outlier_percentage_df,OOD_outlier_percentage_df),axis=1)
        concatenated_outlier_percentage_df = concatenated_outlier_percentage_df.applymap(lambda l: l if not np.isnan(l) else random.uniform(-2,-1))
        concatenated_outlier_percentage_df.columns = ['ID KNN Outlier Percentage', 'OOD KNN Outlier Percentage']
        table_data = wandb.Table(data= concatenated_outlier_percentage_df)
        wandb.log({wandb_name:table_data})
    
    def get_eval_results(self,ftrain, ftest, food):
        """
            None.
        """
        ftrain_norm,ftest_norm,food_norm = self.normalise(ftrain,ftest,food)
        # Nawid - obtain the scores for the test data and the OOD data
        ID_outlier_percentage, OOD_outlier_percentage = self.get_nearest_neighbours(ftest_norm, food_norm)
        name = f'K:{self.K} NN Outlier Percentage OOD: {self.OOD_dataname}'
        self.datasaving(ID_outlier_percentage, OOD_outlier_percentage,name)


    

