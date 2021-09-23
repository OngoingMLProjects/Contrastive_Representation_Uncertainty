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
from Contrastive_uncertainty.general.callbacks.ood_callbacks import get_roc_sklearn

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



# Performs 1D typicality using the nearest neighbours as the batch for the data
class NearestNeighbours1DTypicality(NearestNeighbours):
    def __init__(self, Datamodule,OOD_Datamodule,
        quick_callback:bool = True):

        super().__init__(Datamodule,OOD_Datamodule, quick_callback)
        # Used to save the summary value
        self.summary_key = f'Normalized One Dim Marginal Typicality KNN - {self.K} OOD - {self.OOD_Datamodule.name}'

    def forward_callback(self, trainer, pl_module):
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
        return super().get_features(pl_module, dataloader)
    
    def normalise(self, ftrain, ftest, food):
        return super().normalise(ftrain, ftest, food)
    
    def get_nearest_neighbours(self, ftest, food):
        num_ID = len(ftest)
        collated_features = np.concatenate((ftest, food))
        # Make a distance matrix for the data
        distance_matrix = scipy.spatial.distance.cdist(collated_features, collated_features)
        bottom_k_indices = np.argsort(distance_matrix,axis=1)[:,:self.K] # shape (num samples,k) , each row has the k indices with the smallest values (including itself), so it gets K indices for each data point
        return bottom_k_indices


    # calculate the std of the 1d likelihood scores as well
    def get_1d_train(self, ftrain):
        # Nawid - get all the features which belong to each of the different classes
        cov = np.cov(ftrain.T, bias=True) # Cov and means part should be fine
        mean = np.mean(ftrain,axis=0,keepdims=True) # Calculates mean from (B,embdim) to (1,embdim)
        
        eigvalues, eigvectors = np.linalg.eigh(cov)
        eigvalues = np.expand_dims(eigvalues,axis=1)
        
        dtrain = np.matmul(eigvectors.T,(ftrain - mean).T)**2/eigvalues
        # calculate the mean and the standard deviations of the different values
        dtrain_1d_mean = np.mean(dtrain,axis= 1,keepdims=True) # shape (dim,1)
        dtrain_1d_std = np.std(dtrain,axis=1,keepdims=True) # shape (dim,1)
        
        #normalised_dtrain = (dtrain - dtrain_1d_mean)
        
        # Value for a particular class
        # Vector of datapoints(embdim,num_eigenvectors) (each column is eigenvector so the different columns is the number of eigenvectors)
        # data - means is shape (B, emb_dim), therefore the matrix multiplication needs to be (num_eigenvectors, embdim), (embdim,batch) to give (num eigenvectors, Batch) and then this is divided by (num eigenvectors,1) 
        # to give (num eigen vectors, batch) different values for 1 dimensional mahalanobis distances 
        
        # I believe the first din is the list of size class, where each entry is a vector of size (emb dim, batch) where each entry of the embed dim is the 1 dimensional mahalanobis distance along that dimension, so a vector of (embdim,1) represents the mahalanobis distance of each of the n dimensions for that particular data point

        #Get entropy based on training data (or could get entropy using the validation data)
        return mean, cov, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std
    

    # get the features of the data which also has the KNN in either the test set or the OOD dataset
    def get_knn_features(self,ftest,food, knn_indices):
        num_ID = len(ftest)
        # collated features to store all the different values for the data 
        collated_features = np.concatenate((ftest, food))
        knn_collated_features = collated_features[knn_indices] # shape (num_samples_collated, K, embeddim)
        knn_ID_features = knn_collated_features[:num_ID] # get all the features and neighbours for the ID data 
        knn_ID_features = np.reshape(knn_ID_features,(knn_ID_features.shape[0]*knn_ID_features.shape[1],knn_ID_features.shape[2]))
        
        knn_OOD_features = knn_collated_features[num_ID:] # get all the features and neighbours for the OOD data
        knn_OOD_features = np.reshape(knn_OOD_features,(knn_OOD_features.shape[0]*knn_OOD_features.shape[1],knn_OOD_features.shape[2]))


        # Need to check with a toy example whether the different numbers are placed sequentially in a batch
        return knn_ID_features, knn_OOD_features

    def get_thresholds(self, fdata, mean, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std):
        thresholds = [] # List of threshold values
        num_batches = len(fdata)//self.K

        for i in range(num_batches):
            fdata_batch = fdata[(i*self.K):((i+1)*self.K)]
            ddata = np.matmul(eigvectors.T,(fdata_batch - mean).T)**2/eigvalues  # shape (dim, batch size)
            # Normalise the data
            ddata = (ddata - dtrain_1d_mean)/(dtrain_1d_std +1e-10) # shape (dim, batch)

            # shape (dim) average of all data in batch size
            ddata = np.mean(ddata,axis= 1) # shape : (dim)
            
            # Sum of the deviations of each individual dimension
            ddata_deviation = np.sum(np.abs(ddata))

            thresholds.append(ddata_deviation)
        
        return thresholds

    def get_scores(self,ftrain, ftest, food):
        # Get information related to the train info
        mean, cov, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std = self.get_1d_train(ftrain)
        # Inference
        # Calculate the scores for the in-distribution data and the OOD data
        knn_indices = self.get_nearest_neighbours(ftest,food)
        knn_ID_features, knn_OOD_features = self.get_knn_features(ftest, food, knn_indices)

        din = self.get_thresholds(knn_ID_features, mean, eigvalues,eigvectors, dtrain_1d_mean,dtrain_1d_std)
        dood = self.get_thresholds(knn_OOD_features, mean, eigvalues,eigvectors, dtrain_1d_mean,dtrain_1d_std)    

        return din, dood


    def get_eval_results(self, ftrain, ftest, food):
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain, ftest, food)
        din, dood = self.get_scores(ftrain_norm,ftest_norm, food_norm)
        AUROC = get_roc_sklearn(din, dood)
        wandb.run.summary[self.summary_key] = AUROC

    