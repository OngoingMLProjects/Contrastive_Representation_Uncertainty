from matplotlib.axis import Axis
from numpy.core.numeric import indices
from numpy.lib.function_base import average
import torch
#from torch._C import per_tensor_affine
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
from plotly.subplots import SubplotRef, make_subplots
from sklearn.metrics import roc_auc_score

from Contrastive_uncertainty.general.utils.ood_utils import get_measures # Used to calculate the AUROC, FPR and AUPR
from Contrastive_uncertainty.general.utils.hybrid_utils import OOD_conf_matrix
#from Contrastive_uncertainty.Contrastive.models.loss_functions import class_discrimination
from Contrastive_uncertainty.general.callbacks.general_callbacks import quickloading
from Contrastive_uncertainty.general.utils.pl_metrics import precision_at_k
from Contrastive_uncertainty.general.callbacks.ood_callbacks import get_roc_sklearn
from Contrastive_uncertainty.general.callbacks.nearest_neighbours_callbacks import NearestNeighboursQuadraticClass1DTypicality

################## Analysis - Used to save the individual scores ######################### 
# Saving the individual scores for the approach
# Performs 1D typicality using a quadratic summation using the nearest neighbours as the batch for the data, and obtaining specific classes for the data

class AnalysisQuadraticClass1DScoresTypicality(NearestNeighboursQuadraticClass1DTypicality):
    def __init__(self, Datamodule,OOD_Datamodule,
        quick_callback:bool = True,K:int = 10):

        super().__init__(Datamodule,OOD_Datamodule, quick_callback,K)
        # Used to save the summary value
        self.K = K
        self.summary_key = f'Analysis Normalized One Dim Scores Class Quadratic Typicality KNN - {self.K} OOD - {self.OOD_Datamodule.name}'

    def forward_callback(self, trainer, pl_module):
        return super().forward_callback(trainer, pl_module)
        
    def get_features(self, pl_module, dataloader):
        return super().get_features(pl_module, dataloader)
    
    def normalise(self, ftrain, ftest, food):
        return super().normalise(ftrain, ftest, food)
    
    def get_nearest_neighbours(self, ftest, food):
        return super().get_nearest_neighbours(ftest, food)
    
    # get the features of the data which also has the KNN in either the test set or the OOD dataset
    def get_knn_features(self, ftest, food, knn_indices):
        return super().get_knn_features(ftest, food, knn_indices)
    
    def get_1d_train(self, ftrain, ypred):
        return super().get_1d_train(ftrain, ypred)
    
    # Look at obtaining the OOD datast for the situation which is the worse case
    def get_thresholds(self, fdata, means, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std):    
        
        collated_class_dimesional_scores = [[] for i in range(self.Datamodule.num_classes)]
        num_batches = len(fdata)//self.K
        # Currently goes through a single data point at a time which is not very efficient
        for i in range(num_batches):
            fdata_batch = fdata[(i*self.K):((i+1)*self.K)]
            # Added additional constant to the eignvalue for the purpose of numerical stability
            # Perturbation to prevent numerical error
            perturbations = [(np.sign(eigvalues[class_num])*1e-10) for class_num in range(len(means))]
            # Make any zero value into 1e-10
            for class_num in range(len(means)):
                perturbations[class_num][perturbations[class_num]==0] = 1e-10
            #perturbations[class_num][perturbations[class_num] == 0] = 1e-10 for class_num in range(len(means))]
            #[perturbations[class_num][perturbations[class_num] == 0] = 1e-10 for class_num in range(len(means))]# Replace any zero values with 1e-10 (this is done as np.sign is zero when the specific class eigvalue is zero)
            ddata = [np.matmul(eigvectors[class_num].T,(fdata_batch - means[class_num]).T)**2/(eigvalues[class_num] + perturbations[class_num]) for class_num in range(len(means))] 
            #ddata = [np.matmul(eigvectors[class_num].T,(fdata_batch - means[class_num]).T)**2/(eigvalues[class_num] + (np.sign(eigvalues[class_num])*1e-10)) for class_num in range(len(means))] # Calculate the 1D scores for all the different classes 
            #
            # obtain the normalised the scores for the different classes
            ddata_deviation = [ddata[class_num] - dtrain_1d_mean[class_num]/(dtrain_1d_std[class_num]  +1e-10) for class_num in range(len(means))] # shape (dim, batch)
            # shape (dim) average of all data in batch size
            ddata_mean_deviation = [np.mean(ddata_deviation[class_num],axis=1) for class_num in range(len(means))] # shape dim
            ###### Only change from the previous class is this line ###########
            # Obtain the sum of absolute normalised scores squared (to emphasis the larger penalty when the deviation of a dimension is larger)
            scores = [np.sum(np.abs(ddata_mean_deviation[class_num]**2),axis=0) for class_num in range(len(means))]
            ##################################################################
            # Obtain the 1d scores corresponding to the lowest class
            
            
            class_val = np.argmin(scores,axis=0) # get the specific clas of interest for the particular approach
            class_dimensional_scores = ddata_mean_deviation[np.argmin(scores,axis=0)]
            # Get the dimensional scores for the particular class
            collated_class_dimesional_scores[class_val].append(class_dimensional_scores)
        
        return collated_class_dimesional_scores

    def get_scores(self,ftrain, ftest, food,labelstrain):
        # Get information related to the train info
        mean, cov, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std = self.get_1d_train(ftrain,labelstrain)
        # Inference
        # Calculate the scores for the in-distribution data and the OOD data
        knn_indices = self.get_nearest_neighbours(ftest,food)
        knn_ID_features, knn_OOD_features = self.get_knn_features(ftest, food, knn_indices)
        
        # Gets a list of the different scores for the case of the ID and OOD dataset for the different class of the ID dataset
        din = self.get_thresholds(knn_ID_features, mean, eigvalues,eigvectors, dtrain_1d_mean,dtrain_1d_std)
        dood = self.get_thresholds(knn_OOD_features, mean, eigvalues,eigvectors, dtrain_1d_mean,dtrain_1d_std)    
        
        
        #in_class_counts = [len(din[class_val]) for class_val in range(self.Datamodule.num_classes)]
        #ood_class_counts = [len(dood[class_val]) for class_val in range(self.Datamodule.num_classes)]
        
        #- Find the numbers of datapoints in the ID and OOD classes
		#- Remove any classes which have zero from the OOD dataset

        in_class_counts = np.array([len(din[class_val]) for class_val in range(self.Datamodule.num_classes)])
        ood_class_counts = np.array([len(dood[class_val]) for class_val in range(self.Datamodule.num_classes)])
        
        in_class_counts[ood_class_counts ==0] = 0 # Change the in-distribution values to only choose the values which are in common between th edifferent datasets 
        # find the INTERSECTION BTWEEN THE IN AND Ood CLASS COUNTS
        
		#- Choose the 3 classes which have the deviation related to the different approach
        return din, dood, in_class_counts, eigvalues

    def datasaving(self,din, dood,in_class_counts,eigvalues,wandb_name):
        # obtain the top 3 indices of the in-class counts in order, https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        # There could be a situation where there is no overlap between the 2 different groups
        # In that situation I should get the values which correspond to the particular value of interest, I should save all the values for all the different classes for the OOD and ID datasets
        #import ipdb; ipdb.set_trace()
        idx = (-in_class_counts).argsort()[:3]
        
        #idx = (-arr).argsort()[:n]
        for k,i in enumerate(idx):
            if in_class_counts[i] ==0: # Checks whether there are any data points belonging to the class
                average_in_deviations, average_ood_deviations = np.zeros((eigvalues[0].shape[0],)), np.zeros((eigvalues[0].shape[0],))
            else:
                average_in_deviations, average_ood_deviations = np.mean(din[i],axis=0), np.mean(dood[i],axis=0)
            
            #table_data = {'Dimension':[], 'OOD-ID Deviation':[], 'Eigvalues':[]}
        
            OOD_ID_Deviation = average_ood_deviations - average_in_deviations
            Dimensions = np.arange(0,OOD_ID_Deviation.shape[0])
            
            class_eigvalues = np.array(eigvalues[i]).squeeze() #  squeeze used to get from a shape of (128,1) to (128,) so it is the shme shape as the other arrays which it gets stacked with
            table_data = np.stack((Dimensions,OOD_ID_Deviation,class_eigvalues),axis=1)

            table_df = pd.DataFrame(table_data)
            table_df.columns = ['Dimension', 'OOD-ID Deviation','Eigvalues']
            table = wandb.Table(dataframe=table_df)
            # Need to not override wandb name overwise it will keep on adding top k to each subsequent name
            log_name = wandb_name + f' Top {k+1}'
            wandb.log({log_name:table})
            
    '''
    def datasaving(self,din, dood,wandb_name):
        average_in_deviations, average_ood_deviations = np.mean(din,axis=0), np.mean(dood,axis=0)
        table_data = {'Dimension':[], 'OOD-ID Deviation':[]}
        
        OOD_ID_Deviation = average_ood_deviations - average_in_deviations
        Dimensions = np.arange(0,OOD_ID_Deviation.shape[0])
        table_data = np.stack((Dimensions,OOD_ID_Deviation),axis=1)
        table_df = pd.DataFrame(table_data)

        table_df.columns = ['Dimension', 'OOD-ID Deviation']
        table = wandb.Table(dataframe=table_df)
        wandb.log({wandb_name:table})
    ''' 
    def get_eval_results(self, ftrain, ftest, food,labelstrain):
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain, ftest, food)
        # Get the 1d scores for each class in the ID and OOD datase tas well as the counts of which class is being categorized
        din, dood, in_class_counts, eigvalues = self.get_scores(ftrain_norm,ftest_norm, food_norm,labelstrain)
        # get the average deviations for the in-distribution and OOD data
        self.datasaving(din,dood,in_class_counts,eigvalues,self.summary_key)

# Look at separating the analysis into the individual classes to get a better idea
# Obtain the classes of all the different classifications of the data
# Separate the scores into th edifferent classes
# Look at the datasets where there is both ID and OOD data classified into the class)
# LOok at the deviation belonging to a particular class.
