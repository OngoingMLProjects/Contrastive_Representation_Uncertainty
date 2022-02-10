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

class NearestNeighbours(pl.Callback):
    def __init__(self, Datamodule,OOD_Datamodule,
        quick_callback:bool = True, K:int=10):

        super().__init__()
        self.Datamodule = Datamodule
        self.OOD_Datamodule = OOD_Datamodule
        self.OOD_Datamodule.test_transforms = self.Datamodule.test_transforms #  Make the transform of the OOD data the same as the actual data
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly
              
        self.OOD_dataname = self.OOD_Datamodule.name
        self.K = K

    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer=trainer, pl_module=pl_module) 

    def forward_callback(self,trainer,pl_module):
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA

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


# Checks how many of the neighbours (including itself) belong to the same class
class NearestClassNeighbours(pl.Callback):
    def __init__(self, Datamodule,
        quick_callback:bool = True, K:int= 10):

        super().__init__()
        self.Datamodule = Datamodule
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly      
        self.K = K

    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer=trainer, pl_module=pl_module) 

    def forward_callback(self,trainer,pl_module):
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA

        train_loader = self.Datamodule.deterministic_train_dataloader()
        test_loader = self.Datamodule.test_dataloader()

        features_train, labels_train = self.get_features(pl_module, train_loader)
        features_test, labels_test = self.get_features(pl_module, test_loader)

        self.get_eval_results(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(labels_test))

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

    def normalise(self,ftrain,ftest):
        # Nawid -normalise the featues for the training, test and ood data
        # standardize data
        
        ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10
        ftest /= np.linalg.norm(ftest, axis=-1, keepdims=True) + 1e-10
        # Nawid - calculate the mean and std of the traiing features
        m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)
        # Nawid - normalise data using the mean and std
        ftrain = (ftrain - m) / (s + 1e-10)
        ftest = (ftest - m) / (s + 1e-10)
        
        return ftrain, ftest

    # only get the nearest neighbours belonging to the same class
    def get_nearest_neighbours(self,ftest,labelstest):
        # Makes a distance matrix for the data
        distance_matrix = scipy.spatial.distance.cdist(ftest, ftest)
        
        bottom_k_indices = np.argsort(distance_matrix,axis=1)[:,:self.K] # shape (num samples,k), get the k nearest neighbours including itself 
        # Get the labels for the KNN
        knn_labels = labelstest[bottom_k_indices] 
        # https://stackoverflow.com/questions/61712303/row-or-column-wise-most-frequent-elements-in-2-d-numpy-array - Calculate the nearest neighbours for the different values
        values, count = scipy.stats.mode(knn_labels,1)
        normalized_count = count/self.K
        
        return normalized_count
        
    def datasaving(self,normalized_count,wandb_name):
        normalized_count_df = pd.DataFrame(normalized_count)
        normalized_count_df.columns = ['ID KNN Class Fraction']

        table_data = wandb.Table(data = normalized_count_df)
        wandb.log({wandb_name:table_data})
    
    def get_eval_results(self,ftrain, ftest, labelstest):
        """
            None.
        """
        ftrain_norm, ftest_norm = self.normalise(ftrain,ftest)
        # Nawid - obtain the scores for the test data and the OOD data
        normalized_count = self.get_nearest_neighbours(ftest_norm,labelstest)
        name = f'K:{self.K} NN Class Fraction'
        self.datasaving(normalized_count,name)



# Performs 1D typicality using the nearest neighbours as the batch for the data
class NearestNeighbours1DTypicality(NearestNeighbours):
    def __init__(self, Datamodule,OOD_Datamodule,
        quick_callback:bool = True,K:int = 10):

        super().__init__(Datamodule,OOD_Datamodule, quick_callback,K)
        # Used to save the summary value
        self.summary_key = f'Normalized One Dim Marginal Typicality KNN - {self.K} OOD - {self.OOD_Datamodule.name}'
        self.summary_aupr = f'Normalized One Dim Marginal Typicality KNN - {self.K} AUPR OOD - {self.OOD_Datamodule.name}'
        self.summary_fpr = f'Normalized One Dim Marginal Typicality KNN - {self.K} FPR OOD - {self.OOD_Datamodule.name}'


    def forward_callback(self, trainer, pl_module):
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA
        
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
        
        # Used to prevent numerical error
        perturbation = (np.sign(eigvalues)*1e-10)
        perturbation[perturbation==0] = 1e-10 # Replace any zero values with 1e-10 (this is done as np.sign is zero when the specific class eigvalue is zero)


        dtrain = np.matmul(eigvectors.T,(ftrain - mean).T)**2/(eigvalues + perturbation)
        
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
        
        '''
        a3D = np.array([[[1, 2], [3, 4]],
                    [[5, 6], [7, 8]]])
        reshaped_array = np.reshape(a3D,(a3D.shape[0]*a3D.shape[1],a3D.shape[2])) 
        
        array = np.arange(27)
        basic_reshape = np.reshape(array,(3,3,3))
        flatten_reshape = np.reshape(basic_reshape,(9,3))
        

        array = np.arange(18)
        basic_reshape = np.reshape(array,(3,3,2))
        flatten_reshape = np.reshape(basic_reshape,(9,2))
        '''    
        # Need to check with a toy example whether the different numbers are placed sequentially in a batch
        return knn_ID_features, knn_OOD_features

    def get_thresholds(self, fdata, mean, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std):
        thresholds = [] # List of threshold values
        num_batches = len(fdata)//self.K

        for i in range(num_batches):
            fdata_batch = fdata[(i*self.K):((i+1)*self.K)]

            perturbation = (np.sign(eigvalues)*1e-10)
            perturbation[perturbation==0] = 1e-10 # Replace any zero values with 1e-10 (this is done as np.sign is zero when the specific class eigvalue is zero)
            ddata = np.matmul(eigvectors.T,(fdata_batch - mean).T)**2/(eigvalues + perturbation)  # shape (dim, batch size)

            
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
        
        auroc, aupr, fpr = get_measures(dood,din)
        
        wandb.run.summary[self.summary_key] = auroc
        wandb.run.summary[self.summary_aupr] = aupr
        wandb.run.summary[self.summary_fpr] = fpr

        '''
        AUROC = get_roc_sklearn(din, dood)
        wandb.run.summary[self.summary_key] = AUROC
        '''


# Performs 1D typicality using the nearest neighbours as the batch for the data, and obtaining specific classes for the data
class NearestNeighboursClass1DTypicality(NearestNeighbours1DTypicality):
    def __init__(self, Datamodule,OOD_Datamodule,
        quick_callback:bool = True,K:int = 10):

        super().__init__(Datamodule,OOD_Datamodule, quick_callback,K)
        # Used to save the summary value
        self.K = K
        self.summary_key = f'Normalized One Dim Class Typicality KNN - {self.K} OOD - {self.OOD_Datamodule.name}'

        self.summary_aupr = f'Normalized One Dim Class Typicality KNN - {self.K} AUPR OOD - {self.OOD_Datamodule.name}'
        self.summary_fpr = f'Normalized One Dim Class Typicality KNN - {self.K} FPR OOD - {self.OOD_Datamodule.name}'

    def forward_callback(self, trainer, pl_module):
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA

        train_loader = self.Datamodule.deterministic_train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()

        features_train, labels_train = self.get_features(pl_module, train_loader)
        features_test, labels_test = self.get_features(pl_module, test_loader)
        features_ood, labels_ood = self.get_features(pl_module, ood_loader)

        self.get_eval_results(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood),
            np.copy(labels_train))
        
    def get_features(self, pl_module, dataloader):
        return super().get_features(pl_module, dataloader)
    
    def normalise(self, ftrain, ftest, food):
        return super().normalise(ftrain, ftest, food)
    
    def get_nearest_neighbours(self, ftest, food):
        return super().get_nearest_neighbours(ftest, food)
    
    # get the features of the data which also has the KNN in either the test set or the OOD dataset
    def get_knn_features(self, ftest, food, knn_indices):
        return super().get_knn_features(ftest, food, knn_indices)

    def get_1d_train(self,ftrain, ypred):
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        covs = [np.cov(x.T, bias=True) for x in xc] # Cov and means part should be fine
        means = [np.mean(x,axis=0,keepdims=True) for x in xc] # Calculates mean from (B,embdim) to (1,embdim)

        eigvalues = []
        eigvectors = []
        
        dtrain_1d_mean = [] # 1D mean for each class
        dtrain_1d_std = [] # 1D std for each class
        for class_num, class_cov in enumerate(covs):
            class_eigvals, class_eigvectors = np.linalg.eigh(class_cov) # Each column is a normalised eigenvector of
            
            # Reverse order as the eigenvalues and eigenvectors are in ascending order (lowest value first), therefore it would be beneficial to get them in descending order
            #class_eigvals, class_eigvectors = np.flip(class_eigvals, axis=0), np.flip(class_eigvectors,axis=0)
            eigvalues.append(np.expand_dims(class_eigvals,axis=1))
            eigvectors.append(class_eigvectors)

            # Perturbation to prevent numerical error
            perturbation = (np.sign(eigvalues[class_num])*1e-10)
            perturbation[perturbation==0] = 1e-10 # Replace any zero values with 1e-10 (this is done as np.sign is zero when the specific class eigvalue is zero)

            # Get the distribution of the 1d Scores from the certain class, which involves seeing the one dimensional scores for a specific class and calculating the mean and the standard deviation
            dtrain_class = np.matmul(eigvectors[class_num].T,(xc[class_num] - means[class_num]).T)**2/(eigvalues[class_num] + perturbation)
            #dtrain_class = np.matmul(eigvectors[class_num].T,(xc[class_num] - means[class_num]).T)**2/(eigvalues[class_num] + (np.sign(eigvalues[class_num])*1e-10))
            dtrain_1d_mean.append(np.mean(dtrain_class, axis= 1, keepdims=True))
            dtrain_1d_std.append(np.std(dtrain_class, axis= 1, keepdims=True))        
        
        return means, covs, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std
    
    def get_thresholds(self, fdata, means, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std):
        thresholds = []
        num_batches = len(fdata)//self.K
        # Currently goes through a single data point at a time which is not very efficient
        for i in range(num_batches):
            fdata_batch = fdata[(i*self.K):((i+1)*self.K)]
            #perturbations = [(np.sign(eigvalues[class_num])*1e-10) for class_num in range(len(means))]
            perturbations = [(np.sign(eigvalues[class_num])*1e-10) for class_num in range(len(means))]
            # Make any zero value into 1e-10
            for class_num in range(len(means)):
                perturbations[class_num][perturbations[class_num]==0] = 1e-10


            ddata = [np.matmul(eigvectors[class_num].T,(fdata_batch - means[class_num]).T)**2/(eigvalues[class_num] + perturbations[class_num]) for class_num in range(len(means))] # Calculate the 1D scores for all the different classes 
        
            # obtain the normalised the scores for the different classes
            ddata = [ddata[class_num] - dtrain_1d_mean[class_num]/(dtrain_1d_std[class_num]  +1e-10) for class_num in range(len(means))] # shape (dim, batch)
            
            # shape (dim) average of all data in batch size
            ddata = [np.mean(ddata[class_num],axis=1) for class_num in range(len(means))] # shape dim

            # Obtain the sum of absolute normalised scores
            scores = [np.sum(np.abs(ddata[class_num]),axis=0) for class_num in range(len(means))]
            # Obtain the scores corresponding to the lowest class
            ddata = np.min(scores,axis=0)


            thresholds.append(ddata)

        return thresholds

    def get_scores(self,ftrain, ftest, food,labelstrain):
        # Get information related to the train info
        mean, cov, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std = self.get_1d_train(ftrain,labelstrain)
        # Inference
        # Calculate the scores for the in-distribution data and the OOD data
        knn_indices = self.get_nearest_neighbours(ftest,food)
        knn_ID_features, knn_OOD_features = self.get_knn_features(ftest, food, knn_indices)
        
        din = self.get_thresholds(knn_ID_features, mean, eigvalues,eigvectors, dtrain_1d_mean,dtrain_1d_std)
        dood = self.get_thresholds(knn_OOD_features, mean, eigvalues,eigvectors, dtrain_1d_mean,dtrain_1d_std)    
        
        return din, dood
    
    def get_eval_results(self, ftrain, ftest, food,labelstrain):
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain, ftest, food)
        din, dood = self.get_scores(ftrain_norm,ftest_norm, food_norm,labelstrain)
        
        auroc, aupr, fpr = get_measures(dood,din)
        
        wandb.run.summary[self.summary_key] = auroc
        wandb.run.summary[self.summary_aupr] = aupr
        wandb.run.summary[self.summary_fpr] = fpr

        '''
        AUROC = get_roc_sklearn(din, dood)
        wandb.run.summary[self.summary_key] = AUROC
        '''

# Performs 1D typicality using a quadratic summation using the nearest neighbours as the batch for the data, and obtaining specific classes for the data
class NearestNeighboursQuadraticClass1DTypicality(NearestNeighboursClass1DTypicality):
    def __init__(self, Datamodule,OOD_Datamodule,
        quick_callback:bool = True,K:int = 10):

        super().__init__(Datamodule,OOD_Datamodule, quick_callback,K)
        # Used to save the summary value
        self.K = K
        self.summary_key = f'Normalized One Dim Class Quadratic Typicality KNN - {self.K} OOD - {self.OOD_Datamodule.name}'
        self.summary_aupr = f'Normalized One Dim Class Quadratic Typicality KNN - {self.K} AUPR OOD - {self.OOD_Datamodule.name}'
        self.summary_fpr = f'Normalized One Dim Class Quadratic Typicality KNN - {self.K} FPR OOD - {self.OOD_Datamodule.name}'
    
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
    
    def get_thresholds(self, fdata, means, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std):
        thresholds = []
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
            
            # obtain the normalised the scores for the different classes
            ddata = [ddata[class_num] - dtrain_1d_mean[class_num]/(dtrain_1d_std[class_num]  +1e-10) for class_num in range(len(means))] # shape (dim, batch)
            
            # shape (dim) average of all data in batch size
            ddata = [np.mean(ddata[class_num],axis=1) for class_num in range(len(means))] # shape dim

            ###### Only change from the previous class is this line ###########
            # Obtain the sum of absolute normalised scores squared (to emphasis the larger penalty when the deviation of a dimension is larger)
            scores = [np.sum(np.abs(ddata[class_num]**2),axis=0) for class_num in range(len(means))]
            ##################################################################

            # Obtain the scores corresponding to the lowest class
            ddata = np.min(scores,axis=0)

            thresholds.append(ddata)

        return thresholds

    def get_scores(self, ftrain, ftest, food, labelstrain):
        return super().get_scores(ftrain, ftest, food, labelstrain)

    def get_eval_results(self, ftrain, ftest, food, labelstrain):
        return super().get_eval_results(ftrain, ftest, food, labelstrain)


################## Analysis - Used to save the individual scores ######################### 
# Saving the individual scores for the approach
# Performs 1D typicality using a quadratic summation using the nearest neighbours as the batch for the data, and obtaining specific classes for the data
class NearestNeighboursQuadraticClass1DScoresTypicality(NearestNeighboursQuadraticClass1DTypicality):
    def __init__(self, Datamodule,OOD_Datamodule,
        quick_callback:bool = True,K:int = 10):

        super().__init__(Datamodule,OOD_Datamodule, quick_callback,K)
        # Used to save the summary value
        self.K = K
        self.summary_key = f'Normalized One Dim Scores Class Quadratic Typicality KNN - {self.K} OOD - {self.OOD_Datamodule.name}'

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
    
    def get_thresholds(self, fdata, means, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std):
        thresholds = []
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
            dimensional_scores = ddata_mean_deviation[np.argmin(scores,axis=0)]

            thresholds.append(dimensional_scores)
        
        return thresholds

    def get_scores(self,ftrain, ftest, food,labelstrain):
        # Get information related to the train info
        mean, cov, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std = self.get_1d_train(ftrain,labelstrain)
        # Inference
        # Calculate the scores for the in-distribution data and the OOD data
        knn_indices = self.get_nearest_neighbours(ftest,food)
        knn_ID_features, knn_OOD_features = self.get_knn_features(ftest, food, knn_indices)
        
        din = self.get_thresholds(knn_ID_features, mean, eigvalues,eigvectors, dtrain_1d_mean,dtrain_1d_std)
        dood = self.get_thresholds(knn_OOD_features, mean, eigvalues,eigvectors, dtrain_1d_mean,dtrain_1d_std)    
        
        return din, dood

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
        
    def get_eval_results(self, ftrain, ftest, food,labelstrain):
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain, ftest, food)
        din, dood = self.get_scores(ftrain_norm,ftest_norm, food_norm,labelstrain)
        # get the average deviations for the in-distribution and OOD data
        self.datasaving(din,dood,self.summary_key)


# Ablation
# Performs 1D typicality using a quadratic summation using the nearest neighbours as the batch for the data, without obtaining specific classes for the data
class NearestNeighboursQuadraticMarginal1DTypicality(NearestNeighboursClass1DTypicality):
    def __init__(self, Datamodule,OOD_Datamodule,
        quick_callback:bool = True,K:int = 10):

        super().__init__(Datamodule,OOD_Datamodule, quick_callback,K)
        # Used to save the summary value
        self.K = K
        self.summary_key = f'Normalized One Dim Marginal Quadratic Typicality KNN - {self.K} OOD - {self.OOD_Datamodule.name}'
        self.summary_aupr = f'Normalized One Dim Marginal Quadratic Typicality KNN - {self.K} AUPR OOD - {self.OOD_Datamodule.name}'
        self.summary_fpr = f'Normalized One Dim Marginal Quadratic Typicality KNN - {self.K} FPR OOD - {self.OOD_Datamodule.name}'
    
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
        # Nawid - get all the features which belong to each of the different classes
        cov = np.cov(ftrain.T, bias=True) # Cov and means part should be fine
        mean = np.mean(ftrain,axis=0,keepdims=True) # Calculates mean from (B,embdim) to (1,embdim)
        
        eigvalues, eigvectors = np.linalg.eigh(cov)
        eigvalues = np.expand_dims(eigvalues,axis=1)
        
        # Used to prevent numerical error
        perturbation = (np.sign(eigvalues)*1e-10)
        perturbation[perturbation==0] = 1e-10 # Replace any zero values with 1e-10 (this is done as np.sign is zero when the specific class eigvalue is zero)


        dtrain = np.matmul(eigvectors.T,(ftrain - mean).T)**2/(eigvalues + perturbation)
        
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

    def get_thresholds(self, fdata, means, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std):
        thresholds = []
        num_batches = len(fdata)//self.K
        # Currently goes through a single data point at a time which is not very efficient
        for i in range(num_batches):
            fdata_batch = fdata[(i*self.K):((i+1)*self.K)]
            # Added additional constant to the eignvalue for the purpose of numerical stability

            perturbation = (np.sign(eigvalues)*1e-10)

            perturbation[perturbation==0] = 1e-10 # Replace any zero values with 1e-10 (this is done as np.sign is zero when the specific class eigvalue is zero)
            ddata = np.matmul(eigvectors.T,(fdata_batch - means).T)**2/(eigvalues + perturbation)  # shape (dim, batch size)
                         
            # Normalise the data
            ddata = (ddata - dtrain_1d_mean)/(dtrain_1d_std +1e-10) # shape (dim, batch)
            
            # shape (dim) average of all data in batch size
            ddata = np.mean(ddata,axis= 1) # shape : (dim)

            # Sum of the deviations of each individual dimension
            ddata_deviation = np.sum(np.abs(ddata**2))

            thresholds.append(ddata_deviation)

        return thresholds

    def get_scores(self, ftrain, ftest, food, labelstrain):
        return super().get_scores(ftrain, ftest, food, labelstrain)

    def get_eval_results(self, ftrain, ftest, food, labelstrain):
        return super().get_eval_results(ftrain, ftest, food, labelstrain)



# NEED TO CHECK THAT THE SHAPE OF THE OUTPUTS ARE CORRECT FOR THE DIFFERENT SITUATIONS
# Ablation which does not use the 1 dimensional decomposition of the data
class KNNClassTypicality(NearestNeighboursQuadraticClass1DTypicality):
    def __init__(self, Datamodule,OOD_Datamodule,
        quick_callback:bool = True,K:int = 10):

        super().__init__(Datamodule,OOD_Datamodule, quick_callback,K)
        # Used to save the summary value

        self.K = K
        self.summary_key = f'Normalized All Dim Class Typicality KNN - {self.K} OOD - {self.OOD_Datamodule.name}'
        self.summary_aupr = f'Normalized All Dim Class Typicality KNN - {self.K} AUPR OOD - {self.OOD_Datamodule.name}'
        self.summary_fpr = f'Normalized All Dim Class Typicality KNN - {self.K} FPR OOD - {self.OOD_Datamodule.name}'

    def on_test_epoch_end(self, trainer, pl_module):
        return super().on_test_epoch_end(trainer, pl_module)

    def forward_callback(self, trainer, pl_module):
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA

        train_loader = self.Datamodule.deterministic_train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()

        features_train, labels_train = self.get_features(pl_module, train_loader)
        features_test, labels_test = self.get_features(pl_module, test_loader)
        features_ood, labels_ood = self.get_features(pl_module, ood_loader)

        self.get_eval_results(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood),
            np.copy(labels_train))
        
    def get_features(self, pl_module, dataloader):
        return super().get_features(pl_module, dataloader)
    
    def normalise(self, ftrain, ftest, food):
        return super().normalise(ftrain, ftest, food)
    
    def get_nearest_neighbours(self, ftest, food):
        return super().get_nearest_neighbours(ftest, food)

    # get the features of the data which also has the KNN in either the test set or the OOD dataset
    def get_knn_features(self, ftest, food, knn_indices):
        return super().get_knn_features(ftest, food, knn_indices)

    # Get the mean, covariance and the average distance for a class
    def get_train(self, ftrain,ypred):
        # Nawid - get all the features which belong to each of the different classes
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        # Calculate the means of each of the different classes
        means = [np.mean(x,axis=0,keepdims=True) for x in xc] # Calculates mean from (B,embdim) to (1,embdim)
        # Calculate the covariance matrices for each of the different classes
        covs = [np.cov(x.T, bias=True) for x in xc]
        dtrain_means = []
        dtrain_stds = []
        for class_num in range(len(np.unique(ypred))):
            # Go through each of the different classes and calculate the average likelihood which is the entropy of the class
            xc_class = xc[class_num]
            # calculates the mahalanobis distance for all the data points 
            dtrain = np.sum(
                (xc_class - means[class_num]) # Nawid - distance between the data point and the mean
                * (
                    np.linalg.pinv(covs[class_num]).dot(
                        (xc_class - means[class_num]).T
                    ) # Nawid - calculating the covariance matrix of the data belonging to a particular class and dot product by the distance of the data point from the mean (distance calculation)
                ).T,
                axis=-1,  # # shape (batch, ) IT IS IMPORTANT TO PUT THE COMMA there otherwise the shape is (batch, dimension)
                )
            
            # Calculates the average mahalanobis distance
            class_average_distance = np.mean(dtrain)
            class_std = np.std(dtrain)
            # Places the average mahalanobis distance for a particular class
            dtrain_means.append(class_average_distance)
            dtrain_stds.append(class_std)
            # Would there be benefit in scaling (Not sure but this is just an ablation so should not be too important)

        return means, covs, dtrain_means, dtrain_stds

    def get_thresholds(self,fdata, means, covs, dtrain_means,dtrain_stds):    
        num_classes = len(means)
        thresholds = []
        num_batches = len(fdata)//self.K
        # Currently goes through a single data point at a time which is not very efficient
        
        for i in range(num_batches):
            fdata_batch = fdata[(i*self.K):((i+1)*self.K)]
            # Calculate the scores for a particular batch of data for all the different classes
            # calculates the mahalanobis distance of each data point in each of the different classes
            ddata = [
            np.sum(
                (fdata_batch - means[class_num]) # Nawid - distance between the data point and the mean
                * (
                    np.linalg.pinv(covs[class_num]).dot(
                        (fdata_batch - means[class_num]).T
                    ) # Nawid - calculating the covariance matrix of the data belonging to a particular class and dot product by the distance of the data point from the mean (distance calculation)
                ).T,
                axis=-1,
            )
            for class_num in range(num_classes) # Nawid - done for all the different classes
            ]
            
            # list with size (k_neighbours) - calculates the mahalanobis distance of each of the different neighbours
            # ddata is a list of n classes where each element has a shape of (batch,)  
            # calculate the average scores for the situation within a class and see how the average deviates from the mean it is a list of n classes where each value is a scalar
            #ddata_practice = [np.abs(np.mean(ddata[class_num])-dtrain_mean[class_num]) for class_num in range(num_classes)] # deviation from the mean for the different classes
            
            # calculate the difference between the mean and the other value and then normalize the data
            ddata = [(np.mean(ddata[class_num])-dtrain_means[class_num])/dtrain_stds[class_num] for class_num in range(num_classes)]
            ddata = [np.abs(ddata[class_num]) for class_num in range(num_classes)]
            # NEED TO CONSIDER EXAMINING DIFFERENCE BETWEEN THE MEAN SQUARED OR THE SQUARED MEAN BETWEEN THE DATA POINTS, Main typicality approach finds (value- mean)**2 and then sums the different dimensions, therefore I should do (value -mean)**2
            #ddata = [np.abs(np.mean(ddata[class_num])-dtrain_means[class_num]) for class_num in range(num_classes)] # deviation from the mean for the different classes

            ddata = np.min(ddata ,axis=0) # Finds min of all the class values
            thresholds.append(ddata)
        
        return thresholds

    
    def get_scores(self,ftrain, ftest, food,labelstrain):
        # Get information related to the train info
        means, covs, dtrain_means,dtrain_stds = self.get_train(ftrain,labelstrain)
        # Inference
        # Calculate the scores for the in-distribution data and the OOD data
        knn_indices = self.get_nearest_neighbours(ftest,food)
        knn_ID_features, knn_OOD_features = self.get_knn_features(ftest, food, knn_indices)
        
        din = self.get_thresholds(knn_ID_features, means, covs, dtrain_means, dtrain_stds)
        dood = self.get_thresholds(knn_OOD_features, means, covs, dtrain_means, dtrain_stds)    
        
        return din, dood
    
    def get_eval_results(self, ftrain, ftest, food,labelstrain):
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain, ftest, food)
        din, dood = self.get_scores(ftrain_norm,ftest_norm, food_norm,labelstrain)
        
        auroc, aupr, fpr = get_measures(dood,din)
        
        wandb.run.summary[self.summary_key] = auroc
        wandb.run.summary[self.summary_aupr] = aupr
        wandb.run.summary[self.summary_fpr] = fpr

# Oracle situation where the nearest neighbours are always obtained from the same dataset
class OracleNearestNeighboursClass1DTypicality(NearestNeighboursClass1DTypicality):
    def __init__(self, Datamodule, OOD_Datamodule, quick_callback: bool, K: int):
        super().__init__(Datamodule, OOD_Datamodule, quick_callback=quick_callback, K=K)
    
        self.summary_key = f'Oracle Normalized One Dim Class Typicality KNN - {self.K} OOD - {self.OOD_Datamodule.name}'
    
    def on_test_epoch_end(self, trainer, pl_module):
        return super().on_test_epoch_end(trainer, pl_module)
    
    def forward_callback(self, trainer, pl_module):
        return super().forward_callback(trainer, pl_module)
    
    def get_features(self, pl_module, dataloader):
        return super().get_features(pl_module, dataloader)
    
    def normalise(self, ftrain, ftest, food):
        return super().normalise(ftrain, ftest, food)

    # obtain the nearest neighbours in the same dataset as it is an oracle version
    def get_nearest_neighbours(self, fdata):
        # Make a distance matrix for the data
        distance_matrix = scipy.spatial.distance.cdist(fdata, fdata)
        bottom_k_indices = np.argsort(distance_matrix,axis=1)[:,:self.K] # shape (num samples,k) , each row has the k indices with the smallest values (including itself), so it gets K indices for each data point
        return bottom_k_indices

    # get the features of the data which are in the same dataset as the other situation
    def get_knn_features(self,fdata, knn_indices):        
        knn_features = fdata[knn_indices] # shape (num_samples_collated, K, embeddim)
        knn_features = np.reshape(knn_features,(knn_features.shape[0]*knn_features.shape[1],knn_features.shape[2]))
            
        # Need to check with a toy example whether the different numbers are placed sequentially in a batch
        return knn_features

    def get_1d_train(self, ftrain, ypred):
        return super().get_1d_train(ftrain, ypred)
    
    def get_thresholds(self, fdata, means, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std):
        return super().get_thresholds(fdata, means, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std)

    def get_scores(self,ftrain, ftest, food,labelstrain):
        # Get information related to the train info
        mean, cov, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std = self.get_1d_train(ftrain,labelstrain)
        
        # Inference
        # Calculate the scores for the in-distribution data and the OOD data
        knn_ID_indices, knn_OOD_indices = self.get_nearest_neighbours(ftest), self.get_nearest_neighbours(food) 
        knn_ID_features,knn_OOD_features = self.get_knn_features(ftest,knn_ID_indices), self.get_knn_features(food,knn_OOD_indices)
        
        din = self.get_thresholds(knn_ID_features, mean, eigvalues,eigvectors, dtrain_1d_mean,dtrain_1d_std)
        dood = self.get_thresholds(knn_OOD_features, mean, eigvalues,eigvectors, dtrain_1d_mean,dtrain_1d_std)    
        
        return din, dood

    def get_eval_results(self, ftrain, ftest, food,labelstrain):
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain, ftest, food)
        din, dood = self.get_scores(ftrain_norm,ftest_norm, food_norm,labelstrain)
        
        auroc, aupr, fpr = get_measures(dood,din)
        
        wandb.run.summary[self.summary_key] = auroc
        wandb.run.summary[self.summary_aupr] = aupr
        wandb.run.summary[self.summary_fpr] = fpr
        '''
        AUROC = get_roc_sklearn(din, dood)
        wandb.run.summary[self.summary_key] = AUROC
        '''


########################## Code for the different values of the KNN ###########################################

# Performs 1D typicality using the different K values and saves them in a table
class DifferentKNNClass1DTypicality(NearestNeighboursClass1DTypicality):
    def __init__(self, Datamodule,OOD_Datamodule,
        quick_callback:bool = True,K:int = 10):

        super().__init__(Datamodule,OOD_Datamodule, quick_callback,K)
        # Used to save the summary value

    def on_test_epoch_end(self, trainer, pl_module):
        return super().on_test_epoch_end(trainer, pl_module)

    def forward_callback(self, trainer, pl_module):
         return super().forward_callback(trainer, pl_module)
    
        
    def get_features(self, pl_module, dataloader):
        return super().get_features(pl_module, dataloader)
    
    def normalise(self, ftrain, ftest, food):
        return super().normalise(ftrain, ftest, food)

    # Uses self.K instead of K
    def get_nearest_neighbours(self, ftest, food,K):
        num_ID = len(ftest)
        collated_features = np.concatenate((ftest, food))
        # Make a distance matrix for the data
        distance_matrix = scipy.spatial.distance.cdist(collated_features, collated_features)
        bottom_k_indices = np.argsort(distance_matrix,axis=1)[:,:K] # shape (num samples,k) , each row has the k indices with the smallest values (including itself), so it gets K indices for each data point
        return bottom_k_indices
    
    # get the features of the data which also has the KNN in either the test set or the OOD dataset
    def get_knn_features(self, ftest, food, knn_indices):
        return super().get_knn_features(ftest, food, knn_indices)

    def get_1d_train(self, ftrain, ypred):
        return super().get_1d_train(ftrain, ypred)
    
    def get_thresholds(self, fdata, means, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std,K):
        thresholds = []
        num_batches = len(fdata)//K
        # Currently goes through a single data point at a time which is not very efficient
        for i in range(num_batches):
            fdata_batch = fdata[(i*K):((i+1)*K)]
            
            perturbations = [(np.sign(eigvalues[class_num])*1e-10) for class_num in range(len(means))]
            # Make any zero value into 1e-10
            for class_num in range(len(means)):
                perturbations[class_num][perturbations[class_num]==0] = 1e-10

            ddata = [np.matmul(eigvectors[class_num].T,(fdata_batch - means[class_num]).T)**2/(eigvalues[class_num] + perturbations[class_num]) for class_num in range(len(means))] # Calculate the 1D scores for all the different classes 
        
            # obtain the normalised the scores for the different classes
            ddata = [ddata[class_num] - dtrain_1d_mean[class_num]/(dtrain_1d_std[class_num]  +1e-10) for class_num in range(len(means))] # shape (dim, batch)
            
            # shape (dim) average of all data in batch size
            ddata = [np.mean(ddata[class_num],axis=1) for class_num in range(len(means))] # shape dim

            # Obtain the sum of absolute normalised scores
            scores = [np.sum(np.abs(ddata[class_num]),axis=0) for class_num in range(len(means))]
            # Obtain the scores corresponding to the lowest class
            ddata = np.min(scores,axis=0)


            thresholds.append(ddata)

        return thresholds

    def get_eval_results(self, ftrain, ftest, food,labelstrain):
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain, ftest, food)
        name = f'Different K Normalized One Dim Class Typicality KNN OOD - {self.OOD_Datamodule.name}'
        self.datasaving(ftrain_norm,ftest_norm,food_norm,labelstrain,name)
        
    
    def datasaving(self,ftrain,ftest, food,labelstrain,wandb_name):
        K_values = [1,5,10,15,20,25]
        mean, cov, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std = self.get_1d_train(ftrain,labelstrain)
        table_data = {'K Value':[], 'AUROC':[]}
        for k in K_values:
            knn_indices = self.get_nearest_neighbours(ftest,food,k)
            knn_ID_features, knn_OOD_features = self.get_knn_features(ftest, food, knn_indices)
            din = self.get_thresholds(knn_ID_features, mean, eigvalues,eigvectors, dtrain_1d_mean,dtrain_1d_std,k)
            dood = self.get_thresholds(knn_OOD_features, mean, eigvalues,eigvectors, dtrain_1d_mean,dtrain_1d_std,k)
            AUROC = get_roc_sklearn(din, dood)

            table_data['K Value'].append(k)
            table_data['AUROC'].append(round(AUROC,3))

        table_df = pd.DataFrame(table_data)
        table = wandb.Table(dataframe=table_df)
        wandb.log({wandb_name:table})


# Performs 1D typicality using the different K values and saves them in a table
class DifferentKNNQuadraticClass1DTypicality(DifferentKNNClass1DTypicality):
    def __init__(self, Datamodule,OOD_Datamodule,
        quick_callback:bool = True,K:int = 10):

        super().__init__(Datamodule,OOD_Datamodule, quick_callback,K)
        # Used to save the summary value

    def on_test_epoch_end(self, trainer, pl_module):
        return super().on_test_epoch_end(trainer, pl_module)

    def forward_callback(self, trainer, pl_module):
         return super().forward_callback(trainer, pl_module)
    
        
    def get_features(self, pl_module, dataloader):
        return super().get_features(pl_module, dataloader)
    
    def normalise(self, ftrain, ftest, food):
        return super().normalise(ftrain, ftest, food)

    # The case of the different k, has an additional parameter k to use different k values (though should change code to make it different)
    def get_nearest_neighbours(self, ftest, food,K):
        return super().get_nearest_neighbours(ftest, food,K)
    
    # get the features of the data which also has the KNN in either the test set or the OOD dataset
    def get_knn_features(self, ftest, food, knn_indices):
        return super().get_knn_features(ftest, food, knn_indices)

    def get_1d_train(self, ftrain, ypred):
        return super().get_1d_train(ftrain, ypred)
    
    def get_thresholds(self, fdata, means, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std,K):
        thresholds = []
        num_batches = len(fdata)//K
        # Currently goes through a single data point at a time which is not very efficient
        for i in range(num_batches):
            fdata_batch = fdata[(i*K):((i+1)*K)]
            perturbations = [(np.sign(eigvalues[class_num])*1e-10) for class_num in range(len(means))]
            # Make any zero value into 1e-10
            for class_num in range(len(means)):
                perturbations[class_num][perturbations[class_num]==0] = 1e-10

            ddata = [np.matmul(eigvectors[class_num].T,(fdata_batch - means[class_num]).T)**2/(eigvalues[class_num] + perturbations[class_num]) for class_num in range(len(means))] # Calculate the 1D scores for all the different classes 
            #ddata = [np.matmul(eigvectors[class_num].T,(fdata_batch - means[class_num]).T)**2/(eigvalues[class_num] + (np.sign(eigvalues[class_num])*1e-10)) for class_num in range(len(means))] # Calculate the 1D scores for all the different classes 
        
            # obtain the normalised the scores for the different classes
            ddata = [ddata[class_num] - dtrain_1d_mean[class_num]/(dtrain_1d_std[class_num]  +1e-10) for class_num in range(len(means))] # shape (dim, batch)
            
            # shape (dim) average of all data in batch size
            ddata = [np.mean(ddata[class_num],axis=1) for class_num in range(len(means))] # shape dim

            ###### Only change from the previous class is this line ###########
            # Obtain the sum of absolute normalised scores squared (to emphasis the larger penalty when the deviation of a dimension is larger)
            scores = [np.sum(np.abs(ddata[class_num]**2),axis=0) for class_num in range(len(means))]
            ##################################################################

            # Obtain the scores corresponding to the lowest class
            ddata = np.min(scores,axis=0)

            thresholds.append(ddata)

        return thresholds

    # Only difference is the change of the name for the logging of the tables
    def get_eval_results(self, ftrain, ftest, food,labelstrain):
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain, ftest, food)
        name = f'Different K Normalized Quadratic One Dim Class Typicality KNN OOD - {self.OOD_Datamodule.name}'
        self.datasaving(ftrain_norm,ftest_norm,food_norm,labelstrain,name)

    def datasaving(self, ftrain, ftest, food, labelstrain, wandb_name):
        return super().datasaving(ftrain, ftest, food, labelstrain, wandb_name)


# Ablation studies
# Performs 1D typicality using the different K values and saves them in a table
class DifferentKNNMarginal1DTypicality(NearestNeighbours1DTypicality):
    def __init__(self, Datamodule,OOD_Datamodule,
        quick_callback:bool = True,K:int = 10):

        super().__init__(Datamodule,OOD_Datamodule, quick_callback,K)
        # Used to save the summary value

    def on_test_epoch_end(self, trainer, pl_module):
        return super().on_test_epoch_end(trainer, pl_module)

    def forward_callback(self, trainer, pl_module):
         return super().forward_callback(trainer, pl_module)
        
    def get_features(self, pl_module, dataloader):
        return super().get_features(pl_module, dataloader)
    
    def normalise(self, ftrain, ftest, food):
        return super().normalise(ftrain, ftest, food)
    
    def get_nearest_neighbours(self, ftest, food,K):
        num_ID = len(ftest)
        collated_features = np.concatenate((ftest, food))
        # Make a distance matrix for the data
        distance_matrix = scipy.spatial.distance.cdist(collated_features, collated_features)
        bottom_k_indices = np.argsort(distance_matrix,axis=1)[:,:K] # shape (num samples,k) , each row has the k indices with the smallest values (including itself), so it gets K indices for each data point
        return bottom_k_indices

    # get the features of the data which also has the KNN in either the test set or the OOD dataset
    def get_knn_features(self, ftest, food, knn_indices):
        return super().get_knn_features(ftest, food, knn_indices)

    def get_1d_train(self, ftrain):
        return super().get_1d_train(ftrain)
    
    def get_thresholds(self, fdata, mean, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std,K):
        thresholds = []
        num_batches = len(fdata)//K
        perturbation = (np.sign(eigvalues)*1e-10)
        perturbation[perturbation==0] = 1e-10 # Replace any zero values with 1e-10 (this is done as np.sign is zero when the specific class eigvalue is zero)
        # Currently goes through a single data point at a time which is not very efficient
        for i in range(num_batches):
            fdata_batch = fdata[(i*K):((i+1)*K)]

            

            ddata = np.matmul(eigvectors.T,(fdata_batch - mean).T)**2/(eigvalues + perturbation)  # shape (dim, batch size)
            # Normalise the data
            ddata = (ddata - dtrain_1d_mean)/(dtrain_1d_std +1e-10) # shape (dim, batch)

            # shape (dim) average of all data in batch size
            ddata = np.mean(ddata,axis= 1) # shape : (dim)
            
            # Sum of the deviations of each individual dimension
            ddata_deviation = np.sum(np.abs(ddata))

            thresholds.append(ddata_deviation)
        
        return thresholds

    def get_eval_results(self, ftrain, ftest, food):
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain, ftest, food)
        name = f'Different K Normalized One Dim Marginal Typicality KNN OOD - {self.OOD_Datamodule.name}'
        self.datasaving(ftrain_norm,ftest_norm,food_norm,name)        
    
    def datasaving(self,ftrain,ftest, food,wandb_name):
        K_values = [1,5,10,15,20,25]
        mean, cov, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std = self.get_1d_train(ftrain)
        table_data = {'K Value':[], 'AUROC':[]}
        for k in K_values:
            knn_indices = self.get_nearest_neighbours(ftest,food,k)
            knn_ID_features, knn_OOD_features = self.get_knn_features(ftest, food, knn_indices)
            din = self.get_thresholds(knn_ID_features, mean, eigvalues,eigvectors, dtrain_1d_mean,dtrain_1d_std,k)
            dood = self.get_thresholds(knn_OOD_features, mean, eigvalues,eigvectors, dtrain_1d_mean,dtrain_1d_std,k)
            AUROC = get_roc_sklearn(din, dood)

            table_data['K Value'].append(k)
            table_data['AUROC'].append(round(AUROC,3))

        table_df = pd.DataFrame(table_data)
        table = wandb.Table(dataframe=table_df)
        wandb.log({wandb_name:table})



# Ablation studies
# Performs typicality using the different K values and saves them in a table
class DifferentKNNClassTypicality(DifferentKNNMarginal1DTypicality):
    def __init__(self, Datamodule,OOD_Datamodule,
        quick_callback:bool = True,K:int = 10):

        super().__init__(Datamodule,OOD_Datamodule, quick_callback,K)
        # Used to save the summary value

    def on_test_epoch_end(self, trainer, pl_module):
        return super().on_test_epoch_end(trainer, pl_module)

    def forward_callback(self, trainer, pl_module):
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA

        train_loader = self.Datamodule.deterministic_train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()

        features_train, labels_train = self.get_features(pl_module, train_loader)
        features_test, labels_test = self.get_features(pl_module, test_loader)
        features_ood, labels_ood = self.get_features(pl_module, ood_loader)

        self.get_eval_results(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood),
            np.copy(labels_train))
        
    def get_features(self, pl_module, dataloader):
        return super().get_features(pl_module, dataloader)
    
    def normalise(self, ftrain, ftest, food):
        return super().normalise(ftrain, ftest, food)
    
    def get_nearest_neighbours(self, ftest, food,K):
        return super().get_nearest_neighbours(ftest, food,K)


    # get the features of the data which also has the KNN in either the test set or the OOD dataset
    def get_knn_features(self, ftest, food, knn_indices):
        return super().get_knn_features(ftest, food, knn_indices)

    # Get the mean, covariance and the average distance for a class
    def get_train(self, ftrain,ypred):
        # Nawid - get all the features which belong to each of the different classes
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        # Calculate the means of each of the different classes
        means = [np.mean(x,axis=0,keepdims=True) for x in xc] # Calculates mean from (B,embdim) to (1,embdim)
        # Calculate the covariance matrices for each of the different classes
        covs = [np.cov(x.T, bias=True) for x in xc]

        dtrain_means = []
        for class_num in range(len(np.unique(ypred))):
            # Go through each of the different classes and calculate the average likelihood which is the entropy of the class
            xc_class = xc[class_num]
            dtrain = np.sum(
                (xc_class - means[class_num]) # Nawid - distance between the data point and the mean
                * (
                    np.linalg.pinv(covs[class_num]).dot(
                        (xc_class - means[class_num]).T
                    ) # Nawid - calculating the covariance matrix of the data belonging to a particular class and dot product by the distance of the data point from the mean (distance calculation)
                ).T,
                axis=-1)
            class_average_distance = np.mean(dtrain)
            dtrain_means.append(class_average_distance)
            # Would there be benefit in scaling (Not sure but this is just an ablation so should not be too important)

        return means, covs, dtrain_means

    def get_thresholds(self,fdata, means, covs, dtrain_mean, K):
        
        num_classes = len(means)
        thresholds = []

        num_batches = len(fdata)//K
        # Currently goes through a single data point at a time which is not very efficient
        for i in range(num_batches):
            fdata_batch = fdata[(i*K):((i+1)*K)]
            # Calculate the scores for a particular batch of data for all the different classes
            ddata = [
            np.sum(
                (fdata_batch - means[class_num]) # Nawid - distance between the data point and the mean
                * (
                    np.linalg.pinv(covs[class_num]).dot(
                        (fdata_batch - means[class_num]).T
                    ) # Nawid - calculating the covariance matrix of the data belonging to a particular class and dot product by the distance of the data point from the mean (distance calculation)
                ).T,
                axis=-1,
            )
            for class_num in range(num_classes) # Nawid - done for all the different classes
            ]
            # ddata is a list of n classes where each element has a shape of (batch,)  
            # calculate the average scores for the situation within a class and see how the average deviates from the mean it is a list of n classes where each value is a scalar
            ddata = [np.abs(np.mean(ddata[class_num])-dtrain_mean[class_num]) for class_num in range(num_classes)] # deviation from the mean for the different classes

            ddata = np.min(ddata ,axis=0) # Finds min of all the class values
            thresholds.append(ddata)
        
        return thresholds
    
    def get_eval_results(self, ftrain, ftest, food,labelstrain):
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain, ftest, food)
        name = f'Different K Class Typicality KNN OOD - {self.OOD_Datamodule.name}'
        self.datasaving(ftrain_norm,ftest_norm,food_norm,labelstrain,name)        
    
    def datasaving(self,ftrain,ftest, food,labelstrain, wandb_name):
        
        K_values = [1,5,10,15,20,25]
        means, covs, dtrain_means = self.get_train(ftrain,labelstrain)
        table_data = {'K Value':[], 'AUROC':[]}
        for k in K_values:
            knn_indices = self.get_nearest_neighbours(ftest,food,k)
            knn_ID_features, knn_OOD_features = self.get_knn_features(ftest, food, knn_indices)

            din = self.get_thresholds(knn_ID_features, means, covs, dtrain_means, k)
            dood = self.get_thresholds(knn_OOD_features, means, covs, dtrain_means, k)

            #dood = self.get_thresholds(knn_OOD_features, mean, eigvalues,eigvectors, dtrain_1d_mean,dtrain_1d_std,k)
            AUROC = get_roc_sklearn(din, dood)

            table_data['K Value'].append(k)
            table_data['AUROC'].append(round(AUROC,3))

        table_df = pd.DataFrame(table_data)
        table = wandb.Table(dataframe=table_df)
        wandb.log({wandb_name:table})


class DifferentKNNMarginalTypicality(DifferentKNNMarginal1DTypicality):
    def __init__(self, Datamodule,OOD_Datamodule,
        quick_callback:bool = True,K:int = 10):

        super().__init__(Datamodule,OOD_Datamodule, quick_callback,K)
        # Used to save the summary value

    def on_test_epoch_end(self, trainer, pl_module):
        return super().on_test_epoch_end(trainer, pl_module)

    def forward_callback(self, trainer, pl_module):
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA
        
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
    
    def get_nearest_neighbours(self, ftest, food,K):
        return super().get_nearest_neighbours(ftest, food,K)

    # get the features of the data which also has the KNN in either the test set or the OOD dataset
    def get_knn_features(self, ftest, food, knn_indices):
        return super().get_knn_features(ftest, food, knn_indices)

    # Get the mean, covariance and the average distance for a class
    def get_train(self, ftrain):
        mean = np.mean(ftrain,axis=0,keepdims=True) 
        cov = np.cov(ftrain.T, bias=True)
        
        dtrain = np.sum(
                (ftrain - mean) # Nawid - distance between the data point and the mean
                * (
                    np.linalg.pinv(cov).dot(
                        (ftrain - mean).T
                    ) # Nawid - calculating the covariance matrix of the data belonging to a particular class and dot product by the distance of the data point from the mean (distance calculation)
                ).T,
                axis=-1)
            
        dtrain_mean = np.mean(dtrain)

        return mean, cov, dtrain_mean

    def get_thresholds(self,fdata, mean, cov, dtrain_mean, K):
        thresholds = []
        num_batches = len(fdata)//K
        # Currently goes through a single data point at a time which is not very efficient
        for i in range(num_batches):
            fdata_batch = fdata[(i*K):((i+1)*K)]
            # Calculate the scores for a particular batch of data for all the different classes
            ddata = np.sum(
                (fdata_batch - mean) # Nawid - distance between the data point and the mean
                * (
                    np.linalg.pinv(cov).dot(
                        (fdata_batch - mean).T
                    ) # Nawid - calculating the covariance matrix of the data belonging to a particular class and dot product by the distance of the data point from the mean (distance calculation)
                ).T,
                axis=-1)
            # ddata is a list of n classes where each element has a shape of (batch,)  
            # calculate the average scores for the situation within a class and see how the average deviates from the mean it is a list of n classes where each value is a scalar
            ddata = np.abs(np.mean(ddata)-dtrain_mean)
            thresholds.append(ddata)

        return thresholds
    
    def get_eval_results(self, ftrain, ftest, food):
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain, ftest, food)
        name = f'Different K Marginal Typicality KNN OOD - {self.OOD_Datamodule.name}'
        self.datasaving(ftrain_norm,ftest_norm,food_norm,name)        
    
    def datasaving(self,ftrain,ftest, food, wandb_name):
        
        K_values = [1,5,10,15,20,25]
        
        means, covs, dtrain_means = self.get_train(ftrain)
        table_data = {'K Value':[], 'AUROC':[]}
        for k in K_values:
            knn_indices = self.get_nearest_neighbours(ftest,food,k)
            knn_ID_features, knn_OOD_features = self.get_knn_features(ftest, food, knn_indices)

            din = self.get_thresholds(knn_ID_features, means, covs, dtrain_means, k)
            dood = self.get_thresholds(knn_OOD_features, means, covs, dtrain_means, k)

            #dood = self.get_thresholds(knn_OOD_features, mean, eigvalues,eigvectors, dtrain_1d_mean,dtrain_1d_std,k)
            AUROC = get_roc_sklearn(din, dood)

            table_data['K Value'].append(k)
            table_data['AUROC'].append(round(AUROC,3))

        table_df = pd.DataFrame(table_data)
        table = wandb.Table(dataframe=table_df)
        wandb.log({wandb_name:table})