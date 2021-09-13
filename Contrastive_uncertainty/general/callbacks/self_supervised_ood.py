from numpy.core.fromnumeric import reshape
from numpy.lib.function_base import quantile
from pandas.io.formats.format import DataFrameFormatter
import torch
from torch._C import dtype
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
import copy
import random 


import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score

from Contrastive_uncertainty.general.utils.hybrid_utils import OOD_conf_matrix
from Contrastive_uncertainty.general.callbacks.general_callbacks import quickloading
from Contrastive_uncertainty.general.callbacks.ood_callbacks import get_roc_sklearn, get_roc_plot, table_saving
from Contrastive_uncertainty.general.utils.pl_metrics import precision_at_k

from Contrastive_uncertainty.general.callbacks.one_dim_typicality_callback import Data_Augmented_Point_One_Dim_Class_Typicality_Normalised


# Examining the difference in the loss values using the loss values directly
class Self_supervised_loss_OOD(pl.Callback):
    def __init__(self, Datamodule,OOD_Datamodule,
        quick_callback:bool = True):

        super().__init__()

        self.Datamodule = Datamodule
        self.OOD_Datamodule = OOD_Datamodule

        # Make it so both the train and the test transforms are the same for each case
        self.OOD_Datamodule.test_transforms = self.Datamodule.train_transforms
        self.OOD_Datamodule.test_transforms = self.Datamodule.train_transforms #  Make the transform of the OOD data the same as the actual data

        self.Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly
        
        # Chooses what vector representation to use as well as which level of label hierarchy to use

        self.OOD_dataname = self.OOD_Datamodule.name

        self.summary_key = f'Instance Discrimination Loss AUROC OOD - {self.OOD_dataname}'
    
    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer=trainer, pl_module=pl_module)

    def forward_callback(self, trainer, pl_module):
        train_loader = self.Datamodule.deterministic_train_dataloader()

        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()

        # Obtain representations of the data
        features_train, labels_train = self.get_features(pl_module, train_loader)
        features_test_query,features_test_key, labels_test = self.get_instance_features(pl_module, test_loader)
        features_ood_query, features_ood_key, labels_ood = self.get_instance_features(pl_module, ood_loader)

        self.get_eval_results(pl_module,
            np.copy(features_train),
            np.copy(features_test_query),
            np.copy(features_test_key),
            np.copy(features_ood_query),
            np.copy(features_ood_key))

    
    # Obtain the momentum features of the data
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

     

    # Obtain the features for the momentum situation and the online encoder situation
    def get_instance_features(self, pl_module, dataloader):
        features_q, features_k, labels = [], [],[]
        
        loader = quickloading(self.quick_callback, dataloader)

        # Obtains the different imgs
        for index, (img, *label, indices) in enumerate(loader):
            assert len(loader)>0, 'loader is empty'
            if isinstance(img, tuple) or isinstance(img, list):
                    img1, img2 = img # Used to take into accoutn whether the data is a tuple of the different augmentations

            # Selects the correct label based on the desired label level
            if len(label) > 1:
                label_index = 0
                label = label[label_index]
            else: # Used for the case of the OOD data
                label = label[0]

            img1, img2 = img1.to(pl_module.device), img2.to(pl_module.device)


            feature_vector_k = pl_module.callback_vector(img1) # Performs the callback for the desired level
            feature_vector_q = pl_module.alternative_callback_vector(img2) #  obtains the representation for the case of the online encoder

            # Place feature vectores in list            
            features_q += list(feature_vector_q.data.cpu().numpy())
            features_k += list(feature_vector_k.data.cpu().numpy())
            labels += list(label.data.cpu().numpy())            

        return np.array(features_q), np.array(features_k), np.array(labels)


    # Normalises the data
    def normalise(self,ftrain, ftest_q, ftest_k, food_q, food_k):
        # Nawid -normalise the featues for the training, test and ood data
        # standardize data
        ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10
        ftest_q /= np.linalg.norm(ftest_q, axis=-1, keepdims=True) + 1e-10
        ftest_k /= np.linalg.norm(ftest_k, axis=-1, keepdims=True) + 1e-10

        food_q /= np.linalg.norm(food_q, axis=-1, keepdims=True) + 1e-10
        food_k /= np.linalg.norm(food_k, axis=-1, keepdims=True) + 1e-10


        # Nawid - calculate the mean and std of the traiing features
        m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)
        # Nawid - normalise data using the mean and std
        ftrain = (ftrain - m) / (s + 1e-10)
        ftest_q = (ftest_q - m) / (s + 1e-10)
        ftest_k = (ftest_k - m) / (s + 1e-10)
        food_q = (food_q - m) / (s + 1e-10)
        food_k = (food_k - m) / (s + 1e-10)        
        
        return ftrain, ftest_q, ftest_k, food_q, food_k
    
    # Calculate the scores for the module
    def get_scores(self, pl_module,ftrain_norm, ftest_q_norm, ftest_k_norm, food_q_norm, food_k_norm):
        ftrain_norm = self.tensorify_device(pl_module,ftrain_norm)
        ftest_q_norm = self.tensorify_device(pl_module,ftest_q_norm)
        ftest_k_norm = self.tensorify_device(pl_module,ftest_k_norm)
        food_q_norm = self.tensorify_device(pl_module,food_q_norm)
        food_k_norm = self.tensorify_device(pl_module,food_k_norm)

        loss_ID = self.ssl_loss(pl_module, ftrain_norm, ftest_q_norm, ftest_k_norm)
        loss_OOD = self.ssl_loss(pl_module, ftrain_norm, food_q_norm, food_k_norm)

        return loss_ID,loss_OOD
    
    # Change to a tensor and place on the correct device
    def tensorify_device(self,pl_module,data):
        data = torch.from_numpy(data)
        data = data.to(pl_module.device)
        return data

    # Performs the self supervised loss
    def ssl_loss(self,pl_module, ftrain, fdata_q, fdata_k):
        l_pos = torch.einsum('nc,nc->n', [fdata_q, fdata_k]).unsqueeze(-1) #Nawid - positive logit between output of key and query
        # negative logits: Nxr (train data are the negatives)
        l_neg = torch.einsum('nc,ck->nk', [fdata_q, ftrain.T.detach()]) # Nawid - negative logits (dot product between key and negative samples in a query bank)

        # logits: Nx(1+r)
        logits = torch.cat([l_pos, l_neg], dim=1) # Nawid - total logits - instance based loss to keep property of local smoothness
        logits /= pl_module.hparams.softmax_temperature
        
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long,device = pl_module.device)
        # calculates the loss for the approach

        # Obtain a loss for each dfferent value individually
        # Obtain a subset of the data
        # Prevent samples from being too large
        if len(logits) > 5000:
            logits = logits[0:5000]
            labels = labels[0:5000]
        
        loss = F.cross_entropy(logits, labels,reduction = 'none')
        loss = loss.data.cpu().numpy()
        
        # Increase by an several order of magnitude so that when the loss is quite low, it does become rounded to zero
        scaled_loss = loss * 10000
        return scaled_loss


    def get_eval_results(self,pl_module, ftrain,ftest_q,ftest_k, food_q,food_k):
        ftrain_norm, ftest_q_norm, ftest_k_norm, food_q_norm, food_k_norm = self.normalise(ftrain, ftest_q, ftest_k, food_q, food_k)
        loss_ID, loss_OOD = self.get_scores(pl_module,ftrain_norm, ftest_q_norm, ftest_k_norm, food_q_norm, food_k_norm)
        AUROC = get_roc_sklearn(loss_ID, loss_OOD)
        wandb.run.summary[self.summary_key] = AUROC
        self.datasaving(loss_ID,loss_OOD)

    def datasaving(self, ID_scores, OOD_scores):
        ID_scores = np.expand_dims(ID_scores,axis=1)
        OOD_scores = np.expand_dims(OOD_scores,axis=1)
        ID_df = pd.DataFrame(ID_scores)
        OOD_df = pd.DataFrame(OOD_scores)
        concat_df = pd.concat((ID_df,OOD_df),axis=1)
        concat_df = concat_df.applymap(lambda l: l if not np.isnan(l) else random.uniform(-2,-1))
        concat_df.columns = ['Instance Discrimination ID Loss Values', 'Instance Discrimination OOD Loss Values']
        

        table_data = wandb.Table(data=concat_df)
        wandb.log({f'Instance Discriminaton Loss Values OOD - {self.OOD_dataname}':table_data})



# Average representational similarity
class Average_Cosine_Similarity_OOD(Data_Augmented_Point_One_Dim_Class_Typicality_Normalised):
    def __init__(self, Datamodule, OOD_Datamodule, quick_callback: bool):
        super().__init__(Datamodule, OOD_Datamodule, quick_callback)

        # Make the test trasforms and the multi-transforms the same for the different datasets    
        self.OOD_Datamodule.test_transforms = self.Datamodule.test_transforms
        self.OOD_Datamodule.multi_transforms = self.Datamodule.multi_transforms #  Make the transform of the OOD data the same as the actual data
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly
        
        #
        self.num_augmentations = self.Datamodule.multi_transforms.num_augmentations

    def forward_callback(self, trainer, pl_module): 
        train_loader = self.Datamodule.deterministic_train_dataloader()

        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()

        multi_loader = self.Datamodule.multi_dataloader()
        multi_ood_loader = self.OOD_Datamodule.multi_dataloader()

        # obtain representations of the data for the test loaders and the multiloaders
        features_train, labels_train = self.get_features(pl_module, train_loader)
        features_test, _ = self.get_features(pl_module, test_loader)
        features_ood, labels_ood = self.get_features(pl_module,ood_loader)

        features_aug_test, _ = self.get_augmented_features(pl_module, multi_loader)
        features_aug_ood, _ = self.get_augmented_features(pl_module, multi_ood_loader)
    
        self.get_eval_results(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_aug_test),
            np.copy(features_ood),
            np.copy(features_aug_ood))
    
    def calculate_similarity(self, fdata, aug_fdata):
        # https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html
        cos = torch.nn.CosineSimilarity(dim=1)
        collated_similarity = []
        # shape of fdata is (num_datapoints, embeddim)
        # shape of aug_fdata is (num_augmentation, num_datapoints, embedim)
        # Calculate the cosine similarity between the unperturbed and the test using a loop
        for i in range(self.num_augmentations):
            similarity = cos(fdata, aug_fdata[i])
            collated_similarity.append(similarity)

        collated_similarity = torch.stack(collated_similarity,axis=1)
        average_similarity = torch.mean(collated_similarity,dim=1)

        # change to numpy
        average_similarity = average_similarity.data.cpu().numpy()
        return average_similarity
        

    def normalise(self,ftrain,ftest,aug_ftest, food, aug_food):
        ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10
        ftest /= np.linalg.norm(ftest, axis=-1, keepdims=True) + 1e-10
        food /= np.linalg.norm(food, axis=-1, keepdims=True) + 1e-10

        m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)
        # Nawid - normalise data using the mean and std
        ftrain = (ftrain - m) / (s + 1e-10)
        ftest = (ftest - m) / (s + 1e-10)
        food = (food - m) / (s + 1e-10)

        for i in range(self.num_augmentations):
            aug_ftest[i] /= np.linalg.norm(aug_ftest[i], axis=-1, keepdims=True) + 1e-10
            aug_ftest[i] = (aug_ftest[i] - m) / (s + 1e-10)

            aug_food[i] /= np.linalg.norm(aug_food[i], axis=-1, keepdims=True) + 1e-10
            aug_food[i] = (aug_food[i] - m) / (s + 1e-10)

        # change to torch tensors
        ftrain = torch.from_numpy(ftrain)
        ftest = torch.from_numpy(ftest)
        aug_ftest = torch.from_numpy(aug_ftest)
        food = torch.from_numpy(food)
        aug_food = torch.from_numpy(aug_food)

        return ftrain, ftest, aug_ftest, food, aug_food 

    def get_eval_results(self, ftrain, ftest, aug_ftest, food, aug_food):
        ftrain_norm, ftest_norm, aug_ftest_norm, food_norm,aug_food_norm = self.normalise(ftrain,ftest,aug_ftest, food, aug_food)

        id_average_similarity = self.calculate_similarity(ftest_norm, aug_ftest_norm)
        ood_average_similarity = self.calculate_similarity(food_norm, aug_food_norm)
        self.datasaving(id_average_similarity,ood_average_similarity)

    
    def datasaving(self,ID_scores, OOD_scores):
        ID_scores = np.expand_dims(ID_scores,axis=1)
        OOD_scores = np.expand_dims(OOD_scores,axis=1)
        ID_df = pd.DataFrame(ID_scores)
        OOD_df = pd.DataFrame(OOD_scores)
        concat_df = pd.concat((ID_df,OOD_df),axis=1)
        concat_df = concat_df.applymap(lambda l: l if not np.isnan(l) else random.uniform(-2,-1))
        
        concat_df.columns = ['ID Average Cosine Similarity', 'OOD Average Cosine Similarity']
        
        table_data = wandb.Table(data=concat_df)
        wandb.log({f'Data Augmented Cosine Similarity OOD - {self.OOD_dataname}':table_data})