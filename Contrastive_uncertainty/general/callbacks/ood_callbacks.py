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
from sklearn.ensemble import IsolationForest

from Contrastive_uncertainty.general.utils.ood_utils import get_measures # Used to calculate the AUROC, FPR and AUPR
from Contrastive_uncertainty.general.utils.hybrid_utils import OOD_conf_matrix
#from Contrastive_uncertainty.Contrastive.models.loss_functions import class_discrimination
from Contrastive_uncertainty.general.callbacks.general_callbacks import quickloading
from Contrastive_uncertainty.general.utils.pl_metrics import precision_at_k, mean

class Mahalanobis_OOD(pl.Callback):
    def __init__(self, Datamodule,OOD_Datamodule,
        quick_callback:bool = True):

        super().__init__()
        self.Datamodule = Datamodule
        self.OOD_Datamodule = OOD_Datamodule
        self.OOD_Datamodule.test_transforms = self.Datamodule.test_transforms #  Make the transform of the OOD data the same as the actual data
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly
        
      
        self.OOD_dataname = self.OOD_Datamodule.name
        self.summary_key = f'Mahalanobis AUROC OOD {self.OOD_dataname}'
        self.summary_aupr = self.summary_key.replace("AUROC", "AUPR")
        self.summary_fpr = self.summary_key.replace("AUROC", "FPR")
    
    '''
    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip if fast testing as this can lead to issues with the code
        self.forward_callback(trainer=trainer, pl_module=pl_module) 
    '''

    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer=trainer, pl_module=pl_module) 

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
        #self.visualize_data(trainer,pl_module, ood_loader)
        
        # Number of classes obtained from the max label value + 1 ( to take into account counting from zero)

        self.get_eval_results(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood),
            np.copy(labels_train))

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
    
    
    # Function just to visualise data, not actually required
    def visualize_data(self,trainer, pl_module, dataloader):
        features, labels = [], []
        loader = quickloading(self.quick_callback, dataloader)
        for index, (img, *label, indices) in enumerate(loader):
            assert len(loader)>0, 'loader is empty'
            if isinstance(img, tuple) or isinstance(img, list):
                    img, *aug_img = img # Used to take into accoutn whether the data is a tuple of the different augmentations
            
            from torchvision import transforms
            im = transforms.ToPILImage()(img[0]).convert("RGB")

            
            wandb.log({"img": [wandb.Image(im, caption="Cafe")]})
            
            wandb.log({"img tensor": [wandb.Image(img[0], caption="tensor")]})
            '''
            trainer.logger.experiment.log({
                'Image examining': [wandb.Image(x)
                                for x in img[:8]],
                "global_step": trainer.global_step #pl_module.current_epoch
                })
            '''

            

        return im


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
        auroc, aupr, fpr = get_measures(dood,dtest)

        #auroc= get_roc_sklearn(dtest, dood)
        wandb.run.summary[self.summary_key] = auroc
        wandb.run.summary[self.summary_aupr] = aupr
        wandb.run.summary[self.summary_fpr] = fpr
        
    
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




class IForest(pl.Callback):
    def __init__(self, Datamodule,OOD_Datamodule,
        quick_callback:bool = True):


        super().__init__()
        self.Datamodule = Datamodule
        self.OOD_Datamodule = OOD_Datamodule
        self.OOD_Datamodule.test_transforms = self.Datamodule.test_transforms #  Make the transform of the OOD data the same as the actual data
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly
   
        # Chooses what vector representation to use as well as which level of label hierarchy to use
        self.OOD_dataname = self.OOD_Datamodule.name
        
        self.summary_key = f'Isolation Forest AUROC OOD - {self.OOD_Datamodule.name}'
        self.summary_aupr = f'Isolation Forest AUPR OOD - {self.OOD_Datamodule.name}'
        self.summary_fpr = f'Isolation Forest FPR OOD - {self.OOD_Datamodule.name}'


    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip if fast testing as this can lead to issues with the code
        pass  
    
    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer=trainer, pl_module=pl_module) 

    
    def forward_callback(self, trainer, pl_module):
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA

        train_loader = self.Datamodule.deterministic_train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()

        # Obtain representations of the data
        features_train, labels_train = self.get_features(pl_module, train_loader, self.vector_level)
        features_test, labels_test = self.get_features(pl_module, test_loader, self.vector_level)
        features_ood, labels_ood = self.get_features(pl_module, ood_loader, self.vector_level)

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
      
    
    def get_scores(self,ftrain, ftest, food):
        clf = IsolationForest(random_state=).fit(ftrain)
        din = clf.predict(ftest)
        dood = clf.predict(food)
        return din,dood
        
    
    def get_eval_results(self, ftrain, ftest, food):
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain, ftest, food)
        din, dood = self.get_scores(ftrain_norm,ftest_norm, food_norm)
        # negate scores to make it so that the higher the score, the more abnormal it is (similar to the mahalanobis distance)
        auroc, aupr, fpr = get_measures(-dood,-din)
        
        wandb.run.summary[self.summary_key] = auroc
        wandb.run.summary[self.summary_aupr] = aupr
        wandb.run.summary[self.summary_fpr] = fpr

    
    
    


        




    
# Calculates the class wise mahalanobis distances and places the different values in a table
class Class_Mahalanobis_OOD(Mahalanobis_OOD):
    def __init__(self, Datamodule,OOD_Datamodule,
        quick_callback:bool = True):
        super().__init__(Datamodule,OOD_Datamodule, quick_callback)

    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer,pl_module)
    
    # Performs all the computation in the callback
    def forward_callback(self, trainer, pl_module):
        train_loader = self.Datamodule.deterministic_train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()

        # Obtain representations of the data
        features_train, labels_train = self.get_features(pl_module, train_loader)
        features_test, labels_test = self.get_features(pl_module, test_loader)
        features_ood, labels_ood = self.get_features(pl_module, ood_loader)

        self.get_eval_results(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood),
            np.copy(labels_train))


    def get_eval_results(self,ftrain, ftest, food, labelstrain):
        """
            None.
        """
        ftrain_norm,ftest_norm,food_norm = self.normalise(ftrain,ftest,food)
        # Nawid - obtain the scores for the test data and the OOD data
        
        dtest, dood, indices_dtest, indices_dood = self.get_scores(ftrain_norm, ftest_norm, food_norm, labelstrain)
        # Using actual labels to count the indices
        self.AUROC_saving(dtest, indices_dtest,
            dood,indices_dood,labelstrain,
            f'Class Wise Mahalanobis OOD {self.OOD_dataname} AUROC')


    def AUROC_saving(self,ID_scores,indices_ID, OOD_scores, indices_OOD,labels,wandb_name):
        # NEED TO MAKE IT SO THAT THE CLASS WISE VALUES CAN BE OBTAINED FOR THE TASK as well as the fraction of data points in a particular class
        table_data = {'Class':[], 'AUROC': [], 'ID Samples Fraction':[], 'OOD Samples Fraction':[]}
        class_ID_scores = [ID_scores[indices_ID==i] for i in np.unique(labels)]
        class_OOD_scores = [OOD_scores[indices_OOD==i] for i in np.unique(labels)]

        for class_num in range(len(np.unique(labels))):
            if len(class_ID_scores[class_num]) ==0 or len(class_OOD_scores[class_num])==0:
                class_AUROC = -1.0
            else:  
                class_AUROC = get_roc_sklearn(class_ID_scores[class_num],class_OOD_scores[class_num])
            
            class_ID_fraction = len(class_ID_scores[class_num])/len(ID_scores)
            class_OOD_fraction = len(class_OOD_scores[class_num])/len(OOD_scores)
            table_data['Class'].append(f'{class_num}')
            table_data['AUROC'].append(round(class_AUROC,2))
            table_data['ID Samples Fraction'].append(round(class_ID_fraction,2))
            table_data['OOD Samples Fraction'].append(round(class_OOD_fraction,2))

        # calculate the AUROC for the dataset in general
        All_AUROC = get_roc_sklearn(ID_scores,OOD_scores)
        #table_data['Class'].append(-1)
        table_data['Class'].append('All')
        table_data['AUROC'].append(round(All_AUROC,2))
        table_data['ID Samples Fraction'].append(1.0)
        table_data['OOD Samples Fraction'].append(1.0)


        table_df = pd.DataFrame(table_data)
        #print(table_df)
        table = wandb.Table(dataframe=table_df)
        wandb.log({wandb_name:table})

# Calculates the fraction of data points placed in each class (including the background class)
class Mahalanobis_OOD_Fractions(Class_Mahalanobis_OOD):
    def __init__(self, Datamodule, OOD_Datamodule, quick_callback: bool):
        super().__init__(Datamodule, OOD_Datamodule, quick_callback=quick_callback)
    
    def on_test_epoch_end(self, trainer, pl_module):
        return super().on_test_epoch_end(trainer, pl_module)
    
    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer,pl_module)

    def forward_callback(self, trainer, pl_module):
        return super().forward_callback(trainer, pl_module)
    
    def normalise(self, ftrain, ftest, food):
        return super().normalise(ftrain, ftest, food)

    # Updated to take into account background statistics
    def get_scores(self,ftrain, ftest, food, ypred):
        # Nawid - get all the features which belong to each of the different classes
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        # Look at joining data point for the particular model
        xc.append(ftrain) # Add all the data for the background class to see the performance of the model
        
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
        # Using actual labels to count the indices
        self.AUROC_saving(dtest, indices_dtest,
            dood,indices_dood,labelstrain,
            f'Class Wise Mahalanobis Extended OOD {self.OOD_dataname} AUROC')


    def AUROC_saving(self,ID_scores,indices_ID, OOD_scores, indices_OOD,labels,wandb_name):
        # NEED TO MAKE IT SO THAT THE CLASS WISE VALUES CAN BE OBTAINED FOR THE TASK as well as the fraction of data points in a particular class
        table_data = {'Class':[], 'AUROC': [], 'ID Samples Fraction':[], 'OOD Samples Fraction':[]}
        
        # Add +1 to take into account the background class
        class_ID_scores = [ID_scores[indices_ID==i] for i in range(len(np.unique(labels))+1)]
        class_OOD_scores = [OOD_scores[indices_OOD==i] for i in range(len(np.unique(labels))+1)]
    
        for class_num in range(len(np.unique(labels))+1): # +1 to correspond to the background class
            if len(class_ID_scores[class_num]) ==0 or len(class_OOD_scores[class_num])==0:
                class_AUROC = -1.0
            else:  
                class_AUROC = get_roc_sklearn(class_ID_scores[class_num],class_OOD_scores[class_num])
            
            class_ID_fraction = len(class_ID_scores[class_num])/len(ID_scores)
            class_OOD_fraction = len(class_OOD_scores[class_num])/len(OOD_scores)
            table_data['Class'].append(f'{class_num}')
            table_data['AUROC'].append(round(class_AUROC,2))
            table_data['ID Samples Fraction'].append(round(class_ID_fraction,2))
            table_data['OOD Samples Fraction'].append(round(class_OOD_fraction,2))

        # calculate the AUROC for the dataset in general
        All_AUROC = get_roc_sklearn(ID_scores,OOD_scores)
        table_data['Class'].append('All')
        table_data['AUROC'].append(round(All_AUROC,2))
        table_data['ID Samples Fraction'].append(1.0)
        table_data['OOD Samples Fraction'].append(1.0)
        
        
        background_class = len(np.unique(labels)) # background class is equal to the number of labels due to labels starting to count from 0 
        excluded_ID_scores = ID_scores[indices_ID !=background_class]
        excluded_OOD_scores = OOD_scores[indices_OOD !=background_class]
        if len(excluded_ID_scores)==0 or len(excluded_OOD_scores)==0:
            All_excluded_AUROC = -1.0
        else:              
            All_excluded_AUROC = get_roc_sklearn(excluded_ID_scores,excluded_OOD_scores)
        
        table_data['Class'].append('All Background Excluded')
        table_data['AUROC'].append(round(All_excluded_AUROC,2))
        table_data['ID Samples Fraction'].append(1.0)
        table_data['OOD Samples Fraction'].append(1.0)

        table_df = pd.DataFrame(table_data)
        table = wandb.Table(dataframe=table_df)
        wandb.log({wandb_name:table})


def get_roc_sklearn(xin, xood):
    labels = [0] * len(xin)  + [1] * len(xood)
    data = np.concatenate((xin, xood))
    auroc = skm.roc_auc_score(labels, data)
    return auroc

# calculates aupr
def get_pr_sklearn(xin, xood):
    labels = [0] * len(xin)  + [1] * len(xood)
    data = np.concatenate((xin, xood))
    aupr = skm.average_precision_score(labels, data)
    return aupr

# Nawid - calculate false positive rate
def get_fpr(xin, xood):
    return np.sum(xood < np.percentile(xin, 95)) / len(xood)

def get_roc_plot(xin, xood,OOD_name):
    anomaly_targets = [0] * len(xin)  + [1] * len(xood)
    outputs = np.concatenate((xin, xood))

    fpr, trp, thresholds = skm.roc_curve(anomaly_targets, outputs)
    plt.figure(figsize=(16,10))
    sns.scatterplot(
    x=fpr, y=trp,
    legend="full",
    alpha=0.3
    )
    # Set  x and y-axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    ROC_filename = f'Images/ROC_{OOD_name}.png'
    plt.savefig(ROC_filename)
    wandb_ROC = f'ROC curve: OOD dataset {OOD_name}'
    wandb.log({wandb_ROC:wandb.Image(ROC_filename)})

    '''
    wandb.log({f'ROC_{OOD_name}': wandb.plot.roc_curve(anomaly_targets, outputs,#scores,
                        labels=None, classes_to_plot=None)})
    '''
    

def count_histogram(input_data,num_bins,name):
    sns.displot(data = input_data,multiple ='stack',stat ='count',common_norm=False, bins=num_bins)#,kde =True)
    plt.xlabel('Distance')
    plt.ylabel('Counts')
    # Used to fix the x limit
    plt.xlim([0, 500])
    plt.title(f'Mahalanobis Distance Counts {name}')
    histogram_filename = f'Images/Mahalanobis_distance_counts_{name}.png'
    plt.savefig(histogram_filename,bbox_inches='tight')  #bbox inches used to make it so that the title can be seen effectively
    plt.close()
    wandb_distance = f'Mahalanobis Distance Counts {name}'
    wandb.log({wandb_distance:wandb.Image(histogram_filename)})
    

def probability_histogram(input_data,num_bins,name):
    sns.displot(data = input_data,multiple ='stack',stat ='probability',common_norm=False, bins=num_bins)#,kde =True)
    plt.xlabel('Distance')
    plt.ylabel('Probability')
    # Used to fix the x limit
    plt.xlim([0, 500])
    plt.ylim([0, 1])
    plt.title(f'Mahalanobis Distance Probabilities {name}')
    histogram_filename = f'Images/Mahalanobis_distances_probabilities_{name}.png'
    plt.savefig(histogram_filename,bbox_inches='tight')  #bbox inches used to make it so that the title can be seen effectively
    plt.close()
    wandb_distance = f'Mahalanobis Distance Probabilities {name}'
    wandb.log({wandb_distance:wandb.Image(histogram_filename)})
    
def kde_plot(input_data,name):
    sns.displot(data =input_data,fill=False,common_norm=False,kind='kde')
    plt.xlabel('Distance')
    plt.ylabel('Normalized Density')
    plt.xlim([0, 500])
    plt.title(f'Mahalanobis Distances {name}')
    kde_filename = f'Images/Mahalanobis_distances_kde_{name}.png'
    plt.savefig(kde_filename,bbox_inches='tight')
    plt.close()
    wandb_distance = f'Mahalanobis Distance KDE {name}'
    wandb.log({wandb_distance:wandb.Image(kde_filename)})
    
def pairwise_saving(collated_data,dataset_names,num_bins,ref_index):
    table_data = {'Dataset':[],'Count Absolute Deviation':[],'Prob Absolute Deviation':[],'KL (Nats)':[], 'JS (Nats)':[],'KS':[]}
    # Calculates the values in a pairwise 
    # Calculate the name of the data based on the ref index
    assert ref_index ==0 or ref_index ==1,"ref index only can be 0 or 1 currently" 
    if ref_index == 0:
        ref = 'Train'
    else:
        ref = 'Test'
    
    for i in range(len(collated_data)-(1+ref_index)):
        pairwise_dict = {}
        # Update the for base case 
        index_val = 1 + i +ref_index
        pairwise_dict.update({dataset_names[ref_index]:collated_data[ref_index]})
        pairwise_dict.update({dataset_names[index_val]:collated_data[index_val]})
        data_label = f'{ref} Reference - {dataset_names[index_val]}' 
        
        # Plots the counts, probabilities as well as the kde for pairwise
        count_histogram(pairwise_dict,num_bins,data_label)
        probability_histogram(pairwise_dict,num_bins,data_label)
        kde_plot(pairwise_dict,data_label)
                    
        # https://www.kite.com/python/answers/how-to-plot-a-histogram-given-its-bins-in-python 
        # Plots the histogram of the pairwise distance
        count_hist1, _ = np.histogram(pairwise_dict[dataset_names[ref_index]],range=(0,500), bins = num_bins)
        count_hist2, bin_edges = np.histogram(pairwise_dict[dataset_names[index_val]],range=(0,500), bins = num_bins)
        count_absolute_deviation  = np.sum(np.absolute(count_hist1 - count_hist2))

        # Using density =  True is the same as making it so that you normalise each term by the sum of the counts
        prob_hist1, _ = np.histogram(pairwise_dict[dataset_names[ref_index]],range=(0,500), bins = num_bins,density = True)
        prob_hist2, bin_edges = np.histogram(pairwise_dict[dataset_names[index_val]],range=(0,500), bins = num_bins,density= True)
        prob_absolute_deviation  = round(np.sum(np.absolute(prob_hist1 - prob_hist2)),3)
        kl_div = round(kl_divergence(pairwise_dict[dataset_names[ref_index]], pairwise_dict[dataset_names[index_val]]),3)
        js_div = round(js_metric(pairwise_dict[dataset_names[ref_index]], pairwise_dict[dataset_names[index_val]]),3)
        ks_stat = round(ks_statistic_kde(pairwise_dict[dataset_names[ref_index]], pairwise_dict[dataset_names[index_val]]),3)
        
        table_data['Dataset'].append(data_label)
        table_data['Count Absolute Deviation'].append(count_absolute_deviation)
        table_data['Prob Absolute Deviation'].append(prob_absolute_deviation)
        table_data['KL (Nats)'].append(kl_div)
        table_data['JS (Nats)'].append(js_div)
        table_data['KS'].append(ks_stat)
    
    table_df = pd.DataFrame(table_data)
    
    table = wandb.Table(dataframe=table_df)
    wandb.log({f"{ref} Distance statistics": table})
    table_saving(table_df,f'Mahalanobis Distance {ref} Statistics')
    

def table_saving(table_dataframe,name):
    fig, ax = plt.subplots()
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    #https://stackoverflow.com/questions/15514005/how-to-change-the-tables-fontsize-with-matplotlib-pyplot
    data_table = ax.table(cellText=table_dataframe.values, colLabels=table_dataframe.columns, loc='center')
    data_table.set_fontsize(24)
    data_table.scale(2.0, 2.0)  # may help
    filename = name.replace(" ","_") # Change the values with the empty space to underscore
    filename = f'Images/{filename}.png'
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    wandb_title = f'{name}'
    wandb.log({wandb_title:wandb.Image(filename)})


def calculate_class_ROC( class_ID_scores, class_OOD_scores):
    if len(class_ID_scores) ==0 or len(class_OOD_scores)==0:
        class_AUROC = -1.0
    else:
        class_AUROC = get_roc_sklearn(class_ID_scores, class_OOD_scores)
            
    return class_AUROC