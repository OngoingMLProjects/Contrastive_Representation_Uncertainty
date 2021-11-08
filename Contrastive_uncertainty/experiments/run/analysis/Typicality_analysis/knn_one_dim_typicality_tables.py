# Code for making tables for the ID and OOD datasets

from Contrastive_uncertainty.experiments.run.analysis.Typicality_analysis.knn_one_dim_typicality_diagrams import knn_auroc_table_v2
from numpy.core.numeric import full
from torch.utils.data import dataset
from Contrastive_uncertainty.general.callbacks.ood_callbacks import Mahalanobis_OOD
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import wilcoxon

# Import general params
import json
import re

from Contrastive_uncertainty.experiments.run.analysis.analysis_utils import dataset_dict, key_dict, ood_dataset_string, post_process_latex_table,full_post_process_latex_table

def knn_vector(json_data):
    data = np.array(json_data['data'])
    knn_values = np.around(data,decimals=3)
    return knn_values


def knn_auroc_table():
    # Fixed value of k of interest
    fixed_k = 10
    # Desired ID,OOD and Model  
    root_dir = 'run_data/'

    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py


    key_dict = {'dataset':{'MNIST':0, 'FashionMNIST':1,'KMNIST':2, 'CIFAR10':3, 'CIFAR100':4,'Caltech101':5,'Caltech256':6,'TinyImageNet':7,'Cub200':8,'Dogs':9},
                'model_type':{'CE':0,'SupCon':1}}
    
    all_ID = ['MNIST','FashionMNIST','KMNIST', 'CIFAR10','CIFAR100','Caltech101','Caltech256','TinyImageNet','Cub200','Dogs']
    for ID_dataset in all_ID: # Go through the different ID dataset                
        runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.epochs": 300, 'state':'finished',"config.dataset": f"{ID_dataset}","$or": [{"config.model_type":"SupCon" }, {"config.model_type": "CE"}]})
        # number of OOd datasets for this particular ID dataset
        num_ood = len(dataset_dict[ID_dataset])
        # data array
        data_array = np.empty((num_ood,2)) # 5 different measurements
        data_array[:] = np.nan
        row_names = [None] * num_ood # Make an empty list to take into account all the different values 
        for i, run in enumerate(runs): 
            run_summary = run.summary._json_dict
            # .config contains the hyperparameters.
            #  We remove special values that start with _.
            run_config = {k: v for k,v in run.config.items()
                 if not k.startswith('_')} 

            group_name = run_config['group']
            path_list = run.path
            # include the group name in the run path
            path_list.insert(-1, group_name)
            run_path = '/'.join(path_list)

            ID_dataset = run_config['dataset']
            model_type = run_config['model_type']
            Model_name = 'SupCLR' if model_type=='SupCon' else model_type
            # Make a data array, where the number values are equal to the number of OOD classes present in the datamodule dict or equal to the number of keys

            # name for the different rows of a table
            #row_names = []
            # https://stackoverflow.com/questions/10712002/create-an-empty-list-in-python-with-certain-size
            
            # go through the different knn keys
            
            # Obtain the ID and OOD dataset
            # Make function to obtain quadratic typicality for a particular ID and OOD dataset
            # Make function to obtain mahalanobis from a particular ID and OOD dataset
            desired_string = 'Normalized One Dim Class Quadratic Typicality KNN'.lower()

            # Obtain all the OOD datasets for a particular desired string
            all_OOD_datasets = obtain_ood_datasets(desired_string, run_summary,ID_dataset)
            for OOD_dataset in all_OOD_datasets:
                #print(f'ID:{ID_dataset}, OOD:{OOD_dataset}') 
                quadratic_auroc = obtain_knn_value(desired_string,run_summary,OOD_dataset)
                mahalanobis_auroc = obtain_baseline_mahalanobis(run_summary,OOD_dataset)
                data_index = dataset_dict[ID_dataset][OOD_dataset]
                #print(f'ID:{ID_dataset}, OOD:{OOD_dataset},Data index: {data_index}')
                data_array[data_index,0] = mahalanobis_auroc 
                data_array[data_index,1] = quadratic_auroc
                row_names[data_index] = f'ID:{ID_dataset}, OOD:{OOD_dataset}' 

            
        column_names = ['Mahalanobis', f'Quadratic {fixed_k} NN',]
        auroc_df = pd.DataFrame(data_array,columns = column_names, index=row_names)
        
        caption = ID_dataset + ' Dataset'
        label = f'tab:{ID_dataset}_Dataset'
        latex_table = full_post_process_latex_table(auroc_df, caption, label)
        
        print(latex_table)
        
# Obtain the OOD datasets for a particular string
def obtain_ood_datasets(desired_string,summary,ID_dataset):
    
    keys = [key for key, value in summary.items() if desired_string in key.lower()]
    all_OOD_datasets = []
    # obtain all the OOD datasets which are not known
    for key in keys:
        OOD_dataset = ood_dataset_string(key, dataset_dict, ID_dataset)
        if OOD_dataset is None or OOD_dataset == ID_dataset:
            pass
        else: 
            all_OOD_datasets.append(OOD_dataset)

    return all_OOD_datasets
# obtain the value from a specific OOD dataset
def obtain_knn_value(desired_string,summary,OOD_dataset):
    keys = [key for key, value in summary.items() if desired_string in key.lower()]
    # Need to split the key so that I can remove the datasets where the name is part of another name eg MNIST and KMNIST, or CIFAR10 and CIFAR100
    ood_dataset_specific_key = [key for key in keys if OOD_dataset.lower() in str.split(key.lower())]
    knn_auroc = round(summary[ood_dataset_specific_key[0]],3)
    return knn_auroc

# obtain the value for the mahalanobis for a particular situation
def obtain_baseline_mahalanobis(summary,OOD_dataset):

    baseline_string_1, baseline_string_2 = 'Mahalanobis AUROC OOD'.lower(), 'Mahalanobis AUROC: instance vector'.lower() 
    baseline_mahalanobis_keys_1 = [key for key, value in summary.items() if baseline_string_1 in key.lower()]
    baseline_mahalanobis_keys_1 = [key for key in baseline_mahalanobis_keys_1 if 'relative' not in key.lower()]

    baseline_mahalanobis_keys_2 = [key for key, value in summary.items() if baseline_string_2 in key.lower()]
    baseline_mahalanobis_keys_2 = [key for key in baseline_mahalanobis_keys_2 if 'relative' not in key.lower()]
    baseline_mahalanobis_keys = [*baseline_mahalanobis_keys_1,*baseline_mahalanobis_keys_2]

    # get the specific mahalanobis keys for the specific OOD dataset
    OOD_dataset_specific_mahalanobis_keys = [key for key in baseline_mahalanobis_keys if OOD_dataset.lower() in str.split(key.lower())]
    # https://stackoverflow.com/questions/16380326/check-if-substring-is-in-a-list-of-strings/16380569
    OOD_dataset_specific_mahalanobis_key = [key for key in OOD_dataset_specific_mahalanobis_keys if baseline_string_1 in key.lower()] 
    # Make it so that I choose baseline string 2 if the first case has no strings 
    if len(OOD_dataset_specific_mahalanobis_key) == 0:
        OOD_dataset_specific_mahalanobis_key = [key for key in OOD_dataset_specific_mahalanobis_keys if baseline_string_2 in key.lower()]
        # if there is no mahalanobis key for the KNN situation, pass otherwise plot graph
    if len(OOD_dataset_specific_mahalanobis_key) == 0:
        return None
    else:
        mahalanobis_AUROC = round(summary[OOD_dataset_specific_mahalanobis_key[0]],3)
        return mahalanobis_AUROC


if __name__== '__main__':
    knn_auroc_table()