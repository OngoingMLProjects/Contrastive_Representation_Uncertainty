# Calculating the values as the value of K changes for the typicality based approach


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

import json
import re


from Contrastive_uncertainty.experiments.run.analysis.analysis_utils import dataset_dict, key_dict, ood_dataset_string
from Contrastive_uncertainty.experiments.run.analysis.Typicality_analysis.knn_one_dim_typicality_tables import obtain_ood_datasets

def knn_vector(json_data):
    data = np.array(json_data['data'])
    knn_values = np.around(data,decimals=3)
    return knn_values

def knn_vector(json_data):
    data = np.array(json_data['data'])
    knn_values = np.around(data,decimals=3)
    return knn_values


# Change to allow using quadratic typicality
def knn_auroc_robustness():
    # Desired ID,OOD and Model  
    root_dir = 'run_data/'

    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py
    

    run_filter={"config.group":"OOD hierarchy baselines","config.model_type": "SupCon","config.dataset": "CIFAR100"}    
    runs = api.runs(path="nerdk312/evaluation", filters=run_filter)
    #runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.model_type": "SupCon","config.epochs": 300})
    higher_counter = 0
    lower_counter = 0    
    key_dict = {'dataset':{'CIFAR10':0, 'CIFAR100':1,'Caltech101':2,'Caltech256':3,'TinyImageNet':4,'Cub200':4,'Dogs':5},
                'model_type':{'SupCon':0}}
    all_ID = ['CIFAR10','CIFAR100','Caltech101','Caltech256','TinyImageNet','Cub200','Dogs']
    for ID_dataset in all_ID: # Go through the different ID dataset                
        
        runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.epochs": 300, 'state':'finished',"config.dataset": f"{ID_dataset}","config.model_type":"SupCon" })
        for i,run in enumerate(runs):
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

            desired_string = 'Different K Normalized Quadratic One Dim Class Typicality KNN'.lower()
            linear_string = 'Different K Normalized One Dim Class Typicality KNN OOD'.lower()

            all_OOD_datasets = obtain_ood_datasets(desired_string, run_summary,ID_dataset)

            for OOD_dataset in all_OOD_datasets:
                quadratic_std = knn_std(desired_string,OOD_dataset,run_summary,root_dir,run_path)
                linear_std = knn_std(linear_string,OOD_dataset,run_summary,root_dir,run_path)
                
                higher_counter += quadratic_std
                lower_counter += linear_std
                '''
                if quadratic_std < linear_std:
                    print('yes')
                    lower_counter += 1
                else:
                    print('no')
                    higher_counter +=1
                '''
                #print('quadratic std',quadratic_std)
                #print('linear std',linear_std)
    '''
    for ID_dataset in all_ID: # Go through the different ID dataset                
        
        runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.epochs": 300, 'state':'finished',"config.dataset": f"{ID_dataset}","config.model_type":"SupCon" })
        
        for i, run in enumerate(runs): 
            

            summary_list.append(run.summary._json_dict)
            # .config contains the hyperparameters.
            #  We remove special values that start with _.
            config_list.append(
                {k: v for k,v in run.config.items()
                 if not k.startswith('_')})

            group_name = config_list[i]['group'] # get the name of the group
            path_list = runs[i].path

            # include the group name in the run path
            path_list.insert(-1, group_name)
            run_path = '/'.join(path_list)

            ID_dataset = config_list[i]['dataset']
            model_type = config_list[i]['model_type']
            Model_name = 'SupCLR' if model_type=='SupCon' else model_type
            # .name is the human-readable name of the run.dir
            name_list.append(run.name)

            desired_string = 'Different K Normalized Quadratic One Dim Class Typicality KNN'.lower()
            linear_string = 'Different K Normalized One Dim Class Typicality KNN OOD'.lower()

            keys = [key for key, value in summary_list[i].items() if desired_string in key.lower()]
            keys = [key for key in keys if 'table' not in key.lower()]
            for key in keys:
                if run_path == 'nerdk312/evaluation/OOD hierarchy baselines/p4ojvcp2':
                    data_dir = summary_list[i][key]['path'] 
                    run_dir = root_dir + run_path
                    print('Initial data dir',data_dir)
                
            all_OOD_datasets = obtain_ood_datasets(desired_string, summary_list[i],ID_dataset)
            
            for OOD_dataset in all_OOD_datasets:
                quadratic_std = knn_std(desired_string,OOD_dataset,summary_list[i],root_dir,run_path)
                #linear_std = knn_std(linear_string,OOD_dataset,summary_list[i],root_dir,run_path)
    '''         

            
                        


def knn_std(string,OOD_dataset,summary,root_dir,run_path):
    
    keys = [key for key, value in summary.items() if string in key.lower()]
    
    ood_dataset_specific_key = [key for key in keys if OOD_dataset.lower() in str.split(key.lower())]
    #print(ood_dataset_specific_key)
    data_dir = summary[ood_dataset_specific_key[0]]['path']
    '''
    if run_path == 'nerdk312/evaluation/OOD hierarchy baselines/p4ojvcp2':
        print('data dir:',data_dir)
        #import ipdb; ipdb.set_trace()
    '''
    run_dir = root_dir + run_path
    read_dir = run_dir + '/' + data_dir
    with open(read_dir) as f: 
        data = json.load(f)
    knn_values = knn_vector(data)
    knn_auroc_values = knn_values[:,1]
    knn_std = round(np.std(knn_auroc_values),4)
    return knn_std


if __name__ == '__main__':
    knn_auroc_robustness()