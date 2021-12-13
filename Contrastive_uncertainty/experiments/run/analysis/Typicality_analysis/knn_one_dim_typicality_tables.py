# Code for making tables for the ID and OOD datasets

from numpy.core.defchararray import count
from pandas.core import base
from torch.utils import data
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
import copy

from Contrastive_uncertainty.experiments.run.analysis.analysis_utils import add_baseline_names_row, collated_baseline_post_process_latex_table, combine_multiple_tables, dataset_dict, key_dict, ood_dataset_string, post_process_latex_table,full_post_process_latex_table, remove_hline_processing, separate_top_columns, single_baseline_post_process_latex_table, collated_multiple_baseline_post_process_latex_table,combine_multiple_tables, separate_columns, separate_top_columns, update_double_col_table, update_headings_additional,\
    collated_multiple_baseline_post_process_latex_table_insignificance, separate_ID_datasets

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


# Calculates the mean AUROC value compared to a single value
def knn_auroc_table_mean():
    suffix = 'FPR'
    desired_string = 'Normalized One Dim Class Quadratic Typicality KNN - 10 '
    baseline_string_1 = 'Mahalanobis '
    if suffix == 'OOD':
        desired_string = (desired_string + suffix).lower()
        baseline_string_1 = (baseline_string_1 +'AUROC '+ suffix).lower()
    else:
        desired_string = (desired_string + suffix).lower()
        baseline_string_1 = (baseline_string_1 + suffix).lower()
    
    baseline_string_2 = baseline_string_1    
    value = 'min' if suffix =='FPR' else 'max'
    
    
    # Make it so that the desired string and the baseline strings are decided by the suffix (ood, FPR, AUPR)

    #desired_string = 'Normalized One Dim Class Quadratic Typicality KNN - 10 OOD'.lower() # Only get the key for the AUROC
    #desired_string = 'Normalized One Dim Class Quadratic Typicality KNN - 10 AUPR'.lower()

    #desired_string = 'Normalized One Dim Class Quadratic Typicality KNN - 10 FPR'.lower()
    #baseline_string_1,baseline_string_2 = 'Mahalanobis FPR'.lower(),'Mahalanobis FPR'.lower()
    '''
    baseline_string_1, baseline_string_2 = 'Mahalanobis AUPR'.lower(),'Mahalanobis AUPR'.lower()
    FPR= False
    value = 'min' if FPR else 'max' # Used to control whether to bold the max or the min values
    '''
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
        #runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.epochs": 300, 'state':'finished',"config.dataset": f"{ID_dataset}","$or": [{"config.model_type":"SupCon" }, {"config.model_type": "CE"}]})
        runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats","config.epochs": 300, "config.dataset": f"{ID_dataset}","config.model_type":"SupCon"})
        # number of OOd datasets for this particular ID dataset
        num_ood = len(dataset_dict[ID_dataset])
        # data array
        data_array = np.zeros((num_ood,2)) # 5 different measurements
        count_array = np.zeros((num_ood,2)) # 5 different measurements
        

        '''
        data_array = np.empty((num_ood,2)) # 5 different measurements
        data_array[:] = np.nan

        count_array = np.empty((num_ood,2)) # 5 different measurements
        count_array[:] = np.nan
        '''

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
            
           
        

            # Obtain all the OOD datasets for a particular desired string
            all_OOD_datasets = obtain_ood_datasets(desired_string, run_summary,ID_dataset)
            for OOD_dataset in all_OOD_datasets:
                #print(f'ID:{ID_dataset}, OOD:{OOD_dataset}') 
                quadratic_auroc = obtain_knn_value(desired_string,run_summary,OOD_dataset)
                mahalanobis_auroc = obtain_baseline_mahalanobis(run_summary,OOD_dataset,baseline_string_1,baseline_string_2)
                data_index = dataset_dict[ID_dataset][OOD_dataset] # index location
                #print(f'ID:{ID_dataset}, OOD:{OOD_dataset},Data index: {data_index}')
                data_array[data_index,0] += mahalanobis_auroc 
                data_array[data_index,1] += quadratic_auroc
                count_array[data_index,0] += 1
                count_array[data_index,1] += 1
                row_names[data_index] = f'ID:{ID_dataset}, OOD:{OOD_dataset}' 
        #import ipdb; ipdb.set_trace()
        data_array = np.round(data_array/count_array,decimals=3)    
        column_names = ['Mahalanobis', f'Quadratic {fixed_k} NN',]
        auroc_df = pd.DataFrame(data_array,columns = column_names, index=row_names)
        
        caption = ID_dataset + ' Dataset'
        label = f'tab:{ID_dataset}_Dataset'
        latex_table = single_baseline_post_process_latex_table(auroc_df, caption, label,value)
        #latex_table = full_post_process_latex_table(auroc_df, caption, label,value)
        
        print(latex_table)



# Calculates the AUROC, AUPR as well as the false positive rate
def knn_auroc_table_collated():
    
    # Make it so that the desired string and the baseline strings are decided by the suffix (ood, FPR, AUPR)

    desired_string_AUROC = 'Normalized One Dim Class Quadratic Typicality KNN - 10 OOD'.lower() # Only get the key for the AUROC
    desired_string_AUPR= 'Normalized One Dim Class Quadratic Typicality KNN - 10 AUPR'.lower()
    desired_string_FPR = 'Normalized One Dim Class Quadratic Typicality KNN - 10 FPR'.lower()

    baseline_string_1_AUROC,baseline_string_2_AUROC = 'Mahalanobis AUROC OOD'.lower(),'Mahalanobis AUROC OOD'.lower()
    baseline_string_1_AUPR,baseline_string_2_AUPR = 'Mahalanobis AUPR'.lower(),'Mahalanobis AUPR'.lower()
    baseline_string_1_FPR,baseline_string_2_FPR = 'Mahalanobis FPR'.lower(),'Mahalanobis FPR'.lower()

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
        #runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.epochs": 300, 'state':'finished',"config.dataset": f"{ID_dataset}","$or": [{"config.model_type":"SupCon" }, {"config.model_type": "CE"}]})
        runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats","config.epochs": 300, "config.dataset": f"{ID_dataset}","config.model_type":"SupCon"})
        # number of OOd datasets for this particular ID dataset
        num_ood = len(dataset_dict[ID_dataset])
        # data array
        data_array_AUROC = np.zeros((num_ood,2)) # 5 different measurements
        count_array_AUROC = np.zeros((num_ood,2)) # 5 different measurements

        data_array_AUPR = np.zeros((num_ood,2)) # 5 different measurements
        count_array_AUPR = np.zeros((num_ood,2)) # 5 different measurements

        data_array_FPR = np.zeros((num_ood,2)) # 5 different measurements
        count_array_FPR = np.zeros((num_ood,2)) # 5 different measurements


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
            
            # Obtain all the OOD datasets for a particular desired string
            all_OOD_datasets = obtain_ood_datasets(desired_string_AUROC, run_summary,ID_dataset)
            for OOD_dataset in all_OOD_datasets:
                #print(f'ID:{ID_dataset}, OOD:{OOD_dataset}') 
                quadratic_auroc = obtain_knn_value(desired_string_AUROC,run_summary,OOD_dataset)
                mahalanobis_auroc = obtain_baseline_mahalanobis(run_summary,OOD_dataset,baseline_string_1_AUROC,baseline_string_2_AUROC)
                
                quadratic_aupr = obtain_knn_value(desired_string_AUPR,run_summary,OOD_dataset)
                mahalanobis_aupr = obtain_baseline_mahalanobis(run_summary,OOD_dataset,baseline_string_1_AUPR,baseline_string_2_AUPR)

                quadratic_fpr = obtain_knn_value(desired_string_FPR,run_summary,OOD_dataset)
                mahalanobis_fpr = obtain_baseline_mahalanobis(run_summary,OOD_dataset,baseline_string_1_FPR,baseline_string_2_FPR)

                data_index = dataset_dict[ID_dataset][OOD_dataset] # index location
                #print(f'ID:{ID_dataset}, OOD:{OOD_dataset},Data index: {data_index}')
                data_array_AUROC[data_index,0] += mahalanobis_auroc 
                data_array_AUROC[data_index,1] += quadratic_auroc
                count_array_AUROC[data_index,0] += 1
                count_array_AUROC[data_index,1] += 1


                data_array_AUPR[data_index,0] += mahalanobis_aupr 
                data_array_AUPR[data_index,1] += quadratic_aupr
                count_array_AUPR[data_index,0] += 1
                count_array_AUPR[data_index,1] += 1


                data_array_FPR[data_index,0] += mahalanobis_fpr 
                data_array_FPR[data_index,1] += quadratic_fpr
                count_array_FPR[data_index,0] += 1
                count_array_FPR[data_index,1] += 1


                row_names[data_index] = f'ID:{ID_dataset}, OOD:{OOD_dataset}' 
        
        data_array_AUROC = np.round(data_array_AUROC/count_array_AUROC,decimals=3)  
        data_array_AUPR = np.round(data_array_AUPR/count_array_AUPR,decimals=3)  
        data_array_FPR = np.round(data_array_FPR/count_array_FPR,decimals=3)

        column_names = ['Mahalanobis', f'Quadratic {fixed_k} NN',]
        auroc_df = pd.DataFrame(data_array_AUROC,columns = column_names, index=row_names)
        aupr_df = pd.DataFrame(data_array_AUPR,columns = column_names, index=row_names)
        fpr_df = pd.DataFrame(data_array_FPR,columns = column_names, index=row_names)
        
        caption = ID_dataset + ' Dataset'
        label = f'tab:{ID_dataset}_Dataset'
        #latex_table = single_baseline_post_process_latex_table(auroc_df, caption, label,value)
        latex_table = collated_baseline_post_process_latex_table(auroc_df,aupr_df, fpr_df,caption, label)
        #latex_table = full_post_process_latex_table(auroc_df, caption, label,value)
        
        print(latex_table)


# Calculates the AUROC, AUPR as well as the false positive rate
def knn_table_collated(desired_approach = 'Quadratic_typicality', desired_model_type = 'SupCon', baseline_approach = 'Mahalanobis', baseline_model_type = 'CE'):
    
    # Make it so that the desired string and the baseline strings are decided by the suffix (ood, FPR, AUPR)
    if desired_approach == 'Quadratic_typicality':
        desired_string_AUROC = 'Normalized One Dim Class Quadratic Typicality KNN - 10 OOD'.lower() # Only get the key for the AUROC
        desired_string_AUPR= 'Normalized One Dim Class Quadratic Typicality KNN - 10 AUPR'.lower()
        desired_string_FPR = 'Normalized One Dim Class Quadratic Typicality KNN - 10 FPR'.lower()
        desired_function = obtain_knn_value # Used to calculate the value
    #desired_model_type = 'SupCon'

    if baseline_approach =='Mahalanobis':
        baseline_string_AUROC = 'Mahalanobis AUROC OOD'.lower()
        baseline_string_AUPR = 'Mahalanobis AUPR'.lower()
        baseline_string_FPR = 'Mahalanobis FPR'.lower()
    elif baseline_approach =='Softmax':        
        baseline_string_AUROC = 'Maximum Softmax Probability AUROC OOD'.lower()
        baseline_string_AUPR = 'Maximum Softmax Probability AUPR OOD'.lower()
        baseline_string_FPR = 'Maximum Softmax Probability FPR OOD'.lower()
    else:
        assert baseline_approach == 'Mahalanobis' or baseline_approach =='Softmax', 'No other baselines implemented'
        

    baseline_function = obtain_baseline
    #baseline_model_type = 'SupCon 
    
    # Fixed value of k of interest
    fixed_k = 10
    # Desired ID,OOD and Model  
    root_dir = 'run_data/'

    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py

    all_ID = ['MNIST','FashionMNIST','KMNIST', 'CIFAR10','CIFAR100','Caltech101','Caltech256','TinyImageNet','Cub200','Dogs']
    for ID_dataset in all_ID: # Go through the different ID dataset                
        #runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.epochs": 300, 'state':'finished',"config.dataset": f"{ID_dataset}","$or": [{"config.model_type":"SupCon" }, {"config.model_type": "CE"}]})
        runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats","config.epochs": 300, 'state':'finished',"config.dataset": f"{ID_dataset}","$or": [{"config.model_type":"SupCon" }, {"config.model_type": "CE"}]})
        #runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats","config.epochs": 300, "config.dataset": f"{ID_dataset}","config.model_type":"SupCon"})
        #runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats","config.epochs": 300, "config.dataset": f"{ID_dataset}","config.model_type":"CE"})
        # number of OOd datasets for this particular ID dataset
        num_ood = len(dataset_dict[ID_dataset])
        # data array
        data_array_AUROC = np.zeros((num_ood,2)) # 5 different measurements
        count_array_AUROC = np.zeros((num_ood,2)) # 5 different measurements

        data_array_AUPR = np.zeros((num_ood,2)) # 5 different measurements
        count_array_AUPR = np.zeros((num_ood,2)) # 5 different measurements

        data_array_FPR = np.zeros((num_ood,2)) # 5 different measurements
        count_array_FPR = np.zeros((num_ood,2)) # 5 different measurements


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
            
            # Obtain all the OOD datasets for a particular desired string
            all_OOD_datasets = obtain_ood_datasets_baseline(desired_string_AUROC,baseline_string_AUROC, run_summary,ID_dataset)
            for OOD_dataset in all_OOD_datasets:
                data_index = dataset_dict[ID_dataset][OOD_dataset] # index location
                # updates the data array and the count array at a certain location with
                
                data_array_AUROC, count_array_AUROC =update_metric_and_count(data_array_AUROC,count_array_AUROC,data_index,1,desired_function,desired_string_AUROC,desired_model_type,model_type,run_summary,OOD_dataset)
                data_array_AUPR, count_array_AUPR =update_metric_and_count(data_array_AUPR,count_array_AUPR,data_index,1,desired_function,desired_string_AUPR,desired_model_type,model_type,run_summary,OOD_dataset)
                data_array_FPR, count_array_FPR =update_metric_and_count(data_array_FPR,count_array_FPR,data_index,1,desired_function,desired_string_FPR,desired_model_type,model_type,run_summary,OOD_dataset)

                data_array_AUROC, count_array_AUROC =update_metric_and_count(data_array_AUROC,count_array_AUROC,data_index,0,baseline_function,baseline_string_AUROC,baseline_model_type,model_type,run_summary,OOD_dataset)
                data_array_AUPR, count_array_AUPR =update_metric_and_count(data_array_AUPR,count_array_AUPR,data_index,0,baseline_function,baseline_string_AUPR,baseline_model_type,model_type,run_summary,OOD_dataset)
                data_array_FPR, count_array_FPR =update_metric_and_count(data_array_FPR,count_array_FPR,data_index,0,baseline_function,baseline_string_FPR,baseline_model_type,model_type,run_summary,OOD_dataset)

                row_names[data_index] = f'ID:{ID_dataset}, OOD:{OOD_dataset}' 
                
        data_array_AUROC = np.round(data_array_AUROC/count_array_AUROC,decimals=3)  
        data_array_AUPR = np.round(data_array_AUPR/count_array_AUPR,decimals=3)  
        data_array_FPR = np.round(data_array_FPR/count_array_FPR,decimals=3)

        column_names = ['Mahalanobis', f'Quadratic {fixed_k} NN',]
        auroc_df = pd.DataFrame(data_array_AUROC,columns = column_names, index=row_names)
        aupr_df = pd.DataFrame(data_array_AUPR,columns = column_names, index=row_names)
        fpr_df = pd.DataFrame(data_array_FPR,columns = column_names, index=row_names)
        #desired_approach.split("_")

        caption = ID_dataset + ' Dataset'+ f' with {desired_approach.replace("_"," ")} {desired_model_type} vs {baseline_approach.replace("_"," ")} {baseline_model_type} Baseline'  # replace Underscore with spaces for the caption
        label = f'tab:{ID_dataset}_Dataset_{desired_approach}_{desired_model_type}_{baseline_approach}_{baseline_model_type}'
        #latex_table = single_baseline_post_process_latex_table(auroc_df, caption, label,value)
        latex_table = collated_baseline_post_process_latex_table(auroc_df,aupr_df, fpr_df,caption, label)
        #latex_table = full_post_process_latex_table(auroc_df, caption, label,value='max')
        
        print(latex_table)



# Calculates the AUROC, AUPR as well as the false positive rate - Pass in list to calculate all baselines
def knn_table_collated_v2(desired_approach = 'Quadratic_typicality', desired_model_type = 'SupCon', baseline_approaches = ['Softmax','Mahalanobis'], baseline_model_types = ['CE','CE'],dataset_type ='grayscale'):

    baselines_dict = {'Mahalanobis':{'AUROC':'Mahalanobis AUROC OOD'.lower(),'AUPR':'Mahalanobis AUPR'.lower(),'FPR':'Mahalanobis FPR'.lower()},
                
                'Softmax':{'AUROC':'Maximum Softmax Probability AUROC OOD'.lower(),'AUPR':'Maximum Softmax Probability AUPR OOD'.lower(),'FPR':'Maximum Softmax Probability FPR OOD'.lower()},
                }

    assert len(baseline_approaches) == len(baseline_model_types), 'number of baseline approaches do not match number of baseline models'
    num_baselines = len(baseline_approaches)
    # Make it so that the desired string and the baseline strings are decided by the suffix (ood, FPR, AUPR)
    if desired_approach == 'Quadratic_typicality':
        desired_string_AUROC = 'Normalized One Dim Class Quadratic Typicality KNN - 10 OOD'.lower() # Only get the key for the AUROC
        desired_string_AUPR= 'Normalized One Dim Class Quadratic Typicality KNN - 10 AUPR'.lower()
        desired_string_FPR = 'Normalized One Dim Class Quadratic Typicality KNN - 10 FPR'.lower()
        desired_function = obtain_knn_value # Used to calculate the value

    ####################
    # make a loop to add the baseline strings and the baseline AUPR etc
    baseline_strings_AUROC = []
    baseline_strings_AUPR = []
    baseline_strings_FPR = []
    for approach in baseline_approaches:
        assert approach == 'Mahalanobis' or approach =='Softmax', 'No other baselines implemented'
        baseline_strings_AUROC.append(baselines_dict[approach]['AUROC'])
        baseline_strings_AUPR.append(baselines_dict[approach]['AUPR'])
        baseline_strings_FPR.append(baselines_dict[approach]['FPR'])

    baseline_function = obtain_baseline
    
    # Fixed value of k of interest
    fixed_k = 10
    # Desired ID,OOD and Model  
    root_dir = 'run_data/'

    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py
    all_latex_tables = []
    all_ID = ['MNIST','FashionMNIST','KMNIST'] if dataset_type =='grayscale' else ['CIFAR10','CIFAR100','Caltech256','TinyImageNet'] 
    #all_ID = ['MNIST','FashionMNIST','KMNIST', 'CIFAR10','CIFAR100','Caltech101','Caltech256','TinyImageNet','Cub200','Dogs']
    #all_ID = ['MNIST','FashionMNIST','KMNIST']
    for ID_dataset in all_ID: # Go through the different ID dataset                
        #runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.epochs": 300, 'state':'finished',"config.dataset": f"{ID_dataset}","$or": [{"config.model_type":"SupCon" }, {"config.model_type": "CE"}]})
        runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats","config.epochs": 300, 'state':'finished',"config.dataset": f"{ID_dataset}","$or": [{"config.model_type":"SupCon" }, {"config.model_type": "CE"}]})
        #runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats","config.epochs": 300, "config.dataset": f"{ID_dataset}","config.model_type":"SupCon"})
        #runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats","config.epochs": 300, "config.dataset": f"{ID_dataset}","config.model_type":"CE"})
        # number of OOd datasets for this particular ID dataset
        num_ood = len(dataset_dict[ID_dataset])
        # data array
        data_array_AUROC = np.zeros((num_ood,num_baselines+1)) 
        count_array_AUROC = np.zeros((num_ood,num_baselines+1))  

        data_array_AUPR = np.zeros((num_ood,num_baselines+1)) 
        count_array_AUPR = np.zeros((num_ood,num_baselines+1)) 

        data_array_FPR = np.zeros((num_ood,num_baselines+1)) 
        count_array_FPR = np.zeros((num_ood,num_baselines+1)) 

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
            
            # Obtain all the OOD datasets for a particular desired string
            all_OOD_datasets = obtain_ood_datasets_baseline(desired_string_AUROC,baseline_strings_AUROC[0], run_summary,ID_dataset)
            for OOD_dataset in all_OOD_datasets:
                data_index = dataset_dict[ID_dataset][OOD_dataset] # index location
                # updates the data array and the count array at a certain location with
                
                data_array_AUROC, count_array_AUROC =update_metric_and_count(data_array_AUROC,count_array_AUROC,data_index,-1,desired_function,desired_string_AUROC,desired_model_type,model_type,run_summary,OOD_dataset)
                data_array_AUPR, count_array_AUPR =update_metric_and_count(data_array_AUPR,count_array_AUPR,data_index,-1,desired_function,desired_string_AUPR,desired_model_type,model_type,run_summary,OOD_dataset)
                data_array_FPR, count_array_FPR =update_metric_and_count(data_array_FPR,count_array_FPR,data_index,-1,desired_function,desired_string_FPR,desired_model_type,model_type,run_summary,OOD_dataset)

                for i in range(len(baseline_approaches)):

                    data_array_AUROC, count_array_AUROC =update_metric_and_count(data_array_AUROC,count_array_AUROC,data_index,i,baseline_function,baseline_strings_AUROC[i],baseline_model_types[i],model_type,run_summary,OOD_dataset)
                    data_array_AUPR, count_array_AUPR =update_metric_and_count(data_array_AUPR,count_array_AUPR,data_index,i,baseline_function,baseline_strings_AUPR[i],baseline_model_types[i],model_type,run_summary,OOD_dataset)
                    data_array_FPR, count_array_FPR =update_metric_and_count(data_array_FPR,count_array_FPR,data_index,i,baseline_function,baseline_strings_FPR[i],baseline_model_types[i],model_type,run_summary,OOD_dataset)

                row_names[data_index] = f'ID:{ID_dataset}, OOD:{OOD_dataset}' 

        
        data_array_AUROC = np.round(data_array_AUROC/count_array_AUROC,decimals=3)  
        data_array_AUPR = np.round(data_array_AUPR/count_array_AUPR,decimals=3)  
        data_array_FPR = np.round(data_array_FPR/count_array_FPR,decimals=3)
        column_names = baseline_approaches + [f'Quadratic {fixed_k} NN']        
        auroc_df = pd.DataFrame(data_array_AUROC,columns = column_names, index=row_names)
        aupr_df = pd.DataFrame(data_array_AUPR,columns = column_names, index=row_names)
        fpr_df = pd.DataFrame(data_array_FPR,columns = column_names, index=row_names)
        
        #desired_approach.split("_")
        caption = ID_dataset + ' Dataset'+ f' with {desired_approach.replace("_"," ")} {desired_model_type}'  # replace Underscore with spaces for the caption
        label = f'tab:{ID_dataset}_Dataset_{desired_approach}_{desired_model_type}'
        #caption = ID_dataset + ' Dataset'+ f' with {desired_approach.replace("_"," ")} {desired_model_type} vs {baseline_approach.replace("_"," ")} {baseline_model_type} Baseline'  # replace Underscore with spaces for the caption
        #label = f'tab:{ID_dataset}_Dataset_{desired_approach}_{desired_model_type}_{baseline_approach}_{baseline_model_type}'
        #latex_table = single_baseline_post_process_latex_table(auroc_df, caption, label,value)
        latex_table =  collated_multiple_baseline_post_process_latex_table(auroc_df,aupr_df, fpr_df,caption, label)
        #latex_table = full_post_process_latex_table(auroc_df, caption, label,value='max')
        
        #print(latex_table)
        all_latex_tables.append(latex_table)
    

    baseline_names = ''
    for index in range(len(baseline_approaches)):
        if index ==0:
            baseline_names = baseline_names + baseline_approaches[index] + '_' + baseline_model_types[index]
        else:
            baseline_names = baseline_names + '_and_'+ baseline_approaches[index] + '_' + baseline_model_types[index]

    combined_caption = f'AUROC, AUPR and FPR for different ID-OOD dataset pairs using {baseline_names.replace("_"," ")} baselines and {desired_approach.replace("_"," ")} {desired_model_type}'
    combined_label =f'tab:datasets_comparison_{desired_approach}_{desired_model_type}_{baseline_names}'
    combined_table = combine_multiple_tables(all_latex_tables,combined_caption, combined_label)
    combined_table = separate_columns(combined_table)
    combined_table = separate_top_columns(combined_table)
    combined_table = add_baseline_names_row(combined_table,baseline_approaches)
    combined_table = remove_hline_processing(combined_table)
    combined_table = update_headings_additional(combined_table)
    combined_table = update_double_col_table(combined_table)

    combined_table = separate_ID_datasets(combined_table) # Used to add a line between the ID datasets
    
    print(combined_table)
        

# Same as before but also calculates the wilcoxon values to see whether the value is higher than the threshold
def knn_table_collated_wilcoxon(desired_approach = 'Quadratic_typicality', desired_model_type = 'SupCon', baseline_approaches = ['Softmax','Mahalanobis'], baseline_model_types = ['CE','CE'],dataset_type ='grayscale',t_test='less'):
    num_repeats = 8
    baselines_dict = {'Mahalanobis':{'AUROC':'Mahalanobis AUROC OOD'.lower(),'AUPR':'Mahalanobis AUPR'.lower(),'FPR':'Mahalanobis FPR'.lower()},
                
                'Softmax':{'AUROC':'Maximum Softmax Probability AUROC OOD'.lower(),'AUPR':'Maximum Softmax Probability AUPR OOD'.lower(),'FPR':'Maximum Softmax Probability FPR OOD'.lower()},
                }

    assert len(baseline_approaches) == len(baseline_model_types), 'number of baseline approaches do not match number of baseline models'
    num_baselines = len(baseline_approaches)

    if desired_approach == 'Quadratic_typicality':
        desired_string_AUROC = 'Normalized One Dim Class Quadratic Typicality KNN - 10 OOD'.lower() # Only get the key for the AUROC
        desired_string_AUPR= 'Normalized One Dim Class Quadratic Typicality KNN - 10 AUPR'.lower()
        desired_string_FPR = 'Normalized One Dim Class Quadratic Typicality KNN - 10 FPR'.lower()
        desired_function = obtain_knn_value # Used to calculate the value
    
    ####################
    # make a loop to add the baseline strings and the baseline AUPR etc
    baseline_strings_AUROC = []
    baseline_strings_AUPR = []
    baseline_strings_FPR = []
    for approach in baseline_approaches:
        assert approach == 'Mahalanobis' or approach =='Softmax', 'No other baselines implemented'
        baseline_strings_AUROC.append(baselines_dict[approach]['AUROC'])
        baseline_strings_AUPR.append(baselines_dict[approach]['AUPR'])
        baseline_strings_FPR.append(baselines_dict[approach]['FPR'])

    baseline_function = obtain_baseline
    
    # Fixed value of k of interest
    fixed_k = 10
    # Desired ID,OOD and Model  
    root_dir = 'run_data/'

    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py
    all_latex_tables = []
    all_ID = ['MNIST','FashionMNIST','KMNIST'] if dataset_type =='grayscale' else ['CIFAR10','CIFAR100','Caltech256','TinyImageNet'] 
    #all_ID = ['MNIST','FashionMNIST','KMNIST', 'CIFAR10','CIFAR100','Caltech101','Caltech256','TinyImageNet','Cub200','Dogs']
    #all_ID = ['MNIST','FashionMNIST','KMNIST']
    for ID_dataset in all_ID: # Go through the different ID dataset                
        runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats","config.epochs": 300,"config.dataset": f"{ID_dataset}","$or": [{"config.model_type":"SupCon" }, {"config.model_type": "CE"}]})
        # number of OOd datasets for this particular ID dataset
        num_ood = len(dataset_dict[ID_dataset])
        
        # data array
        data_array_AUROC = np.zeros((num_ood,num_baselines+1)) 
        count_array_AUROC = np.zeros((num_ood,num_baselines+1))  

        data_array_AUPR = np.zeros((num_ood,num_baselines+1)) 
        count_array_AUPR = np.zeros((num_ood,num_baselines+1)) 

        data_array_FPR = np.zeros((num_ood,num_baselines+1)) 
        count_array_FPR = np.zeros((num_ood,num_baselines+1))
        

        ########### Code to calculate p-values ###################
        # Arrays to calculate the p-values
        baseline_AUROC_values, baseline_AUPR_values, baseline_FPR_values =  np.empty((num_baselines,num_ood,num_repeats)), np.empty((num_baselines,num_ood,num_repeats)), np.empty((num_baselines,num_ood,num_repeats))
        desired_AUROC_values, desired_AUPR_values, desired_FPR_values = np.empty((1,num_ood,num_repeats)), np.empty((1,num_ood,num_repeats)), np.empty((1,num_ood,num_repeats))

        baseline_AUROC_values[:], baseline_AUPR_values[:], baseline_FPR_values[:] =  np.nan, np.nan, np.nan
        desired_AUROC_values[:], desired_AUPR_values[:], desired_FPR_values[:] = np.nan, np.nan, np.nan

        # Get scores for each of the OOD datasets for this ID dataset
        collated_rank_score_auroc = np.empty((num_ood,num_baselines)) # + 1 to take into account an additional wilcoxon score which takes into account the entire dataset
        collated_rank_score_auroc[:] = np.nan
        
        collated_rank_score_aupr = np.empty((num_ood,num_baselines)) # + 1 to take into account an additional wilcoxon score which takes into account the entire dataset
        collated_rank_score_aupr[:] = np.nan

        collated_rank_score_fpr = np.empty((num_ood,num_baselines)) # + 1 to take into account an additional wilcoxon score which takes into account the entire dataset
        collated_rank_score_fpr[:] = np.nan

        ##############################################################
        for i, run in enumerate(runs): # go through the runs for a particular ID dataset
            # .summary contains the output keys/values for metrics like accuracy.
            #  We call ._json_dict to omit large files 

            run_summary = run.summary._json_dict
            # .config contains the hyperparameters.
            #  We remove special values that start with _.
            run_config = {k: v for k,v in run.config.items()
                 if not k.startswith('_')} 
            
            group_name = run_config['group']
            path_list = run.path
            # include the group name in the run path
            path_list.insert(-1, group_name)
            model_type = run_config['model_type']
            Model_name = 'SupCLR' if model_type=='SupCon' else model_type
            # Make a data array, where the number values are equal to the number of OOD classes present in the datamodule dict or equal to the number of keys

            # name for the different rows of a table
            # https://stackoverflow.com/questions/10712002/create-an-empty-list-in-python-with-certain-size
            row_names = [None] * num_ood # Make an empty list to take into account all the different values 
            # go through the different knn keys
            
            # Obtain the ID and OOD dataset
            # Make function to obtain quadratic typicality for a particular ID and OOD dataset
            # Make function to obtain mahalanobis from a particular ID and OOD dataset
            
            # Obtain all the OOD datasets for a particular desired string
            all_OOD_datasets = obtain_ood_datasets(desired_string_AUROC, run_summary,ID_dataset)
            #all_OOD_datasets = obtain_ood_datasets_baseline(desired_string_AUROC,baseline_strings_AUROC[0], run_summary,ID_dataset)
            for OOD_dataset in all_OOD_datasets:
                data_index = dataset_dict[ID_dataset][OOD_dataset] # index location
                row_names[data_index] = f'ID:{ID_dataset}, OOD:{OOD_dataset}'
                # updates the data array and the count array at a certain location with
                
                data_array_AUROC, count_array_AUROC =update_metric_and_count(data_array_AUROC,count_array_AUROC,data_index,-1,desired_function,desired_string_AUROC,desired_model_type,model_type,run_summary,OOD_dataset)
                data_array_AUPR, count_array_AUPR =update_metric_and_count(data_array_AUPR,count_array_AUPR,data_index,-1,desired_function,desired_string_AUPR,desired_model_type,model_type,run_summary,OOD_dataset)
                data_array_FPR, count_array_FPR =update_metric_and_count(data_array_FPR,count_array_FPR,data_index,-1,desired_function,desired_string_FPR,desired_model_type,model_type,run_summary,OOD_dataset)

                desired_AUROC_values = update_metric_array(desired_AUROC_values, 0,data_index,desired_function,desired_string_AUROC,desired_model_type,model_type,run_summary,OOD_dataset,run_config['seed'])
                desired_AUPR_values = update_metric_array(desired_AUPR_values, 0,data_index,desired_function,desired_string_AUPR,desired_model_type, model_type,run_summary,OOD_dataset,run_config['seed'])
                desired_FPR_values = update_metric_array(desired_FPR_values, 0,data_index,desired_function,desired_string_FPR,desired_model_type, model_type,run_summary,OOD_dataset,run_config['seed'])

                for i in range(num_baselines):
                    
                    data_array_AUROC, count_array_AUROC =update_metric_and_count(data_array_AUROC,count_array_AUROC,data_index,i,baseline_function,baseline_strings_AUROC[i],baseline_model_types[i],model_type,run_summary,OOD_dataset)
                    data_array_AUPR, count_array_AUPR =update_metric_and_count(data_array_AUPR,count_array_AUPR,data_index,i,baseline_function,baseline_strings_AUPR[i],baseline_model_types[i],model_type,run_summary,OOD_dataset)
                    data_array_FPR, count_array_FPR =update_metric_and_count(data_array_FPR,count_array_FPR,data_index,i,baseline_function,baseline_strings_FPR[i],baseline_model_types[i],model_type,run_summary,OOD_dataset)
                    
                    baseline_AUROC_values = update_metric_array(baseline_AUROC_values,i,data_index,baseline_function,baseline_strings_AUROC[i],baseline_model_types[i] ,model_type,run_summary,OOD_dataset,run_config['seed'])
                    baseline_AUPR_values = update_metric_array(baseline_AUPR_values,i,data_index,baseline_function,baseline_strings_AUPR[i],baseline_model_types[i], model_type,run_summary,OOD_dataset,run_config['seed'])
                    baseline_FPR_values = update_metric_array(baseline_FPR_values,i,data_index,baseline_function,baseline_strings_FPR[i], baseline_model_types[i], model_type,run_summary,OOD_dataset,run_config['seed'])

        ####### Calculates p-values #############################
        for i in range(num_baselines):
            difference_auroc = np.array(baseline_AUROC_values[i]) - np.array(desired_AUROC_values[0]) # shape (num ood, repeats)
            difference_aupr = np.array(baseline_AUPR_values[i]) - np.array(desired_AUPR_values[0]) # shape (num ood, repeats)

            # REVERSED THE DIRECTION FOR FPR DUE TO LOWER BEING BETTER FOR FPR, SO I DO NOT NEED TO REVERSE THE DIRECTION OF THE TEST STATISTIC
            difference_fpr = np.array(desired_FPR_values[0]) - np.array(baseline_FPR_values[i]) # shape (num ood, repeats)
            # Calculate the p values for a particular OOD dataset for this ID dataset
            for j in range(len(difference_auroc)): # go through all the different ID OOD dataset pairs
                #stat, p_value  = wilcoxon(difference[i],alternative='less') # calculate the p value for a particular ID OOD dataset pair
                stat, p_value_auroc  = wilcoxon(difference_auroc[j],alternative=t_test) # calculate the p value for a particular ID OOD dataset pair
                stat, p_value_aupr  = wilcoxon(difference_aupr[j],alternative=t_test) # calculate the p value for a particular ID OOD dataset pair
                stat, p_value_fpr  = wilcoxon(difference_fpr[j],alternative=t_test) # calculate the p value for a particular ID OOD dataset pair
                
                collated_rank_score_auroc[j,i] = p_value_auroc # add the p_value to the rank score for this particular dataset
                collated_rank_score_aupr[j,i] = p_value_aupr # add the p_value to the rank score for this particular dataset
                collated_rank_score_fpr[j,i] = p_value_fpr # add the p_value to the rank score for this particular dataset
        
        p_value_column_names = copy.deepcopy(baseline_approaches)
        
        # Post processing latex table

        auroc_insignificance_df = insignificance_dataframe(collated_rank_score_auroc,0.05,row_names)
        #print('rank auroc',collated_rank_score_auroc)
        aupr_insignificance_df = insignificance_dataframe(collated_rank_score_aupr,0.05,row_names)
        fpr_insignificance_df = insignificance_dataframe(collated_rank_score_fpr,0.05,row_names)
        
        ##########################################################
        data_array_AUROC = np.round(data_array_AUROC/count_array_AUROC,decimals=3)  
        data_array_AUPR = np.round(data_array_AUPR/count_array_AUPR,decimals=3)  
        data_array_FPR = np.round(data_array_FPR/count_array_FPR,decimals=3)
        column_names = baseline_approaches + [f'Quadratic {fixed_k} NN']        
        auroc_df = pd.DataFrame(data_array_AUROC,columns = column_names, index=row_names)
        aupr_df = pd.DataFrame(data_array_AUPR,columns = column_names, index=row_names)
        fpr_df = pd.DataFrame(data_array_FPR,columns = column_names, index=row_names)
        
        #desired_approach.split("_")
        caption = ID_dataset + ' Dataset'+ f' with {desired_approach.replace("_"," ")} {desired_model_type}'  # replace Underscore with spaces for the caption
        label = f'tab:{ID_dataset}_Dataset_{desired_approach}_{desired_model_type}'
        #caption = ID_dataset + ' Dataset'+ f' with {desired_approach.replace("_"," ")} {desired_model_type} vs {baseline_approach.replace("_"," ")} {baseline_model_type} Baseline'  # replace Underscore with spaces for the caption
        #label = f'tab:{ID_dataset}_Dataset_{desired_approach}_{desired_model_type}_{baseline_approach}_{baseline_model_type}'
        #latex_table = single_baseline_post_process_latex_table(auroc_df, caption, label,value)
        latex_table = collated_multiple_baseline_post_process_latex_table_insignificance(auroc_df,aupr_df, fpr_df,auroc_insignificance_df,aupr_insignificance_df, fpr_insignificance_df,t_test,caption, label)
        #latex_table =  collated_multiple_baseline_post_process_latex_table(auroc_df,aupr_df, fpr_df,caption, label)
        #latex_table = full_post_process_latex_table(auroc_df, caption, label,value='max')
        
        all_latex_tables.append(latex_table)
    
    baseline_names = ''
    for index in range(len(baseline_approaches)):
        if index ==0:
            baseline_names = baseline_names + baseline_approaches[index] + '_' + baseline_model_types[index]
        else:
            baseline_names = baseline_names + '_and_'+ baseline_approaches[index] + '_' + baseline_model_types[index]

    combined_caption = f'AUROC, AUPR and FPR for different ID-OOD dataset pairs using {baseline_names.replace("_"," ")} baselines and {desired_approach.replace("_"," ")} {desired_model_type}'
    combined_label =f'tab:datasets_comparison_{desired_approach}_{desired_model_type}_{baseline_names}'
    combined_table = combine_multiple_tables(all_latex_tables,combined_caption, combined_label)
    combined_table = separate_columns(combined_table)
    combined_table = separate_top_columns(combined_table)
    combined_table = add_baseline_names_row(combined_table,baseline_approaches)
    combined_table = remove_hline_processing(combined_table)
    combined_table = update_headings_additional(combined_table)
    combined_table = update_double_col_table(combined_table)
    combined_table = separate_ID_datasets(combined_table) # Used to add a line between the ID datasets
    
    print(combined_table)
    
# Used to create a dataframe
def create_dataframe(data_array,column_names,OOD_names,ID_names):
    data_dict = {'OOD datasets':OOD_names}
    data = {column_names[i]:data_array[:,i] for i in range(len(column_names))}
    data_dict.update(data)
    df = pd.DataFrame(data= data_dict,index=ID_names)
    return df


# Checks if there is an insignificant difference (if false, the results are statistically significant)
def insignificance_dataframe(data_array,significance_value,index_names):
    '''
    inputs:
    data_array: array which shows the the p-value scores
    significance_value:  the threshold where values below have statisitcally significant results and values above are not significant
    index_names: name of index to make the data frame 

    output:
    insignificance_df: data frame with true and false values to see if the approach is not statistically better, false means results are statistically better
    '''
    
    data_array = data_array - significance_value
    data_array = np.where(data_array>0,data_array, 0) #  any value which is higher than zero stays as zero,whilst values below become zero
    array = np.any(data_array, axis=1) # check all the columns I believe, checks whether any of the columns are true, if true along any column, the approach is not statistically significant to all the baselines, aim to have all columns as false
    insignificance_df = pd.DataFrame(array, index = index_names)
    return insignificance_df


def update_metric(metric_array,data_index, second_index,metric_function, metric_string, metric_model_type,run_model_type, summary, OOD_dataset):
    if metric_model_type == run_model_type:
        metric_value = metric_function(metric_string,summary,OOD_dataset) # calculates the value
        metric_array[data_index,second_index] += metric_value
        return metric_array
    else:
        return metric_array # metric array with no changes
# Have both so that the count array only updates when the metric array updata
def update_metric_and_count(metric_array,count_array,data_index, second_index,metric_function, metric_string, metric_model_type,run_model_type, summary, OOD_dataset):
    #print('metric model type:',metric_model_type)
    if metric_model_type == run_model_type:
        metric_value = metric_function(metric_string,summary,OOD_dataset) # calculates the value
        metric_array[data_index,second_index] += metric_value
        count_array[data_index,second_index] +=1
        return metric_array, count_array
    else:
        return metric_array, count_array # metric array with no changes

# Obtain the OOD_datasets which are in common between the desired string and the baseline
def obtain_ood_datasets_baseline(desired_string,baseline_string,summary,ID_dataset):
    
    keys = [key for key, value in summary.items() if desired_string in key.lower()]
    all_OOD_datasets = []
    # obtain all the OOD datasets which are not known
    for key in keys:
        OOD_dataset = ood_dataset_string(key, dataset_dict, ID_dataset)
        if OOD_dataset is None or OOD_dataset == ID_dataset:
            pass
        else: 
            all_OOD_datasets.append(OOD_dataset)
    # go through the baseline string case
    if len(all_OOD_datasets) ==0:
        keys = [key for key, value in summary.items() if baseline_string in key.lower()]
        for key in keys:
            OOD_dataset = ood_dataset_string(key, dataset_dict, ID_dataset)
            if OOD_dataset is None or OOD_dataset == ID_dataset:
                pass
            else: 
                all_OOD_datasets.append(OOD_dataset)

    return all_OOD_datasets        


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
    knn_output = round(summary[ood_dataset_specific_key[0]],3)
    return knn_output

# General function to obtain the baseline value
def obtain_baseline(desired_string, summary,OOD_dataset):
    #print('OOD dataset',OOD_dataset)
    desired_string = desired_string.lower() # double check that it has been lowered 
    keys = [key for key, value in summary.items() if desired_string in key.lower()]
    # get the specific mahalanobis keys for the specific OOD dataset
    OOD_dataset_specific_key = [key for key in keys if OOD_dataset.lower() in str.split(key.lower())]
    
    baseline_output = round(summary[OOD_dataset_specific_key[0]],3)
    
    return baseline_output

# obtain the value for the mahalanobis for a particular situation
def obtain_baseline_mahalanobis(summary,OOD_dataset,string1 = 'Mahalanobis AUROC OOD', string2='Mahalanobis AUROC: instance vector'):

    baseline_string_1, baseline_string_2 = string1.lower(), string2.lower() 
    #baseline_string_1, baseline_string_2 = 'Mahalanobis AUROC OOD'.lower(), 'Mahalanobis AUROC: instance vector'.lower() 
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




# small hack to make it work for the seeds of interest
def update_metric_list(metric_list,data_index,metric_function, metric_string, metric_model_type,run_model_type, summary, OOD_dataset,seed):
    #print('metric model type:',metric_model_type)
    seeds = [25,50,75,100,125,150,175]
    if metric_model_type == run_model_type and seed in seeds:
        metric_value = metric_function(metric_string,summary,OOD_dataset) # calculates the value
        metric_list[data_index].append(metric_value)
        return metric_list
    else:
        return metric_list # metric array with no changes


# Used to calculate the metric when using a numpy array instead of a list
def update_metric_array(metric_array,baseline_index,data_index,metric_function, metric_string, metric_model_type,run_model_type, summary, OOD_dataset,run_seed):
    #print('metric model type:',metric_model_type)
    seeds = [25,50,75,100,125,150,175,200]

    if metric_model_type == run_model_type and run_seed in seeds:

        metric_value = metric_function(metric_string,summary,OOD_dataset) # calculates the value
        repeat_index = seeds.index(run_seed)
        
        metric_array[baseline_index,data_index,repeat_index] = metric_value
        return metric_array
    else:
        return metric_array# metric array with no changes

if __name__== '__main__':
    #knn_auroc_table()
    #knn_auroc_table_mean()
    #knn_auroc_table_collated()
    #knn_table_collated(baseline_approach='Mahalanobis',baseline_model_type='SupCon')
    #knn_table_collated(baseline_approach='Mahalanobis',baseline_model_type='CE')
    #knn_table_collated(desired_approach = 'Quadratic_typicality', desired_model_type = 'CE', baseline_approach = 'Mahalanobis', baseline_model_type = 'CE')
    #knn_table_collated(desired_approach = 'Quadratic_typicality', desired_model_type = 'SupCon', baseline_approach = 'Mahalanobis', baseline_model_type = 'CE')
    #knn_table_collated(desired_approach = 'Quadratic_typicality', desired_model_type = 'SupCon', baseline_approach = 'Softmax', baseline_model_type = 'CE')
    #knn_table_collated(desired_approach = 'Quadratic_typicality', desired_model_type = 'SupCon', baseline_approach = 'Mahalanobis', baseline_model_type = 'SupCon')
    #knn_table_collated_v2(desired_approach = 'Quadratic_typicality', desired_model_type = 'SupCon', baseline_approaches = ['Softmax','Mahalanobis'], baseline_model_types = ['CE','CE'],dataset_type ='RGB')
    #knn_table_collated_v2(desired_approach = 'Quadratic_typicality', desired_model_type = 'CE', baseline_approaches = ['Mahalanobis'], baseline_model_types = ['CE'],dataset_type ='RGB')
    #knn_table_collated_v2(desired_approach = 'Quadratic_typicality', desired_model_type = 'SupCon', baseline_approaches = ['Mahalanobis'], baseline_model_types = ['SupCon'],dataset_type ='RGB')
    #knn_table_collated_v2(desired_approach = 'Quadratic_typicality', desired_model_type = 'SupCon', baseline_approaches = ['Mahalanobis'], baseline_model_types = ['SupCon'],dataset_type ='RGB')
    #knn_table_collated_wilcoxon(desired_approach = 'Quadratic_typicality', desired_model_type = 'SupCon', baseline_approaches = ['Softmax','Mahalanobis'], baseline_model_types = ['CE','CE'],dataset_type ='RGB',t_test='less')
    #knn_table_collated_wilcoxon(desired_approach = 'Quadratic_typicality', desired_model_type = 'CE', baseline_approaches = ['Mahalanobis'], baseline_model_types = ['CE'],dataset_type ='RGB',t_test='two-sided')
    knn_table_collated_wilcoxon(desired_approach = 'Quadratic_typicality', desired_model_type = 'SupCon', baseline_approaches = ['Mahalanobis'], baseline_model_types = ['SupCon'],dataset_type ='RGB',t_test='two-sided')


