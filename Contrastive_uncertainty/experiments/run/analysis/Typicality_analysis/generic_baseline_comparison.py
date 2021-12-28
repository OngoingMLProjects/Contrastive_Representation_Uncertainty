# Code for making tables for the ID and OOD datasets

from types import prepare_class
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
    collated_multiple_baseline_post_process_latex_table_insignificance, separate_ID_datasets, add_model_names_row

from Contrastive_uncertainty.experiments.run.analysis.Typicality_analysis.knn_one_dim_typicality_tables import obtain_baseline, insignificance_dataframe, update_metric_and_count, update_metric_array






# same as the previous function but used for a generic case of calculating the values
def general_table_collated_wilcoxon(desired_approach = 'Quadratic_typicality', desired_model_type = 'SupCon', baseline_approaches = ['Softmax','Mahalanobis'], baseline_model_types = ['CE','CE'],dataset_type ='grayscale',t_test='less'):
    num_repeats = 8

    dataset_dict = {'MNIST':['FashionMNIST','KMNIST'],
            'FashionMNIST':['MNIST','KMNIST'],
            'KMNIST':['MNIST','FashionMNIST'],
            
            'CIFAR10':['SVHN', 'CIFAR100','MNIST','FashionMNIST','KMNIST'],
            'CIFAR100':['SVHN', 'CIFAR10','MNIST','FashionMNIST','KMNIST'],
            'TinyImageNet':['SVHN', 'CIFAR10','CIFAR100','Caltech256','MNIST','FashionMNIST','KMNIST'],
            'Caltech256':['SVHN','CIFAR10','CIFAR100','TinyImageNet','MNIST','FashionMNIST','KMNIST'],

            'Caltech101':['STL10', 'CelebA','WIDERFace','SVHN', 'CIFAR10','CIFAR100', 'VOC', 'Places365', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
            #'Caltech256':['STL10', 'CelebA','WIDERFace','SVHN','Caltech101', 'CIFAR10','CIFAR100', 'VOC', 'Places365', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
            'Cub200':['STL10', 'CelebA','WIDERFace','SVHN','Caltech101', 'Caltech256','CIFAR10','CIFAR100', 'VOC', 'Places365','TinyImageNet','Dogs', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
            'Dogs':['STL10', 'CelebA','WIDERFace','SVHN','Caltech101', 'Caltech256','CIFAR10','CIFAR100', 'VOC', 'Places365','TinyImageNet','Cub200', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
}

    dataset_dict = process_dataset_dict(dataset_dict)

    baselines_dict = {'Mahalanobis':{'AUROC':'Mahalanobis AUROC OOD'.lower(),'AUPR':'Mahalanobis AUPR'.lower(),'FPR':'Mahalanobis FPR'.lower()},
                
                'Softmax':{'AUROC':'Maximum Softmax Probability AUROC OOD'.lower(),'AUPR':'Maximum Softmax Probability AUPR OOD'.lower(),'FPR':'Maximum Softmax Probability FPR OOD'.lower()},

                'ODIN':{'AUROC':'ODIN AUROC OOD'.lower(),'AUPR':'ODIN AUPR OOD'.lower(),'FPR':'ODIN FPR OOD'.lower()},

                'KDE':{'AUROC':'KDE AUROC OOD'.lower(),'AUPR':'KDE AUPR OOD'.lower(),'FPR':'KDE FPR OOD'.lower()},
                
                }

    assert len(baseline_approaches) == len(baseline_model_types), 'number of baseline approaches do not match number of baseline models'
    num_baselines = len(baseline_approaches)
    
    desired_string_AUROC = baselines_dict[desired_approach]['AUROC'] # Only get the key for the AUROC
    desired_string_AUPR= baselines_dict[desired_approach]['AUPR']
    desired_string_FPR = baselines_dict[desired_approach]['FPR']
    desired_function = obtain_baseline # Used to calculate the value
    
    ####################
    # make a loop to add the baseline strings and the baseline AUPR etc
    baseline_strings_AUROC = []
    baseline_strings_AUPR = []
    baseline_strings_FPR = []
    for approach in baseline_approaches:
        assert approach == 'Mahalanobis' or approach =='Softmax' or approach =='ODIN' or approach =='KDE', 'No other baselines implemented'
        baseline_strings_AUROC.append(baselines_dict[approach]['AUROC'])
        baseline_strings_AUPR.append(baselines_dict[approach]['AUPR'])
        baseline_strings_FPR.append(baselines_dict[approach]['FPR'])

    baseline_function = obtain_baseline


    # Desired ID,OOD and Model  
    root_dir = 'run_data/'

    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py
    all_latex_tables = []
    all_ID = ['MNIST','FashionMNIST','KMNIST'] if dataset_type =='grayscale' else ['CIFAR10','CIFAR100','MNIST','FashionMNIST','KMNIST'] 
    #all_ID = ['MNIST','FashionMNIST','KMNIST', 'CIFAR10','CIFAR100','Caltech101','Caltech256','TinyImageNet','Cub200','Dogs']
    #all_ID = ['MNIST','FashionMNIST','KMNIST']
    for ID_dataset in all_ID: # Go through the different ID dataset                
        runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats","config.epochs": 300,"config.dataset": f"{ID_dataset}"})
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
            all_OOD_datasets = obtain_ood_datasets_generic(desired_string_AUROC, run_summary,ID_dataset,dataset_dict)
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
        #import ipdb; ipdb.set_trace()
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
        column_names = baseline_model_types + [desired_model_type]        
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
    combined_table = add_model_names_row(combined_table,baseline_model_types,desired_model_type)
    combined_table = remove_hline_processing(combined_table)
    combined_table = update_headings_additional(combined_table)
    combined_table = update_double_col_table(combined_table)
    combined_table = separate_ID_datasets(combined_table) # Used to add a line between the ID datasets
    
    print(combined_table)




# Obtain the OOD datasets for a particular string
def obtain_ood_datasets_generic(desired_string,summary,ID_dataset, dataset_dict):
    
    
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

def process_dataset_dict(dataset_dict):
    # Converts dataset dict to have same shape as before
    # https://careerkarma.com/blog/python-convert-list-to-dictionary/#:~:text=To%20convert%20a%20list%20to%20a%20dictionary%20using%20the%20same,the%20values%20of%20a%20list.
    # Makes a list of numbers
    for key in dataset_dict.keys():
        ood_datasets = dataset_dict[key]
        # https://stackoverflow.com/questions/18265935/python-create-list-with-numbers-between-2-values?rq=1
        indices = np.arange(len(ood_datasets)).tolist() # Create a list of numbers with from 0 to n based on the number of data points present
        dataset_dict[key] = dict(zip(ood_datasets,indices)) # make a dictionary from combining two lists together
    
    return dataset_dict



if __name__== '__main__':
    general_table_collated_wilcoxon(desired_approach = 'KDE', desired_model_type = 'CE', baseline_approaches = ['KDE'], baseline_model_types = ['Moco'],dataset_type ='rgb',t_test='two-sided')

