from bdb import set_trace
from dataclasses import replace
from distutils.log import error
from importlib.resources import path
from socket import IP_DEFAULT_MULTICAST_LOOP, IP_DROP_MEMBERSHIP
from tempfile import TemporaryFile
from tkinter import E
from turtle import update
from types import LambdaType
from numpy.core.defchararray import join, split
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import copy
# Import general params
import json
import re
from decimal import *

from Contrastive_uncertainty.general.datamodules.datamodule_dict import OOD_dict

# global variable to be used to control the rounding of the data
rounding_value = 3
# For each ID dataset, it maps the dict to another value
# FULL dataset dict
'''
dataset_dict = {'MNIST':['FashionMNIST','KMNIST','EMNIST'],
            'FashionMNIST':['MNIST','KMNIST','EMNIST'],
            'KMNIST':['MNIST','FashionMNIST','EMNIST'],
            'EMNIST':['MNIST','FashionMNIST','KMNIST'],
            
            'CIFAR10':['STL10', 'CelebA','WIDERFace','SVHN','Caltech101', 'Caltech256','CIFAR100', 'VOC', 'Places365','TinyImageNet','Cub200','Dogs', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
            'CIFAR100':['STL10', 'CelebA','WIDERFace','SVHN','Caltech101', 'Caltech256','CIFAR10', 'VOC', 'Places365','TinyImageNet','Cub200','Dogs', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
            'Caltech101':['STL10', 'CelebA','WIDERFace','SVHN', 'Caltech256','CIFAR10','CIFAR100', 'VOC', 'Places365','TinyImageNet','Cub200','Dogs', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
            'Caltech256':['STL10', 'CelebA','WIDERFace','SVHN','Caltech101','CIFAR10','CIFAR100', 'VOC', 'Places365','TinyImageNet','Cub200','Dogs', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
            'TinyImageNet':['STL10', 'CelebA','WIDERFace','SVHN','Caltech101', 'Caltech256','CIFAR10','CIFAR100', 'VOC', 'Places365','Cub200','Dogs', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
            'Cub200':['STL10', 'CelebA','WIDERFace','SVHN','Caltech101', 'Caltech256','CIFAR10','CIFAR100', 'VOC', 'Places365','TinyImageNet','Dogs', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
            'Dogs':['STL10', 'CelebA','WIDERFace','SVHN','Caltech101', 'Caltech256','CIFAR10','CIFAR100', 'VOC', 'Places365','TinyImageNet','Cub200', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
}
'''
# Condensed dataset dict
dataset_dict = {'MNIST':['FashionMNIST','KMNIST'],
            'FashionMNIST':['MNIST','KMNIST'],
            'KMNIST':['MNIST','FashionMNIST'],
            'CIFAR10':['SVHN', 'CIFAR100','TinyImageNet','MNIST','FashionMNIST','KMNIST'],
            'CIFAR100':['SVHN', 'CIFAR10','TinyImageNet','MNIST','FashionMNIST','KMNIST'],
            'TinyImageNet':['SVHN', 'CIFAR10','CIFAR100','Caltech256','MNIST','FashionMNIST','KMNIST'],
            'Caltech256':['SVHN','CIFAR10','CIFAR100','TinyImageNet','MNIST','FashionMNIST','KMNIST'],
            


            #'CIFAR10':['SVHN', 'CIFAR100','Caltech256','TinyImageNet','MNIST','FashionMNIST','KMNIST'],
            #'CIFAR100':['SVHN', 'CIFAR10','Caltech256','TinyImageNet','MNIST','FashionMNIST','KMNIST'],
            #'TinyImageNet':['SVHN', 'CIFAR10','CIFAR100','Caltech256','MNIST','FashionMNIST','KMNIST'],
            #'Caltech256':['SVHN','CIFAR10','CIFAR100','TinyImageNet','MNIST','FashionMNIST','KMNIST'],
            
            
            # 'CIFAR10':['STL10','SVHN', 'CIFAR100','Caltech256','TinyImageNet'],
            # 'CIFAR100':['STL10','SVHN', 'CIFAR10','Caltech256','TinyImageNet'],
            # 'TinyImageNet':['STL10', 'SVHN', 'CIFAR10','CIFAR100','Caltech256'],
            # 'Caltech256':['STL10','SVHN','CIFAR10','CIFAR100','TinyImageNet'],
            'Caltech101':['STL10', 'CelebA','WIDERFace','SVHN', 'CIFAR10','CIFAR100', 'VOC', 'Places365', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
            #'Caltech256':['STL10', 'CelebA','WIDERFace','SVHN','Caltech101', 'CIFAR10','CIFAR100', 'VOC', 'Places365', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
            'Cub200':['STL10', 'CelebA','WIDERFace','SVHN','Caltech101', 'Caltech256','CIFAR10','CIFAR100', 'VOC', 'Places365','TinyImageNet','Dogs', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
            'Dogs':['STL10', 'CelebA','WIDERFace','SVHN','Caltech101', 'Caltech256','CIFAR10','CIFAR100', 'VOC', 'Places365','TinyImageNet','Cub200', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
}


# Converts dataset dict to have same shape as before
# https://careerkarma.com/blog/python-convert-list-to-dictionary/#:~:text=To%20convert%20a%20list%20to%20a%20dictionary%20using%20the%20same,the%20values%20of%20a%20list.
# Makes a list of numbers
for key in dataset_dict.keys():
    ood_datasets = dataset_dict[key]
    # https://stackoverflow.com/questions/18265935/python-create-list-with-numbers-between-2-values?rq=1
    indices = np.arange(len(ood_datasets)).tolist() # Create a list of numbers with from 0 to n based on the number of data points present
    dataset_dict[key] = dict(zip(ood_datasets,indices)) # make a dictionary from combining two lists together
    
            
# Dict for the specific case to the other value
key_dict = {'model_type':{'CE':0, 'Moco':1, 'SupCon':2},
            'dataset': {'MNIST':0, 'FashionMNIST':1, 'KMNIST':2, 'CIFAR10':3,'CIFAR100':4}}

# Check if ood_dataset substring is present in string
def ood_dataset_string(key, dataset_dict, ID_dataset):
    split_keys = key.lower().split() # Make the key lower and then split the string at locations where is a space
    #print(key)    
    OOD_dict = dataset_dict[ID_dataset]
    # Iterates for dictionaries as well as lists 
    iterable = OOD_dict.keys() if isinstance(OOD_dict, dict) else OOD_dict
        
    for key in iterable: #OOD_dict.keys():
        if ID_dataset.lower() in split_keys:
            ood_dataset = None
            return ood_dataset 
        elif key.lower() in split_keys:
            ood_dataset = key
            return ood_dataset

    #print(f'{key} does not have OOD dataset')
    return None 



# updated version which saves the files in more specific folders
def generic_saving(desired_key,run_filter):
    desired_key = desired_key.lower()

    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py


    # Second part used of the filter used for the purpose of an or statement to calculate the different values (Shown in the github link above )
    runs = api.runs(path="nerdk312/evaluation", filters=run_filter)
    summary_list, config_list, name_list = [], [], []

    # Change the root directory to save the file in the total KL divergence section
    root_dir = 'run_data/'
    for i, run in enumerate(runs): 
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
             if not k.startswith('_')})

        ID_dataset = config_list[i]['dataset']
        model_type = config_list[i]['model_type']
        group_name = config_list[i]['group']
        seed_value = str(config_list[i]['seed'])
        summary_list.append(run.summary._json_dict)
        
        #updated path which includes the group of the dataset
        path_list = runs[i].path
        path_list.insert(-1, group_name) # insert the group name in the location one before the last value (rather than the last value which is peculiar)
        # Additional lines added compared to the previous file
        path_list.insert(-1,model_type)
        path_list.insert(-1,ID_dataset)
        path_list.insert(-1,seed_value)
        ###################################################
        run_path = '/'.join(path_list)
        
        #run_path = '/'.join(runs[i].path)
        # .name is the human-readable name of the run.dir
        name_list.append(run.name)
        keys = [key for key, value in summary_list[i].items() if desired_key in key.lower()]
        keys = [key for key in keys if 'table' not in key.lower()]
        print(keys)
        for key in keys:
            data_dir = summary_list[i][key]['path'] 
            run_dir = root_dir + run_path
            

            total_dir = os.path.join(run_dir, data_dir)
            
            # Checks if the file is already present
            if os.path.isfile(total_dir):
                pass
            else:
                file_data = json.load(run.file(data_dir).download(root=run_dir))
            


# Utils for the latex table
def full_post_process_latex_table(df,caption,label,value = 'max'):
    
    latex_table = df.to_latex()
    latex_table = replace_headings(df,latex_table)
    
    if value == 'max':
        latex_table = bold_max_value(df,latex_table)
    elif value == 'min':
        latex_table = bold_min_value(df,latex_table)
    else: 
        assert value =='max' or value =='min', 'Incorrect value'

   
    latex_table = post_process_latex_table(latex_table)
    latex_table = initial_table_info(latex_table)
    latex_table = add_caption(latex_table,caption)
    latex_table = add_label(latex_table,label) 
    latex_table = end_table_info(latex_table)

    #latex_table = '\\begin{table}[]\n\\centering\n' + latex_table + '\n\\caption{'+ caption + '}\n\\label{' + label + '}\n\\end{table}'
    
    return latex_table

# Utils for the latex table - compares between a single baseline and a single metric eg AUROC
def single_baseline_post_process_latex_table(df,caption,label,value = 'max'):
    
    latex_table = df.to_latex()
    latex_table = replace_headings(df,latex_table)
    
    if value == 'max':
        latex_table = bold_max_value(df,latex_table)
    elif value == 'min':
        latex_table = bold_min_value(df,latex_table)
    else: 
        assert value =='max' or value =='min', 'Incorrect value'

    
    # used to get the pattern of &, then empy space, then any character, empty space,  then & then empty space
    latex_table = join_columns(latex_table)
    
    latex_table = post_process_latex_table(latex_table)
    latex_table = initial_table_info(latex_table)
    latex_table = add_caption(latex_table,caption)
    latex_table = add_label(latex_table,label) 
    latex_table = end_table_info(latex_table)

    #latex_table = '\\begin{table}[]\n\\centering\n' + latex_table + '\n\\caption{'+ caption + '}\n\\label{' + label + '}\n\\end{table}'
    
    return latex_table



# Utils for the latex table - Compares between a single baseline and several metrics AUROC, AUPR and FPR
def collated_baseline_post_process_latex_table(df_auroc, df_aupr, df_fpr,caption,label):
    
    latex_table_auroc = df_auroc.to_latex()
    latex_table_auroc = replace_headings(df_auroc,latex_table_auroc)
    latex_table_auroc = bold_max_value(df_auroc,latex_table_auroc)

    latex_table_aupr = df_aupr.to_latex()
    latex_table_aupr = replace_headings(df_aupr,latex_table_aupr)
    latex_table_aupr = bold_max_value(df_aupr,latex_table_aupr)

    latex_table_fpr = df_fpr.to_latex()
    latex_table_fpr = replace_headings(df_fpr,latex_table_fpr)
    latex_table_fpr = bold_min_value(df_fpr,latex_table_fpr)

    
    # used to get the pattern of &, then empy space, then any character, empty space,  then & then empty space
    latex_table_auroc = join_columns(latex_table_auroc,'AUROC')
    latex_table_aupr = join_columns(latex_table_aupr,'AUPR')
    latex_table_fpr = join_columns(latex_table_fpr,'FPR')

    latex_table = join_different_columns(latex_table_auroc,latex_table_aupr) # joins the auroc and aupr table together
    #
    latex_table = join_different_columns(latex_table, latex_table_fpr) # joins the auroc+aupr table with the fpr table
    latex_table = replace_headings_collated_table(latex_table) # replaces the heading to take into account the collated readings
    latex_table = post_process_latex_table(latex_table)
    latex_table = initial_table_info(latex_table)
    latex_table = add_caption(latex_table,caption)
    latex_table = add_label(latex_table,label) 
    latex_table = end_table_info(latex_table)

    #latex_table = '\\begin{table}[]\n\\centering\n' + latex_table + '\n\\caption{'+ caption + '}\n\\label{' + label + '}\n\\end{table}'
    
    return latex_table



# join columns within a single table
def join_columns(latex_table,metric):
    desired_key = "&\s+.+\s+&\s+.+\s+"
    #desired_key = "&\s+.+\s+" 
    string = re.findall(desired_key,latex_table)
    updated_string = []
    # NEED TO UPDATE CODE WITH THE RECURSIVE APPROACH TO PREVENT OVERLAPPING VALUES
    for index in range(len(string)):
        if index ==0:
            updated_string.append(f'& {metric} \\\\\n')
        else:
            updated_string.append(replace_nth('&','/',string[index],2))
        latex_table = latex_table.replace(f'{string[index]}',f'{updated_string[index]}')
    

    return latex_table


# Utils for the latex table - Compares between a single baseline and several metrics AUROC, AUPR and FPR
def collated_multiple_baseline_post_process_latex_table(df_auroc, df_aupr, df_fpr,caption,label):
    
    latex_table_auroc = df_auroc.to_latex()
    latex_table_auroc = replace_headings(df_auroc,latex_table_auroc)
    latex_table_auroc = bold_max_value(df_auroc,latex_table_auroc)

    latex_table_aupr = df_aupr.to_latex()
    latex_table_aupr = replace_headings(df_aupr,latex_table_aupr)
    latex_table_aupr = bold_max_value(df_aupr,latex_table_aupr)

    latex_table_fpr = df_fpr.to_latex()
    latex_table_fpr = replace_headings(df_fpr,latex_table_fpr)
    latex_table_fpr = bold_min_value(df_fpr,latex_table_fpr)

    # used to get the pattern of &, then empy space, then any character, empty space,  then & then empty space
    latex_table_auroc = join_multiple_columns(latex_table_auroc,'AUROC')
    latex_table_aupr = join_multiple_columns(latex_table_aupr,'AUPR')
    latex_table_fpr = join_multiple_columns(latex_table_fpr,'FPR')

    latex_table = join_different_columns(latex_table_auroc,latex_table_aupr) # joins the auroc and aupr table together
    latex_table = join_different_columns(latex_table, latex_table_fpr) # joins the auroc+aupr table with the fpr table
    latex_table = replace_headings_collated_table(latex_table) # replaces the heading to take into account the collated readings
    latex_table = post_process_latex_table(latex_table)
    latex_table = initial_table_info(latex_table)
    latex_table = add_caption(latex_table,caption)
    latex_table = add_label(latex_table,label) 
    latex_table = end_table_info(latex_table)
    
    return latex_table


# places a asterix if the approaches are better than the baseline
def asterix_surpass_baseline(latex_table,condition):
    
    desired_key = r'&.+\d.+\\' # get the sequence 
    string = re.findall(desired_key,latex_table)
    initial_string = copy.deepcopy(string)
    for index in range(len(string)):
        numbers = re.findall("\d+\.\d+", string[index]) # fnd all the numbers in the substring (gets rid of the &)
        for i, number in enumerate(numbers):
            if condition == 'greater':
                if float(number)> float(numbers[0]):
                    numbers[i] = number+'*' # update the values
            elif condition =='less':
                if float(number) < float(numbers[0]):
                    numbers[i] = number+'*' # update the values

            string[index] = string[index].replace(number,numbers[i])
        latex_table = latex_table.replace(f'{initial_string[index]}',f'{string[index]}')    
    return latex_table


# Utils for the situation where I calculat the diffeent tables and check if the different approaches are better than the baseline
def collated_multiple_baseline_post_process_latex_table_baseline(df_auroc, df_aupr, df_fpr,caption,label):
    latex_table_auroc = df_auroc.to_latex()
    latex_table_auroc = replace_headings(df_auroc,latex_table_auroc)
    latex_table_auroc = bold_max_value(df_auroc,latex_table_auroc)
    latex_table_auroc = asterix_surpass_baseline(latex_table_auroc,'greater')

    latex_table_aupr = df_aupr.to_latex()
    latex_table_aupr = replace_headings(df_aupr,latex_table_aupr)
    latex_table_aupr = bold_max_value(df_aupr,latex_table_aupr)
    latex_table_aupr = asterix_surpass_baseline(latex_table_aupr,'greater')

    latex_table_fpr = df_fpr.to_latex()
    latex_table_fpr = replace_headings(df_fpr,latex_table_fpr)
    latex_table_fpr = bold_min_value(df_fpr,latex_table_fpr)
    latex_table_fpr = asterix_surpass_baseline(latex_table_fpr,'less')


    # used to get the pattern of &, then empy space, then any character, empty space,  then & then empty space
    latex_table_auroc = join_multiple_columns(latex_table_auroc,'AUROC')
    latex_table_aupr = join_multiple_columns(latex_table_aupr,'AUPR')
    latex_table_fpr = join_multiple_columns(latex_table_fpr,'FPR')

    latex_table = join_different_columns(latex_table_auroc,latex_table_aupr) # joins the auroc and aupr table together
    latex_table = join_different_columns(latex_table, latex_table_fpr) # joins the auroc+aupr table with the fpr table
    
    latex_table = replace_headings_collated_table(latex_table) # replaces the heading to take into account the collated readings
    
    latex_table = post_process_latex_table(latex_table)
    latex_table = initial_table_info(latex_table)
    latex_table = add_caption(latex_table,caption)
    latex_table = add_label(latex_table,label) 
    latex_table = end_table_info(latex_table)

    return latex_table
    

# Utils for the latex table, same as before but also takes into accountthe p values- Compares between a single baseline and several metrics AUROC, AUPR and FPR
def collated_multiple_baseline_post_process_latex_table_insignificance(df_auroc, df_aupr, df_fpr,df_auroc_insignificance, df_aupr_insignificance, df_fpr_insignificance,t_test, caption,label):
    latex_table_auroc = df_auroc.to_latex()
    latex_table_auroc = replace_headings(df_auroc,latex_table_auroc)
    latex_table_auroc = bold_max_value(df_auroc,latex_table_auroc)

    latex_table_aupr = df_aupr.to_latex()
    latex_table_aupr = replace_headings(df_aupr,latex_table_aupr)
    latex_table_aupr = bold_max_value(df_aupr,latex_table_aupr)

    latex_table_fpr = df_fpr.to_latex()
    latex_table_fpr = replace_headings(df_fpr,latex_table_fpr)
    latex_table_fpr = bold_min_value(df_fpr,latex_table_fpr)

    latex_table_auroc_insignificance, latex_table_aupr_insignificance, latex_table_fpr_insignificance = df_auroc_insignificance.to_latex(), df_aupr_insignificance.to_latex(), df_fpr_insignificance.to_latex()
    
    if t_test =='less' or t_test =='greater':
        latex_table_auroc = asterix_typicality_values(latex_table_auroc, latex_table_auroc_insignificance)
        latex_table_aupr = asterix_typicality_values(latex_table_aupr, latex_table_aupr_insignificance)
        latex_table_fpr = asterix_typicality_values(latex_table_fpr, latex_table_fpr_insignificance)
    elif t_test == 'two-sided':
        latex_table_auroc = asterix_significant_values(latex_table_auroc, latex_table_auroc_insignificance)
        latex_table_aupr = asterix_significant_values(latex_table_aupr, latex_table_aupr_insignificance)
        latex_table_fpr = asterix_significant_values(latex_table_fpr, latex_table_fpr_insignificance)    
    else:
        print('Wrong name for t-test')

    # used to get the pattern of &, then empy space, then any character, empty space,  then & then empty space
    latex_table_auroc = join_multiple_columns(latex_table_auroc,'AUROC')
    latex_table_aupr = join_multiple_columns(latex_table_aupr,'AUPR')
    latex_table_fpr = join_multiple_columns(latex_table_fpr,'FPR')

    
    # Controls whether to combine the AUROC, AUPR and FPR or to only join 2 tables
    latex_table = join_different_columns(latex_table_auroc,latex_table_aupr) # joins the auroc and aupr table together
    latex_table = join_different_columns(latex_table, latex_table_fpr) # joins the auroc+aupr table with the fpr table
    '''
    latex_table = join_different_columns(latex_table_auroc,latex_table_fpr) # joins the auroc and aupr table together    
    '''
    latex_table = replace_headings_collated_table(latex_table) # replaces the heading to take into account the collated readings
    
    latex_table = post_process_latex_table(latex_table)
    latex_table = initial_table_info(latex_table)
    latex_table = add_caption(latex_table,caption)
    latex_table = add_label(latex_table,label) 
    latex_table = end_table_info(latex_table)

    return latex_table


# pass in the measurements as well as whether the values are insignificant
def asterix_typicality_values(latex_table_values, latex_table_insignificance):
    if rounding_value == 2:
        value_strings = re.findall(r'\\textbf{\d\.\d{2}}\s+\\\\\n|\d\.\d{2}\s+\\\\\n',latex_table_values) # Checks whether it has one pattern or another , aim is to get the last value of interest
    elif rounding_value ==3:
        value_strings = re.findall(r'\\textbf{\d\.\d{3}}\s+\\\\\n|\d\.\d{3}\s+\\\\\n',latex_table_values) # Checks whether it has one pattern or another , aim is to get the last value of interest
    
    insignificance_strings = re.findall('&\s+[^ 0].+\\\\\n',latex_table_insignificance) # Need to place a space between the ^ to show that the string should not be used 
    #print('latex table values', latex_table_values)
    #print('latex_table_insignficance',latex_table_insignificance)
    updated_string = []
    concatenated_list = []
    # Recursive approach to prevent replacing values which appear multiple times
    recursive_string = copy.deepcopy(latex_table_values)
    for index in range(len(value_strings)):
        # Break string into first part and second part (without value_strings[index])
        first_string, recursive_string = recursive_string.split(value_strings[index],1)
        # growing string
        first_string = first_string + value_strings[index] #  add the value_strings[index] nacl
        
        value_only,_ = value_strings[index].split('\\\\\n',1)
        # add asterisk and then remove whitespace
        bold_string = (value_only+'*').replace(' ','') if 'False' in insignificance_strings[index] else value_only 
        # check if False is present in the insignificance string, to make it bold
        updated_string.append(bold_string)
        
        first_string = first_string.replace(value_only, updated_string[index])
        concatenated_list.append(first_string)

    # at the end of the list(add on the remaining of the recursive string)
    concatenated_list.append(recursive_string)
    latex_table = ''.join(concatenated_list)
    return latex_table
    
# pass in the measurements as well as whether the values are insignificant, used to two-sided t-test
def asterix_significant_values(latex_table_values, latex_table_insignificance):
    if rounding_value == 2:
        value_strings = re.findall(r'&\s+\\textbf{\d\.\d{2}}\s+&|&\s+\\textbf{\d\.\d{2}}\s+\\\\\n',latex_table_values) # Need to use or to prevent getting double bolds in the case of ties
    elif rounding_value ==3:
        value_strings = re.findall(r'&\s+\\textbf{\d\.\d{3}}\s+&|&\s+\\textbf{\d\.\d{3}}\s+\\\\\n',latex_table_values) # Need to use or to prevent getting double bolds in the case of ties
    
    insignificance_strings = re.findall('&\s+[^ 0].+\\\\\n',latex_table_insignificance) # Need to place a space between the ^ to show that the string should not be used 
    #print('latex table values', latex_table_values)
    #print('latex_table_insignficance',latex_table_insignificance)
    updated_string = []
    concatenated_list = []
    # Recursive approach to prevent replacing values which appear multiple times
    recursive_string = copy.deepcopy(latex_table_values)
    for index in range(len(value_strings)):
        # Break string into first part and second part (without value_strings[index])
        first_string, recursive_string = recursive_string.split(value_strings[index],1)
        # growing string
        first_string = first_string + value_strings[index] #  add the value_strings[index] nacl
        # add asterisk and then remove whitespace
        if rounding_value == 2:
            value_only = re.findall(r'\\textbf{\d\.\d{2}}',value_strings[index])[0]
        elif rounding_value ==3:
            value_only = re.findall(r'\\textbf{\d\.\d{3}}',value_strings[index])[0]
        #value_only = re.findall(r'\\textbf{\d\.\d{rounding_value}}',value_strings[index])[0]
        
        bold_string = (value_only+'*') if 'False' in insignificance_strings[index] else value_only 
        # check if False is present in the insignificance string, to make it bold
        updated_string.append(bold_string)
        
        first_string = first_string.replace(value_only, updated_string[index])
        concatenated_list.append(first_string)

    # at the end of the list(add on the remaining of the recursive string)
    concatenated_list.append(recursive_string)
    latex_table = ''.join(concatenated_list)
    return latex_table    

# Used to combine tables from different datasets together
def combine_multiple_tables(latex_tables,caption,label): # Several latex tables
    # Split the first table at tabular
    # Split the second table at both heading, then split the second component at tabular
    # Split the last table at heading and tabular
    # Add tabular at the end of the table
    # Add caption
    # Add label
    # Add table info
    pattern_string = r'\s{Datasets}.+line\s'
    desired_string = obtain_first_occurence(latex_tables[0],pattern_string)
    combined_table=''
    for i,latex_table in enumerate(latex_tables):
        if i ==0:
            latex_table =latex_table.split("\n\\end{tabular}")[0]  # Get the first section before end tabular
            combined_table = combined_table+latex_table # build the string
        else:
            latex_table = latex_table.split(desired_string)[1]
            #latex_table = latex_table.split("\n{Datasets} & AUROC & AUPR & FPR \\\\ \\hline\n")[1] # split the beginning and get second half 
            latex_table = latex_table.split("\n\\end{tabular}")[0] #  split at the end and get the first part
            combined_table = combined_table + latex_table
    
    combined_table = add_end_tabular(combined_table)
    combined_table = add_caption(combined_table,caption)
    combined_table = add_label(combined_table,label) 
    combined_table = end_table_info(combined_table)
    return combined_table

# join multiple columns within a single latex table, so makes the format x/y/z for the different baselines
def join_multiple_columns(latex_table,metric):
    
    # Used to find the number of columns present
    desired_column_key = '&.+\\\\\n'
    # Used to find the strings with the desired key (which is esentially the number of '&)
    string = re.findall(desired_column_key,latex_table)
    num_columns = string[0].count('&') # number of separate columns

    #### Change code to take into account the number of columns present
    desired_key = "&\s+.+\s+"*(num_columns)
    string = re.findall(desired_key,latex_table)

    #desired_key = "&\s+.+\s+&\s+.+\s+"# *(num_columns)
    #desired_key = "&\s+.+\s+" 
    #string = re.findall(desired_key,latex_table)
    updated_string = []
    # NEED TO UPDATE CODE WITH THE RECURSIVE APPROACH TO PREVENT OVERLAPPING VALUES
    for index in range(len(string)):
        if index ==0:
            updated_string.append(f'& {metric} \\\\\n')
        else:
            new_string = copy.deepcopy(string[index]) # make a copy of the string
            for i in range(num_columns-1):
                new_string = replace_nth('&','/',new_string,2) # replace the second & multiple times
            updated_string.append(new_string)
        latex_table = latex_table.replace(f'{string[index]}',f'{updated_string[index]}')
    
    return latex_table

# Join the columns of two different metrics (can be used recursively )
def join_different_columns(latex_table_1,latex_table_2):
    desired_key = '&.+\\\\\n' 
    string_1 = re.findall(desired_key,latex_table_1)
    string_2 = re.findall(desired_key,latex_table_2)
    
    updated_string = []

    concatenated_list = []
    # Recursive approach to prevent replacing values which appear multiple times
    recursive_string = copy.deepcopy(latex_table_1)
    for index in range(len(string_1)):
        first_string, recursive_string = recursive_string.split(string_1[index],1)
        first_string = first_string + string_1[index]
        joint_string = string_1[index] + string_2[index]
        updated_string.append(replace_nth('\\\\\n','',joint_string,1))
        first_string = first_string.replace(string_1[index], updated_string[index])
        concatenated_list.append(first_string)
    # at the end of the list(add on the remaining of the recursive string)
    concatenated_list.append(recursive_string)
    '''
    for index in range(len(string_1)):
        joint_string = string_1[index] + string_2[index]
        updated_string.append(replace_nth('\\\\\n','',joint_string,1)) # replace first occurence of joint string
        latex_table_1 = latex_table_1.replace(f'{string_1[index]}',f'{updated_string[index]}')
    '''
    latex_table = ''.join(concatenated_list)
    return latex_table

# Make two separate columns from the data, an ID dataset column and OOD dataset column
def separate_columns(latex_table):
    
    desired_key = '\w+\:\w+\,\s+\w+\:\w+'
    string = re.findall(desired_key, latex_table)

    desired_ID_key = '\w+\:\w+\,'
    ID_string = re.findall(desired_ID_key,latex_table) # Obtains all the ID datasets

    updated_string = []
    current_ID = None
    previous_ID = None

    # NEED TO UPDATE CODE WITH THE RECURSIVE APPROACH TO PREVENT OVERLAPPING VALUES
    for index in range(len(string)):
        # Replace ID:, replace the command and replace OOD:
        new_string = string[index].replace('ID:','')
        new_string = new_string.replace(',',' &')
        new_string = new_string.replace('OOD:','')
        
        # Replace the ID data with a empty string if it is the same in different rows
        current_ID = ID_string[index]
        current_ID = current_ID.replace('ID:','')
        current_ID = current_ID.replace(',','')
        
        updated_ID = '' if current_ID == previous_ID else current_ID
        previous_ID = current_ID
        # Need to remove first entry only to prevent names from being removed
        new_string = replace_nth(f'{current_ID}',f'{updated_ID}',new_string,1) # what to replace, what to replace with, string and nth entry
        #new_string = new_string.replace(current_ID,updated_ID)
        
        updated_string.append(new_string)
        
        latex_table = latex_table.replace(f'{string[index]}',f'{updated_string[index]}')
    return latex_table

# used to separate the top columns
def separate_top_columns(latex_table):
    original_string = '{Datasets}'
    new_string = 'ID & OOD'
    latex_table = latex_table.replace(original_string,new_string) 
    return latex_table

# row for the baseline names
def add_baseline_names_row(latex_table,baselines):
    '''
    # calculate the number of columns
    desired_column_key = '&.+\\\\\'
    # Used to find the strings with the desired key (which is esentially the number of '&)
    string = re.findall(desired_column_key,latex_table)
    num_columns = string[0].count('&') # number of separate columns
    '''
    baseline_names = ''
    for i in range(len(baselines)):
        baseline_names = baseline_names + baselines[i] + '/'
    pattern_string = r'ID.+line\s'
    split_string = obtain_first_occurence(latex_table,pattern_string) # More generic way of obtaining the split string
    #split_string = "ID & OOD & AUROC & AUPR & FPR \\\\ \\hline\n"

    num_metrics = obtain_num_metrics(latex_table)
    
    latex_table_splits = latex_table.split(split_string)
    #additional_string = r'&   & \multicolumn{3}{c}' + '{' + f'{baseline_names}1D Typicality'+ r'} \\' + '\n\cmidrule(lr){3-5}'
    
    additional_string = r'&   & \multicolumn{' f'{num_metrics}' +'}' +r'{c}{' + f'{baseline_names}1D Typicality'+ r'} \\' + '\n\cmidrule(lr){3-' f'{2+num_metrics}' +'}'
    #additional_string = r'&   & \multicolumn{3}{c}{MSP/ Mahalanobis/ 1D Typicality} \\' + '\n\cmidrule(lr){3-5}'
    latex_table = latex_table_splits[0] + split_string + additional_string + latex_table_splits[1]        
    return latex_table
    
    # perform regex to get the first column of interest
    # append the regex column with the column of interest to get the data
# Remove each of the different hlines


# row for the baseline names
def add_model_names_row(latex_table,baseline_models, desired_model):
    '''
    # calculate the number of columns
    desired_column_key = '&.+\\\\\'
    # Used to find the strings with the desired key (which is esentially the number of '&)
    string = re.findall(desired_column_key,latex_table)
    num_columns = string[0].count('&') # number of separate columns
    '''
    baseline_names = ''
    for i in range(len(baseline_models)):
        baseline_names = baseline_names + baseline_models[i] + '/'

    pattern_string = r'ID.+line\s'
    
    split_string = obtain_first_occurence(latex_table,pattern_string) # More generic way of obtaining the split string
    #split_string = "ID & OOD & AUROC & AUPR & FPR \\\\ \\hline\n"
    latex_table_splits = latex_table.split(split_string)
    
    num_metrics = obtain_num_metrics(latex_table)

    #additional_string = r'&   & \multicolumn{3}{c}' + '{' + f'{baseline_names}{desired_model}'+ r'} \\' + '\n\cmidrule(lr){3-5}'

    additional_string = r'&   & \multicolumn{' f'{num_metrics}' +r'}{c}{' + f'{baseline_names}{desired_model}'+ r'} \\' + '\n\cmidrule(lr){2-' f'{2+num_metrics}' +'}'

    #additional_string = r'&   & \multicolumn{3}{c}{MSP/ Mahalanobis/ 1D Typicality} \\' + '\n\cmidrule(lr){3-5}'
    
    latex_table = latex_table_splits[0] + split_string + additional_string + latex_table_splits[1]        
    
    return latex_table
    
    # perform regex to get the first column of interest
    # append the regex column with the column of interest to get the data
# Remove each of the different hlines

# Make a line which can be used to separate the different metrics of the data
def line_columns(latex_table):
    num_metrics = obtain_num_metrics(latex_table)

    pattern_string = '{(l|c)+}' # used to take into accoun the situation where there are different values present 
    desired_string= obtain_first_occurence(latex_table,pattern_string)
    updated_string = copy.deepcopy(desired_string)
    for i, value in enumerate(range(num_metrics,0,-1)):
        if value == 1:
            pass
        else:
            updated_string = updated_string[:-value] + '|' + updated_string[-value:]

    latex_table = latex_table.replace(desired_string, updated_string)
    '''
    columns_only = desired_string[1:-1] # remove the brackets 
    updated_desired_string = '| '.join(columns_only[i:i + 1] for i in range(0, len(columns_only)))
    updated_desired_string = '{' + updated_desired_string + '}'
    latex_table = latex_table.replace(desired_string, updated_desired_string)
    return latex_table
    '''
    return latex_table
    
def bold_titles(latex_table):
    pattern_strings = ['Dataset','AUROC','AUPR','FPR','Softmax','ODIN','Mahalanobis','1D Typicality','1D Marginal Typicality','1D Single Typicality','Typicality All Dim', 'ID: CIFAR10','ID: CIFAR100','ID: Caltech256','ID: TinyImageNet']

    for pattern_string in pattern_strings:
        try:
            desired_string = obtain_first_occurence(latex_table,pattern_string)
            updated_desired_string = r'\textbf{' +desired_string + '}'    
            # Use replace nth
            latex_table = replace_nth(desired_string,updated_desired_string,latex_table,1)
            #latex_table = latex_table.replace(desired_string, updated_desired_string)
        except:
            error
    return latex_table
    '''
    pattern_string = '(Dataset|AUROC|AUPR|FPR)'


    desired_strings = re.findall(pattern_string,latex_table)
    import ipdb; ipdb; ipdb.set_trace()
    for desired_string in desired_strings:
        updated_desired_string = r'\textbf{' +desired_string + '}'
        latex_table = latex_table.replace(desired_string, updated_desired_string)

    return latex_table
    '''

def remove_hline_processing(latex_table):
    split_string = '\hline'
    latex_table_splits = latex_table.split(split_string)
    recursive_string = '' # initialise empty string
    for index in range(len(latex_table_splits)):
        if index == 0:
            recursive_string = recursive_string +latex_table_splits[index] + '\n'+r'\toprule'
        elif index == 1:
            recursive_string = recursive_string +latex_table_splits[index] + '\n'+ r'\midrule'
        elif index == len(latex_table_splits)-1:
            recursive_string = recursive_string + '\n'+  r'\bottomrule' +latex_table_splits[index] 
        else:
            recursive_string = recursive_string + latex_table_splits[index]

    return recursive_string

def update_headings_additional(latex_table):
    #num_columns = len(df.columns) # original columns
    #updated_headings = '|p{3cm}|' + 'c|'*num_columns
    desired_key = r'&.+\\'
    # Used to find the strings with the desired key (which is esentially the number of '&)
    string = re.findall(desired_key,latex_table)
    columns = string[0].count('&') # number of separate columns

    heading_key = '\|.+\|' 
    original_headings = re.findall(heading_key, latex_table)[0]
    #updated_headings = 'c'*(columns+1)
    updated_headings = 'l'*(columns+1)
    latex_table = latex_table.replace(original_headings, updated_headings)
    return latex_table


def update_double_col_table(latex_table):
    latex_table = latex_table.replace('table','table*')
    return latex_table

# Code to separate the ID datasets from one another
def separate_ID_datasets(latex_table):
    desired_string = re.findall(r'\n\s+\n',latex_table)

    updated_table = latex_table.replace('\n  \n','\n  \hline ') # whenever there is a gap between the ID datasets, there is two new lines in a row
    return updated_table

# replace the headings for a table which is made from several tables
def replace_headings_collated_table(latex_table):
    #num_columns = len(df.columns) # original columns
    #updated_headings = '|p{3cm}|' + 'c|'*num_columns
    desired_key = '&.+\\\\\n'
    # Used to find the strings with the desired key (which is esentially the number of '&)
    string = re.findall(desired_key,latex_table)
    columns = string[0].count('&') # number of separate columns
    
    heading_key = '\|.+\|' # need to use \ as | is a meta character (needs to be escaped) https://www.youtube.com/watch?v=sa-TUpSx1JA
    
    original_headings = re.findall(heading_key,latex_table)[0] # gets the first element in the list which should eb the key for the heading
    
    updated_headings = '|p{3cm}|' + 'c|'*columns # obtain the updated headings from the number of columns which have been concatenated
    # alternative heading key and updated heading which takes into account the }
    #heading_key = '\|.+\|}' # need to use \ as | is a meta character (needs to be escaped) https://www.youtube.com/watch?v=sa-TUpSx1JA
    #updated_headings = '|p{3cm}|' + 'c|'*columns +'}' # obtain the updated headings from the number of columns which have been concatenated
    latex_table = latex_table.replace(original_headings,updated_headings)
    return latex_table

# removes the column from the initial part of the table

# Based on https://www.youtube.com/watch?v=K8L6KVGG-7o
def remove_column(latex_table):
    #val = re.search(['\begin{tabular}]',latex_table)
    
    #insignificance_strings = re.findall('&\s+[^ 0].+\\\\\n',latex_table_insignificance) # Need to place a space between the ^ to show that the string should not be used 
    #pattern = re.compile(r'\{.+\}')
    
    #pattern = re.compile(r'\{l.+\}')
    # obtain tabu
    tabular_string = obtain_tabular_heading(latex_table)
    num_columns, column_alignment = obtain_type_num_columns(tabular_string)
    # change number of columns for tabular
    updated_tabular_string = r'{tabular}{' + column_alignment*(num_columns-1) + '}'
    latex_table = latex_table.replace(tabular_string, updated_tabular_string)
    
    # Lower the values for cmidrule
    latex_table = lower_cmidrule(latex_table)
    latex_table = obtain_ID_midrule(latex_table)
    latex_table = obtain_ID_hline(latex_table)
    latex_table = remove_additional_ampersans(latex_table)
    latex_table = fix_svhn(latex_table) #  change SVHN to OOD SVHN
    latex_table = replace_ID_OOD_Dataset(latex_table)
    latex_table = line_columns(latex_table)
    latex_table = bold_titles(latex_table)
    return latex_table
    #for i in matches:

# Used to get the number of columns directly from the latex table
def obtain_num_columns(latex_table):
    tabular_string = obtain_tabular_heading(latex_table)
    num_columns, _ = obtain_type_num_columns(tabular_string)     
    return num_columns
# Used to calculate the number of metrics present
def obtain_num_metrics(latex_table): 
    #pattern_string = r'(AUROC|AUPR|FPR).+line'
    pattern_string = '(AUROC|AUPR|FPR).+(line|\n)' # used to take into accoun the situation where there are different values present 
    desired_string=obtain_first_occurence(latex_table,pattern_string)
    num_metrics = desired_string.count('&') +1 # Need to add one as it counts the number of & present which does not take into account t
    return num_metrics


# Used to obtain the headingsi
def obtain_tabular_heading(latex_table):
    pattern_string = r'\{tabular\}\{.+\}'
    tabular_string = obtain_first_occurence(latex_table,pattern_string)
    return tabular_string

#def obtain_num_metrics(latex_table):


def obtain_type_num_columns(tabular_string):
    '''
    Input: tabular string with {tabular}{xxxx}
    return: 
        num_columns -number of columns,
        x - what letter x corresponds to 
    '''
    # obtain the number of columns
    pattern_string = r'\{[lc].+\}'
    column_string = obtain_first_occurence(tabular_string,pattern_string)

    column_string = column_string[1:-1] # remove the first and the last elements to remove the brackets
    num_columns = len(column_string)
    column_alignment = column_string[0]
    return num_columns, column_alignment

def lower_cmidrule(latex_table):
    ''' 
    input : latex_table - initial latex table
    output: latex_table - updated latex table with the cmid values lowered
    '''
    #pattern = re.compile(r'\{tabular\}\{.+\}')
    pattern_string = r'\\cmidrule.+\{.+\}'
    desired_string = obtain_first_occurence(latex_table,pattern_string)
    # Get the number for the columns
    #subpattern_string = r'\{.+\}'
    subpattern_string = '\d.+\d'
    digit_substring = obtain_first_occurence(desired_string,subpattern_string)
    lower_value, upper_value  = int(digit_substring[0]), int(digit_substring[-1])
    
    updated_lower_value = lower_value -1 # lower the value by 1 for the lower and the upper value
    updated_upper_value = upper_value - 1

    updated_desired_string = desired_string.replace(str(lower_value),str(updated_lower_value))
    updated_desired_string = updated_desired_string.replace(str(upper_value),str(updated_upper_value))

    latex_table = latex_table.replace(desired_string, updated_desired_string)
 
    return latex_table

def obtain_ID_midrule(latex_table):
    #pattern_string =  r'cmidrule.+\s.+[&]'
    #Based on  https://stackoverflow.com/questions/7124778/how-to-match-anything-up-until-this-sequence-of-characters-in-a-regular-expres
    pattern_string =  r'cmidrule.+\s.+?&'
    desired_string = obtain_first_occurence(latex_table,pattern_string)

    # gets the digit
    subpattern_string = '\d.+\d'    
    digit_substring = obtain_first_occurence(desired_string,subpattern_string)
    lower_value, upper_value  = int(digit_substring[0]), int(digit_substring[-1])
    
    
    ID_dataset_pattern_string = r'\n.+?\s'
    ID_dataset = obtain_first_occurence(desired_string,ID_dataset_pattern_string) # Gets the ID dataset including \n in front and space at the end
    processed_ID_dataset = ID_dataset[1:-1] # removes the \n and space at the end 
    # obtain the output related to the particular area of interest
            
    #updated_string = r'\n\multicolumn{' +f'{upper_value}' +r'}{L}{ID:'+ processed_ID_dataset+ r'} \\ \n\midrule'

    # Need to use r' ' when using \ as a string literal however when it is a new line, need to use the other function of intereset
    updated_string = '\n'+r'\multicolumn{' +f'{upper_value}' +r'}{L}{ID: '+ processed_ID_dataset+ r'} \\'+ '\n'#+r'\midrule' + '\n'

    # Removing the ID dataset string
    removal_ID_dataset_pattern_string = r'\n.+' # includes the & with the Id dataset name 
    removing_ID_dataset = obtain_first_occurence(desired_string,removal_ID_dataset_pattern_string)
    updated_desired_string = desired_string.replace(removing_ID_dataset,updated_string)
    latex_table = replace_nth(desired_string, updated_desired_string,latex_table, 1)
    return latex_table

# obtain the ID dataset for the situation wehre there is hline present
def obtain_ID_hline(latex_table):
    #Based on  https://stackoverflow.com/questions/7124778/how-to-match-anything-up-until-this-sequence-of-characters-in-a-regular-expres
    
    pattern_string =  r'hline.+?&'
    desired_strings = re.findall(pattern_string,latex_table) # finds all hline 'dataset' &
    ID_pattern_string = r'(CIFAR100|CIFAR10|Caltech256|TinyImageNet)'
    num_columns = obtain_num_columns(latex_table)
    for desired_string in desired_strings:
        ID_dataset = obtain_first_occurence(desired_string,ID_pattern_string)
        updated_desired_string = 'hline'+'\n'+r'\multicolumn{' +f'{num_columns}' +r'}{L}{ID: '+ ID_dataset+ r'} \\'+ '\n'
        latex_table = replace_nth(desired_string, updated_desired_string,latex_table, 1)
    
    return latex_table

def remove_additional_ampersans(latex_table):
    
    #pattern_string =  r'\\\s+?&'
    pattern_string =  r'\\\s+?&'
    desired_strings = re.findall(pattern_string,latex_table) # finds all hline 'dataset' &
    for desired_string in desired_strings:
        updated_desired_string = desired_string[:-1] + 'OOD:' # removes the ampersan    
        latex_table = replace_nth(desired_string, updated_desired_string,latex_table, 1)
    
    # Remove any occurence of 2 ampersans in a row
    #pattern_string = r'&\s+&'
    pattern_string = '&\s+&'
    desired_string = obtain_first_occurence(latex_table,pattern_string)
    updated_desired_string = '&'
    latex_table = replace_nth(desired_string,updated_desired_string,latex_table, 1)
    

    return latex_table
    
# post hoc hack to add OOD before svhn
def fix_svhn(latex_table):
    #pattern_string =  r'\\\s+(SVHN)'
    #pattern_string =  '\n\s.+'
    #pattern_string =  '\n\s+?(SVHN)'#+(SVHN)'
    pattern_string =  r'SVHN'
    
    #pattern_string = r'n\s+SVHN'# This did not work for some reason

    pattern_string = r'\s+SVHN'
    desired_strings = re.findall(pattern_string,latex_table) # finds all hline 'dataset' &
    
    for desired_string in desired_strings:
        updated_desired_string =  '\n ' +'OOD: SVHN'
        #updated_desired_string = r'OOD: SVHN' 
        latex_table = replace_nth(desired_string, updated_desired_string,latex_table, 1) 
    return latex_table

# Replace ID & OOD with dataset
def replace_ID_OOD_Dataset(latex_table):
    pattern_string =  r'ID\s+&\s+OOD'
    desired_string = obtain_first_occurence(latex_table,pattern_string)
    updated_string = 'Dataset' 
    latex_table = replace_nth(desired_string,updated_string,latex_table,1)
    return latex_table



    
# Obtain first occurence of a pattern in a string
def obtain_first_occurence(string, pattern_string):
    pattern = re.compile(pattern_string)
    matches = pattern.finditer(string)
    value = next(matches) # get the ifrst item of the iterable
    output_string = value.group() # Gets the specific string corresponding to the object

    return output_string


# replace nth entry of sub in a text (txt) and join the different values together using replace (replace) - based from https://stackoverflow.com/questions/35091557/replace-nth-occurrence-of-substring-in-string
def replace_nth(sub,repl,txt,nth):
    arr=txt.split(sub)
    part1=sub.join(arr[:nth])
    part2=sub.join(arr[nth:])
    
    return part1+repl+part2

#https://stackoverflow.com/questions/5254445/how-to-add-a-string-in-a-certain-position
def insert_str(string, str_to_insert, index):
    return string[:index] + str_to_insert + string[index:]

# replace the headings  - makes it to datasets as well as changing the number of columns
def replace_headings(df,latex_table):
    num_columns = len(df.columns) # (need an extra column for the case of the dataset situation)
    latex_table = latex_table.replace('{}','{Datasets}')
    original_headings = 'l' +'r'*num_columns
    updated_headings = '|p{3cm}|' + 'c|'*num_columns 
    latex_table = latex_table.replace(original_headings,updated_headings)
    return latex_table

# Make the max value in a column bold
def bold_max_value(df,latex_table):
    num_columns = len(df.columns)
    desired_key = "&\s+\d+\.\d+\s+" *(num_columns)
    string = re.findall(desired_key,latex_table)
    updated_string = []
    for index in range(len(string)):
        numbers = re.findall("\d+\.\d+", string[index]) # fnd all the numbers in the substring (gets rid of the &)
        #max_number = max(numbers,key=lambda x:float(x))
        #max_number = float(max(numbers,key=lambda x:format(float(x),'.3f'))) #  Need to get the output as a float
        #
        
        max_number = float(max(numbers,key=lambda x:float(x))) #  Need to get the output as a float
        # Need to change into 3 decimal places
        if rounding_value == 2:
            max_number = format(max_number,'.2f')
        elif rounding_value == 3:
            max_number = format(max_number,'.3f')
        bold_max = r'\textbf{' + max_number + '}'
        #
        #string[index] = string[index].replace(f'{max_number}',f'\textbf{ {max_number} }') # Need to put spaces around otherwise it just shows max number
        updated_string.append(string[index].replace(f'{max_number}',bold_max)) # Need to put spaces around otherwise it just shows max number), also need to be place to make it so that \t does not act as space
        #updated_string.append(string[index].replace(f'{max_number}',fr'\textbf{ {max_number} }')) # Need to put spaces around otherwise it just shows max number), also need to be place to make it so that \t does not act as space
        latex_table = latex_table.replace(f'{string[index]}',f'{updated_string[index]}') 
    return latex_table

# Make the max value in a column bold
def bold_min_value(df,latex_table):
    num_columns = len(df.columns)
    desired_key = "&\s+\d+\.\d+\s+" *(num_columns)
    string = re.findall(desired_key,latex_table)
    updated_string = []
    for index in range(len(string)):
        numbers = re.findall("\d+\.\d+", string[index]) # fnd all the numbers in the substring (gets rid of the &)
        #max_number = max(numbers,key=lambda x:float(x))
        if rounding_value ==2:
            min_number = float(min(numbers,key=lambda x:format(float(x),'.2f'))) #  Need to get the output as a float
            min_number = format(min_number,'.2f')
        elif rounding_value ==3:
            min_number = float(min(numbers,key=lambda x:format(float(x),'.3f'))) #  Need to get the output as a float
            min_number = format(min_number,'.3f')
            
        bold_min = r'\textbf{' + min_number + '}'
        #min_number = float(min(numbers,key=lambda x:float(x))) #  Need to get the output as a float
        #string[index] = string[index].replace(f'{max_number}',f'\textbf{ {max_number} }') # Need to put spaces around otherwise it just shows max number
        updated_string.append(string[index].replace(f'{min_number}',bold_min)) # Need to put spaces around otherwise it just shows max number), also need to be place to make it so that \t does not act as space
        #updated_string.append(string[index].replace(f'{min_number}',fr'\textbf{ {min_number} }')) # Need to put spaces around otherwise it just shows max number), also need to be place to make it so that \t does not act as space
        latex_table = latex_table.replace(f'{string[index]}',f'{updated_string[index]}') 
    return latex_table

# Adds the part related to the beginning of the latex table
def initial_table_info(latex_table):
    latex_table = '\\begin{table}[h!]\n\\centering\n' + latex_table
    return latex_table

# adds the caption
def add_caption(latex_table,caption):
    latex_table = latex_table + '\n\\caption{'+ caption + '}'
    return latex_table

def add_end_tabular(latex_table):
    latex_table = latex_table + "\n\\end{tabular}"
    return latex_table

# adds the labels
def add_label(latex_table,label):
    latex_table = latex_table + '\n\\label{' + label + '}'
    return latex_table

# Adds the part related to the end of the latex table
def end_table_info(latex_table):
    latex_table = latex_table + '\n\\end{table}'
    return latex_table


# Pass in a latex table which is then postprocessed 
def post_process_latex_table(latex_table):
    
    latex_table = latex_table.replace(r"\toprule",r"\hline")
    latex_table = latex_table.replace(r"\midrule"," ")
    latex_table = latex_table.replace(r"\bottomrule"," ")
    #latex_table = latex_table.replace(r"\midrule",r"\hline")
    #latex_table = latex_table.replace(r"\bottomrule",r"\hline")
    #https://stackoverflow.com/questions/24704299/how-to-treat-t-as-a-regular-string-in-python
    latex_table = latex_table.replace(r'\\',r'\\ \hline')

    return latex_table


if __name__ =='__main__':
    #desired_key = 'Centroid Distances Average vector_table'
    #desired_key = 'KL Divergence(Total||Class)'
    #desired_key = 'Normalized One Dim Scores Class Quadratic Typicality'
    #desired_key ='Analysis Normalized One Dim Scores Class Quadratic Typicality KNN'
    desired_key = 'Different K Normalized Quadratic One Dim Class Typicality KNN'
    #desired_key = 'Different K Normalized One Dim Class Typicality KNN'
    #desired_key = 'Different K Normalized Quadratic One Dim Class Typicality KNN'
    #run_filter={"config.group":"Baselines Repeats"}
    #run_filter={"config.group":"OOD hierarchy baselines","config.model_type": "SupCon", 'state':'finished'}
    #run_filter={"config.group":"OOD hierarchy baselines","config.model_type": "SupCon","config.dataset": "CIFAR100"}
    #run_filter={"config.group":"OOD hierarchy baselines","config.model_type": "SupCon"}
    #run_filter={"config.group":"New Model Testing","config.epochs":300}
    #run_filter={"config.group":"Baselines Repeats","$or": [{"config.model_type":"Moco"}, {"config.model_type": "SupCon"}]}
    #run_filter={"config.group":"Baselines Repeats", "config.model_type": "SupCon","config.dataset": "Cub200"}
    run_filter={"config.group":"Baselines Repeats", "config.model_type": "SupCon"}
    #run_filter={"config.group":"OOD hierarchy baselines", "config.model_type": "SupCon"}
    
    #generic_saving(desired_key,run_filter)
    generic_saving(desired_key,run_filter)