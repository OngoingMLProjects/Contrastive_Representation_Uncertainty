from importlib.resources import path
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
            '''


            'CIFAR10':['SVHN', 'CIFAR100','Caltech256','TinyImageNet','MNIST','FashionMNIST','KMNIST'],
            'CIFAR100':['SVHN', 'CIFAR10','Caltech256','TinyImageNet','MNIST','FashionMNIST','KMNIST'],
            'TinyImageNet':['SVHN', 'CIFAR10','CIFAR100','Caltech256','MNIST','FashionMNIST','KMNIST'],
            'Caltech256':['SVHN','CIFAR10','CIFAR100','TinyImageNet','MNIST','FashionMNIST','KMNIST'],
            '''
            
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

    latex_table = join_different_columns(latex_table_auroc,latex_table_aupr) # joins the auroc and aupr table together
    latex_table = join_different_columns(latex_table, latex_table_fpr) # joins the auroc+aupr table with the fpr table
    latex_table = replace_headings_collated_table(latex_table) # replaces the heading to take into account the collated readings
    latex_table = post_process_latex_table(latex_table)
    latex_table = initial_table_info(latex_table)
    latex_table = add_caption(latex_table,caption)
    latex_table = add_label(latex_table,label) 
    latex_table = end_table_info(latex_table)

    
    return latex_table


# pass in the measurements as well as whether the values are insignificant
def asterix_typicality_values(latex_table_values, latex_table_insignificance):
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
        value_only = re.findall(r'\\textbf{\d\.\d{3}}',value_strings[index])[0]
        
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

    combined_table=''
    for i,latex_table in enumerate(latex_tables):
        if i ==0:
            latex_table =latex_table.split("\n\\end{tabular}")[0]  # Get the first section before end tabular
            combined_table = combined_table+latex_table # build the string
        else:
            latex_table = latex_table.split("\n{Datasets} & AUROC & AUPR & FPR \\\\ \\hline\n")[1] # split the beginning and get second half 
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

    split_string = "ID & OOD & AUROC & AUPR & FPR \\\\ \\hline\n"
    latex_table_splits = latex_table.split(split_string)
    additional_string = r'&   & \multicolumn{3}{c}' + '{' + f'{baseline_names}1D Typicality'+ r'} \\' + '\n\cmidrule(lr){3-5}'
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

    split_string = "ID & OOD & AUROC & AUPR & FPR \\\\ \\hline\n"
    latex_table_splits = latex_table.split(split_string)
    additional_string = r'&   & \multicolumn{3}{c}' + '{' + f'{baseline_names}{desired_model}'+ r'} \\' + '\n\cmidrule(lr){3-5}'
    #additional_string = r'&   & \multicolumn{3}{c}{MSP/ Mahalanobis/ 1D Typicality} \\' + '\n\cmidrule(lr){3-5}'
    latex_table = latex_table_splits[0] + split_string + additional_string + latex_table_splits[1]        
    return latex_table
    
    # perform regex to get the first column of interest
    # append the regex column with the column of interest to get the data
# Remove each of the different hlines

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
    desired_key ='Analysis Normalized One Dim Scores Class Quadratic Typicality KNN'
    #desired_key = 'Different K Normalized One Dim Class Typicality KNN'
    #desired_key = 'Different K Normalized Quadratic One Dim Class Typicality KNN'
    #run_filter={"config.group":"Baselines Repeats"}
    #run_filter={"config.group":"OOD hierarchy baselines","config.model_type": "SupCon", 'state':'finished'}
    #run_filter={"config.group":"OOD hierarchy baselines","config.model_type": "SupCon","config.dataset": "CIFAR100"}
    #run_filter={"config.group":"OOD hierarchy baselines","config.model_type": "SupCon"}
    #run_filter={"config.group":"New Model Testing","config.epochs":300}
    #run_filter={"config.group":"Baselines Repeats","$or": [{"config.model_type":"Moco"}, {"config.model_type": "SupCon"}]}
    run_filter={"config.group":"Baselines Repeats", "config.model_type": "SupCon","config.dataset": "Cub200"}
    #generic_saving(desired_key,run_filter)
    generic_saving(desired_key,run_filter)