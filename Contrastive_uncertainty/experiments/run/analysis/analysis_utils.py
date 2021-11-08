import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import general params
import json
import re

from Contrastive_uncertainty.general.datamodules.datamodule_dict import OOD_dict
# For each ID dataset, it maps the dict to another value

'''
dataset_dict = {'MNIST': {'FashionMNIST':0, 'KMNIST':1,'EMNIST':2},
            'FashionMNIST': {'MNIST':0, 'KMNIST':1,'EMNIST':2},
            'KMNIST': {'MNIST':0, 'FashionMNIST':1,'EMNIST':2},
            'CIFAR10': {'STL10':0,'Caltech101':1, 'CelebA':2,'WIDERFace':3,'SVHN':4, 'CIFAR100':5, 'VOC':6, 'Places365':7, 'MNIST':8, 'FashionMNIST':9, 'KMNIST':10, 'EMNIST':11},
            'CIFAR100': {'STL10':0, 'Caltech101':1, 'CelebA':2,'WIDERFace':3, 'SVHN':4, 'CIFAR10':5,'VOC':6, 'Places365':7,'MNIST':8, 'FashionMNIST':9, 'KMNIST':10, 'EMNIST':11},
            'Caltech101': {'STL10':0, 'CelebA':1,'WIDERFace':2, 'SVHN':3, 'CIFAR10':4,'CIFAR100':5, 'VOC':6, 'Places365':7,'MNIST':8, 'FashionMNIST':9, 'KMNIST':10, 'EMNIST':11},
            'Caltech256': {'STL10':0, 'CelebA':1,'WIDERFace':2,'SVHN':3, 'Caltech101':4, 'CIFAR10':5,'CIFAR100':6,'VOC':7, 'Places365':8,'MNIST':9, 'FashionMNIST':10, 'KMNIST':11, 'EMNIST':12},
}
'''

dataset_dict = {'MNIST':['FashionMNIST','KMNIST','EMNIST'],
            'FashionMNIST':['MNIST','KMNIST','EMNIST'],
            'KMNIST':['MNIST','FashionMNIST','EMNIST'],
            'EMNIST':['MNIST','FashionMNIST','KMNIST'],
            
            'CIFAR10':['STL10','Caltech101', 'CelebA','WIDERFace','SVHN', 'CIFAR100', 'VOC', 'Places365', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
            'CIFAR100':['STL10','Caltech101', 'CelebA','WIDERFace','SVHN', 'CIFAR10', 'VOC', 'Places365', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],

            'Caltech101':['STL10', 'CelebA','WIDERFace','SVHN', 'CIFAR10','CIFAR100', 'VOC', 'Places365', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
            'Caltech256':['STL10', 'CelebA','WIDERFace','SVHN','Caltech101', 'CIFAR10','CIFAR100', 'VOC', 'Places365', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
            'TinyImageNet':['STL10', 'CelebA','WIDERFace','SVHN','Caltech101', 'Caltech256','CIFAR10','CIFAR100', 'VOC', 'Places365', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
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
    
    for key in OOD_dict.keys():
        if ID_dataset.lower() in split_keys:
            ood_dataset = None
            return ood_dataset 
        elif key.lower() in split_keys:
            ood_dataset = key
            return ood_dataset

    print(f'{key} does not have OOD dataset')
    return None 

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

        summary_list.append(run.summary._json_dict)
        
        #updated path which includes the group of the dataset
        path_list = runs[i].path
        path_list.insert(-1, group_name) # insert the group name in the location one before the last value (rather than the last value which is peculiar)
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
def full_post_process_latex_table(df,caption,label):
    
    latex_table = df.to_latex()
    latex_table = replace_headings(df,latex_table)
    
    latex_table = bold_max_value(df,latex_table,)
    latex_table = post_process_latex_table(latex_table)
    latex_table = initial_table_info(latex_table)
    latex_table = add_caption(latex_table,caption)
    latex_table = add_label(latex_table,label) 
    latex_table = end_table_info(latex_table)
    #latex_table = '\\begin{table}[]\n\\centering\n' + latex_table + '\n\\caption{'+ caption + '}\n\\label{' + label + '}\n\\end{table}'

    return latex_table

# replace the headings 
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
        max_number = float(max(numbers,key=lambda x:float(x))) #  Need to get the output as a float
        #string[index] = string[index].replace(f'{max_number}',f'\textbf{ {max_number} }') # Need to put spaces around otherwise it just shows max number
        updated_string.append(string[index].replace(f'{max_number}',fr'\textbf{ {max_number} }')) # Need to put spaces around otherwise it just shows max number), also need to be place to make it so that \t does not act as space
        latex_table = latex_table.replace(f'{string[index]}',f'{updated_string[index]}') 
    return latex_table



def initial_table_info(latex_table):
    latex_table = '\\begin{table}[]\n\\centering\n' + latex_table
    return latex_table

def add_caption(latex_table,caption):
    latex_table = latex_table + '\n\\caption{'+ caption + '}'
    return latex_table

def add_label(latex_table,label):
    latex_table = latex_table + '\n\\label{' + label + '}'
    return latex_table

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
    #desired_key = 'Different K Normalized One Dim Class Typicality KNN'
    desired_key = 'Different K Normalized Quadratic One Dim Class Typicality KNN'
    #run_filter={"config.group":"Baselines Repeats"}
    #run_filter={"config.group":"OOD hierarchy baselines","config.model_type": "SupCon", 'state':'finished'}
    run_filter={"config.group":"OOD hierarchy baselines","config.model_type": "SupCon","config.dataset": "CIFAR100"}
    #run_filter={"config.group":"OOD hierarchy baselines","config.model_type": "SupCon"}
    #run_filter={"config.group":"New Model Testing","config.epochs":300}
    #run_filter={"config.group":"Baselines Repeats","$or": [{"config.model_type":"Moco"}, {"config.model_type": "SupCon"}]}
    generic_saving(desired_key,run_filter)