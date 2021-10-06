import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import general params
import json
# For each ID dataset, it maps the dict to another value
'''
dataset_dict = {'MNIST': {'FashionMNIST':0, 'KMNIST':1,'EMNIST':2},
            'FashionMNIST': {'MNIST':0, 'KMNIST':1,'EMNIST':2},
            'KMNIST': {'MNIST':0, 'FashionMNIST':1,'EMNIST':2},
            'CIFAR10': {'SVHN':0, 'CIFAR100':1},
            'CIFAR100': {'SVHN':0, 'CIFAR10':1}
}
'''
'''
dataset_dict = {'CIFAR10': {'STL10':0,'Caltech101':1, 'CelebA':2,'WIDERFace':3,'SVHN':4, 'CIFAR100':5},
            'CIFAR100': {'STL10':0, 'Caltech101':1, 'CelebA':2,'WIDERFace':3, 'SVHN':4, 'CIFAR10':5}
}
'''

dataset_dict = {'MNIST': {'FashionMNIST':0, 'KMNIST':1,'EMNIST':2},
            'FashionMNIST': {'MNIST':0, 'KMNIST':1,'EMNIST':2},
            'KMNIST': {'MNIST':0, 'FashionMNIST':1,'EMNIST':2},
            'CIFAR10': {'STL10':0,'Caltech101':1, 'CelebA':2,'WIDERFace':3,'SVHN':4, 'CIFAR100':5, 'VOC':6, 'Places365':7, 'MNIST':8, 'FashionMNIST':9, 'KMNIST':10, 'EMNIST':11},
            'CIFAR100': {'STL10':0, 'Caltech101':1, 'CelebA':2,'WIDERFace':3, 'SVHN':4, 'CIFAR10':5,'VOC':6, 'Places365':7,'MNIST':8, 'FashionMNIST':9, 'KMNIST':10, 'EMNIST':11}
}

            
# Dict for the specific case to the other value
key_dict = {'model_type':{'CE':0, 'Moco':1, 'SupCon':2},
            'dataset': {'MNIST':0, 'FashionMNIST':1, 'KMNIST':2, 'CIFAR10':3,'CIFAR100':4}}

# Check if ood_dataset substring is present in string
def ood_dataset_string(key, dataset_dict, ID_dataset):
    split_keys = key.lower().split() # Make the key lower and then split the string at locations where is a space
    OOD_dict = dataset_dict[ID_dataset]
    for key in OOD_dict.keys():
        if key.lower() in split_keys:
            ood_dataset = key

    return ood_dataset

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
            file_data = json.load(run.file(data_dir).download(root=run_dir))

if __name__ =='__main__':
    #desired_key = 'Centroid Distances Average vector_table'
    #desired_key = 'KL Divergence(Total||Class)'
    desired_key = 'Different K Normalized One Dim Class Typicality KNN'
    #run_filter={"config.group":"Baselines Repeats"}
    #run_filter={"config.group":"OOD hierarchy baselines","config.model_type": "SupCon", 'state':'finished'}
    run_filter={"config.group":"OOD hierarchy baselines","config.model_type": "SupCon"}
    #run_filter={"config.group":"New Model Testing","config.epochs":300}
    #run_filter={"config.group":"Baselines Repeats","$or": [{"config.model_type":"Moco"}, {"config.model_type": "SupCon"}]}
    generic_saving(desired_key,run_filter)