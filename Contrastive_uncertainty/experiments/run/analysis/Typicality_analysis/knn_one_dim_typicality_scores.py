# Plotting the the deviations of the different scores for different approaches

from cProfile import label
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import general params
import json
from Contrastive_uncertainty.experiments.run.analysis.analysis_utils import ood_dataset_string
# Examine the deviation for all the different points
def one_dim_typicality_scores_plot():
    # Desired ID,OOD and Model
    Models = ['SupCon']
    #Models = ['SupCon']
    desired_key = 'Normalized One Dim Scores Class Quadratic Typicality'
    desired_key = desired_key.lower()

    all_ID = ['CIFAR10','CIFAR100','Caltech256','TinyImageNet','Caltech101','Cub200','Dogs']
    #all_ID = ['TinyImageNet']
    all_OOD = {'CIFAR10':['MNIST','FashionMNIST','KMNIST','CIFAR10','CIFAR100','Caltech256','TinyImageNet'],
    'CIFAR100':['MNIST','FashionMNIST','KMNIST','CIFAR10','CIFAR100','Caltech256','TinyImageNet'],
    'Caltech256':['MNIST','FashionMNIST','KMNIST','CIFAR10','CIFAR100','Caltech256','TinyImageNet'],

    'Caltech101':['MNIST','FashionMNIST','KMNIST','CIFAR10','CIFAR100','Caltech256','TinyImageNet','SVHN','STL10','Caltech101', 'Cub200','Dogs'],
    'Cub200':['MNIST','FashionMNIST','KMNIST','CIFAR10','CIFAR100','Caltech256','TinyImageNet','SVHN','STL10','Caltech101', 'Cub200','Dogs'],
    'Dogs':['MNIST','FashionMNIST','KMNIST','CIFAR10','CIFAR100','Caltech256','TinyImageNet','SVHN','STL10','Caltech101', 'Cub200','Dogs'],
    
    'TinyImageNet':['MNIST','FashionMNIST','KMNIST', 'CIFAR10','CIFAR100','Caltech256','TinyImageNet']}

    for Model in Models:# Go through the different models
        for ID in all_ID: # Go through the different ID dataset
            api = wandb.Api()
            # Gets the runs corresponding to a specific filter
            # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py
            # Second part used of the filter used for the purpose of an or statement to calculate the different values (Shown in the github link above )
            #runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","$or": [{"config.model_type":"Moco" }, {"config.model_type": "SupCon"}]}) # only look at the runs related to Moco and SupCon
            # Need to use an 'and' statement to combine the different conditions on the approach
            seed = 50
            runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats","config.dataset":ID,"config.model_type":Model, "config.seed":seed})
            #runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.seed":[{"$ne": 26},{"$ne": 42}]})
            #Filtering based on https://docs.mongodb.com/manual/reference/operator/query/ne/#mongodb-query-op.-ne
            summary_list, config_list, name_list = [], [], []
            # Dict to map distances of specific datasets and model types to the data array
            root_dir = 'run_data/'
            collated_total_kl_values = []
            for i, run in enumerate(runs): 
                
                #print('run:',i)
                # .summary contains the output keys/values for metrics like accuracy.
                #  We call ._json_dict to omit large files 
                values = run.summary
                summary_list.append(run.summary._json_dict)
                run_path = '/'.join(runs[i].path)
                # .config contains the hyperparameters.
                #  We remove special values that start with _.
                config_list.append(
                    {k: v for k,v in run.config.items()
                     if not k.startswith('_')})
                # .name is the human-readable name of the run.dir
                name_list.append(run.name)
                group_name = config_list[i]['group']
                path_list = runs[i].path
                path_list.insert(-1, group_name) # insert the group name in the location one before the last value (rather than the last value which is peculiar)
                run_path = '/'.join(path_list)
                run_dir = root_dir + run_path
                #### Look at getting all the files with that particular name to run the simulation with ################
                keys = [key for key, value in summary_list[i].items() if desired_key in key.lower()]
                keys = [key for key in keys if 'table' not in key.lower()]
                
                for key in keys:
                    data_dir = summary_list[i][key]['path']
                    
                    run_dir = root_dir + run_path
                    OOD_string = ood_dataset_string(key,all_OOD,ID)
                    read_dir = os.path.join(run_dir, data_dir)
                    # Checks if the file is already present
                    dataset = config_list[i]['dataset']
                    model_type = config_list[i]['model_type']
                    with open(read_dir) as f:
                        one_dim_score_json = json.load(f)
                        one_dim_score_data = typicality_vector(one_dim_score_json)
                        # https://stackoverflow.com/questions/19125722/adding-a-legend-to-pyplot-in-matplotlib-in-the-simplest-manner-possible
                        plt.plot(one_dim_score_data[:,0], one_dim_score_data[:,1],label=f'{key}')
                        plt.title(f"1D Typicality deviations-ID:{ID}-OOD-{OOD_string}") 
                        plt.xlabel("Dimension") 
                        plt.ylabel("ID-OOD Score difference") 
                        #plt.legend(loc="upper right")
                        #plt.show()
                        folder = f'Scatter_Plots/Typicality_Scores/{model_type}/{ID}/Seed/{seed}'
                        
                        if not os.path.exists(folder):
                            os.makedirs(folder)
                        plt.savefig(f'{folder}/1D Typicality deviations-ID:{ID}-OOD-{OOD_string}.png')
                        plt.close()



# Gets the scores for a particular class of interest
def one_dim_typicality_scores_class_plot():
    # Desired ID,OOD and Model
    Models = ['SupCon']
    #Models = ['SupCon']
    desired_key = 'Analysis Normalized One Dim Scores Class Quadratic Typicality KNN'
    desired_key = desired_key.lower()

    all_ID = ['CIFAR10','CIFAR100']
    #all_ID = ['TinyImageNet']
    all_OOD = {'CIFAR10':['MNIST','FashionMNIST','KMNIST','CIFAR10','CIFAR100','Caltech256','TinyImageNet'],
    'CIFAR100':['MNIST','FashionMNIST','KMNIST','CIFAR10','CIFAR100','Caltech256','TinyImageNet'],
    'Caltech256':['MNIST','FashionMNIST','KMNIST','CIFAR10','CIFAR100','Caltech256','TinyImageNet'],

    'Caltech101':['MNIST','FashionMNIST','KMNIST','CIFAR10','CIFAR100','Caltech256','TinyImageNet','SVHN','STL10','Caltech101', 'Cub200','Dogs'],
    'Cub200':['MNIST','FashionMNIST','KMNIST','CIFAR10','CIFAR100','Caltech256','TinyImageNet','SVHN','STL10','Caltech101', 'Cub200','Dogs'],
    'Dogs':['MNIST','FashionMNIST','KMNIST','CIFAR10','CIFAR100','Caltech256','TinyImageNet','SVHN','STL10','Caltech101', 'Cub200','Dogs'],
    
    'TinyImageNet':['MNIST','FashionMNIST','KMNIST', 'CIFAR10','CIFAR100','Caltech256','TinyImageNet']}

    for Model in Models:# Go through the different models
        for ID in all_ID: # Go through the different ID dataset
            api = wandb.Api()
            # Gets the runs corresponding to a specific filter
            # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py
            # Second part used of the filter used for the purpose of an or statement to calculate the different values (Shown in the github link above )
            #runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","$or": [{"config.model_type":"Moco" }, {"config.model_type": "SupCon"}]}) # only look at the runs related to Moco and SupCon
            # Need to use an 'and' statement to combine the different conditions on the approach
            seed = 50
            runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats","config.dataset":ID,"config.model_type":Model, "config.seed":seed})
            #runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.seed":[{"$ne": 26},{"$ne": 42}]})
            #Filtering based on https://docs.mongodb.com/manual/reference/operator/query/ne/#mongodb-query-op.-ne
            summary_list, config_list, name_list = [], [], []
            # Dict to map distances of specific datasets and model types to the data array
            root_dir = 'run_data/'
            collated_total_kl_values = []
            for i, run in enumerate(runs): 
                
                #print('run:',i)
                # .summary contains the output keys/values for metrics like accuracy.
                #  We call ._json_dict to omit large files 
                values = run.summary
                summary_list.append(run.summary._json_dict)
                run_path = '/'.join(runs[i].path)
                # .config contains the hyperparameters.
                #  We remove special values that start with _.
                config_list.append(
                    {k: v for k,v in run.config.items()
                     if not k.startswith('_')})
                # .name is the human-readable name of the run.dir
                name_list.append(run.name)
                group_name = config_list[i]['group']
                path_list = runs[i].path
                path_list.insert(-1, group_name) # insert the group name in the location one before the last value (rather than the last value which is peculiar)
                run_path = '/'.join(path_list)
                run_dir = root_dir + run_path
                #### Look at getting all the files with that particular name to run the simulation with ################
                keys = [key for key, value in summary_list[i].items() if desired_key in key.lower()]
                keys = [key for key in keys if 'table' not in key.lower()]
                
                for key in keys:
                    data_dir = summary_list[i][key]['path']
                    
                    run_dir = root_dir + run_path
                    OOD_string = ood_dataset_string(key,all_OOD,ID)
                    read_dir = os.path.join(run_dir, data_dir)
                    # Checks if the file is already present
                    dataset = config_list[i]['dataset']
                    model_type = config_list[i]['model_type']
                    with open(read_dir) as f:
                        one_dim_score_json = json.load(f)
                        one_dim_score_data = typicality_vector(one_dim_score_json)
                        # https://stackoverflow.com/questions/19125722/adding-a-legend-to-pyplot-in-matplotlib-in-the-simplest-manner-possible
                        plt.plot(one_dim_score_data[:,0], one_dim_score_data[:,1],label=f'{key}')
                        plt.title(f"1D Typicality deviations-ID:{ID}-OOD-{OOD_string}") 
                        plt.xlabel("Dimension") 
                        plt.ylabel("ID-OOD Score difference") 
                        #plt.legend(loc="upper right")
                        #plt.show()
                        folder = f'Scatter_Plots/Typicality_Scores/{model_type}/{ID}/Seed/{seed}'
                        
                        if not os.path.exists(folder):
                            os.makedirs(folder)
                        plt.savefig(f'{folder}/1D Typicality deviations-ID:{ID}-OOD-{OOD_string}.png')
                        plt.close()


def typicality_vector(json_data):
    data = np.array(json_data['data'])
    data = np.around(data,decimals=3)    
    return data

if __name__ == '__main__':
    one_dim_typicality_scores_plot()
    #total_kl_clp_plot()