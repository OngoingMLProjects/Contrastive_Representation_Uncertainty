# Script used to place the spectral values in  a bar chart

from json.tool import main
import wandb
import pandas as pd
import numpy as np

# Import general params
import json
import math
import matplotlib
matplotlib.rcParams['text.usetex'] = True # Makes it so that latex can be used 
import matplotlib.pyplot as plt

def obtain_spectral_values():
    key_dict = {'model_type':{'CE':0, 'Moco':1, 'SupCon':2},
                'dataset': {'MNIST':0, 'FashionMNIST':1, 'KMNIST':2, 'CIFAR10':3,'CIFAR100':4}}
    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py

    # Second part used of the filter used for the purpose of an or statement to calculate the different values (Shown in the github link above )
    #runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines"})
    
    #parameter = 'dists@intra: instance: fine'  
    parameter1 = 'rho_spectrum@1: instance: fine'
    parameter2 = 'rho_spectrum@2: instance: fine'
    data_array = np.empty((5,3))
    data_array[:] = np.nan
    for dataset in key_dict['dataset'].keys():
    
        runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines",'config.seed':42,'config.epochs':300,'config.dataset':dataset})
    
        # Five different datasets with 3 models
        summary_list, config_list = [], []
        for i, run in enumerate(runs): 
            #import ipdb; ipdb.set_trace()
            # .summary contains the output keys/values for metrics like accuracy.
            #  We call ._json_dict to omit large files 
            values = run.summary
            summary_list.append(run.summary._json_dict)
    
            # .config contains the hyperparameters.
            #  We remove special values that start with _.
            config_list.append(
                {k: v for k,v in run.config.items()
                 if not k.startswith('_')})
    
            ID_dataset = config_list[i]['dataset']
            model_type = config_list[i]['model_type']
            value = abs(summary_list[i][parameter1])  #- summary_list[i][parameter2]) 
            column = key_dict['model_type'][model_type]
            row = key_dict['dataset'][ID_dataset]
            data_array[row, column] = np.around(value,decimals=3)
    
    import ipdb; ipdb.set_trace()
    # Obtain the names of the rows and the name of the columns
    column_names = ['SupCLR' if model =='SupCon' else model for model in key_dict['model_type'].keys()]
    row_names = [dataset for dataset in key_dict['dataset'].keys()]
    data_df = pd.DataFrame(data_array, columns = column_names, index = row_names)
    ax =data_df.plot.bar()
    plt.title('Spectral Decay for different datasets')
    plt.ylabel(r'$\rho$')
    plt.tight_layout()
    plt.show()
    #plt.savefig('spectral_values.png')
    #plt.savefig('absolute MMD distance.png')
    
if __name__ == '__main__':
    obtain_spectral_values()