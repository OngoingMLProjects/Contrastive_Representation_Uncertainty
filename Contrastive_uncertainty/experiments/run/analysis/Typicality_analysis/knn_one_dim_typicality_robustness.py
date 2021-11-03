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
    runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.model_type": "SupCon","config.epochs": 300})

    summary_list, config_list, name_list = [], [], []
    
    key_dict = {'dataset':{'CIFAR10':0, 'CIFAR100':1,'Caltech101':2,'Caltech256':3,'TinyImageNet':4,'Cub200':4,'Dogs':5},
                'model_type':{'SupCon':0}}
    
    for i, run in enumerate(runs): 
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 

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
        desired_quadratic_string = 'Different K Normalized Quadratic One Dim Class Typicality KNN'.lower()
        
        quadratic_knn_keys = [key for key, value in summary_list[i].items() if desired_quadratic_string in key.lower()]

        desired_linear_string = 'Different K Normalized One Dim Class Typicality KNN OOD'.lower()
        quadratic_knn_keys = [key for key, value in summary_list[i].items() if desired_quadratic_string in key.lower()]

        # Make a function which can give me the quadratic value for a particular ID and OOD dataset
        # Make a function which can give me the linear typicality for a particular ID and OOD dataset
        # Make it so that I can show the different values for the approach.


        # go through the different knn keys
        for key in knn_keys:
            OOD_dataset = ood_dataset_string(key, dataset_dict, ID_dataset)
            if OOD_dataset is None:
                pass
            else:
                # get the specific mahalanobis keys for the specific OOD dataset
                OOD_dataset_specific_mahalanobis_keys = [key for key in baseline_mahalanobis_keys if OOD_dataset.lower() in key.lower()]
                # https://stackoverflow.com/questions/16380326/check-if-substring-is-in-a-list-of-strings/16380569
                OOD_dataset_specific_mahalanobis_key = [key for key in OOD_dataset_specific_mahalanobis_keys if baseline_string_1 in key.lower()] 
                # Make it so that I choose baseline string 2 if the first case has no strings 
                if len(OOD_dataset_specific_mahalanobis_key) == 0:
                    OOD_dataset_specific_mahalanobis_key = [key for key in OOD_dataset_specific_mahalanobis_keys if baseline_string_2 in key.lower()]

                # if there is no mahalanobis key for the KNN situation, pass otherwise plot graph
                if len(OOD_dataset_specific_mahalanobis_key) == 0:
                    pass
                else:
                    mahalanobis_AUROC = summary_list[i][OOD_dataset_specific_mahalanobis_key[0]]
                    print(f'ID: {ID_dataset}, OOD {OOD_dataset}:{round(mahalanobis_AUROC,3)}')
                    data_dir = summary_list[i][key]['path']
                    run_dir = root_dir + run_path
                    read_dir = run_dir + '/' + data_dir
                    with open(read_dir) as f: 
                        data = json.load(f)

                    knn_values = knn_vector(data)
                    df = pd.DataFrame(knn_values)
                    columns = ['K', 'AUROC']
                    df.columns = columns

                    fit = np.polyfit(df['K'], df['AUROC'], 1)
                    fig = plt.figure(figsize=(10, 7))
                    plt.hlines(mahalanobis_AUROC,xmin= knn_values[0,0],xmax = knn_values[-1,0],linestyles='dotted')
                    sns.regplot(x = df['K'], y = df['AUROC'],color='blue')
                    plt.annotate('y={:.3f}+{:.4f}*x'.format(fit[1], fit[0]), xy=(0.05, 0.95), xycoords='axes fraction')
                    #plt.text(3.2, -7.12, 'y={:.2f}+{:.2f}*x'.format(fit[1], fit[0]), color='darkblue', size=12)
                    plt.title(f'AUROC for different K values  {ID_dataset}-{OOD_dataset} pair using {Model_name} model', size=12)

                    # regression equations
                    folder = f'Scatter_Plots/{Model_name}/{ID_dataset}/{approach}'
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    plt.savefig(f'{folder}/AUROC for different K values {ID_dataset}-{OOD_dataset} {Model_name}.png')
                    plt.close()
