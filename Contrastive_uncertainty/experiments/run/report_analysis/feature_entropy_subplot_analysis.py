# Script used to save the typicality AUROC values

from pandas.io.pytables import DataCol
import wandb
import pandas as pd
import numpy as np

# Import general params
import json
import math

from ood_centroid_analysis import dataset_dict,key_dict, ood_dataset_string
import matplotlib
matplotlib.rcParams['text.usetex'] = True # Makes it so that latex can be used 
import matplotlib.pyplot as plt


def feature_entropy_subplot():
    all_ID = ['MNIST','FashionMNIST','KMNIST','CIFAR10']
    #desired_key = 'marginal feature entropy'
    desired_key = 'class conditional feature entropy'
    datadict = {}
    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py
    plt.figure(figsize=(12,12)) # Changing size of the figure before I make it into subplots
    subplot_num = 1
    for ID in all_ID:
        # Second part used of the filter used for the purpose of an or statement to calculate the different values (Shown in the github link above )
        runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines"})

        summary_list, config_list, name_list = [], [], []

        root_dir = 'run_data/'
        for i, run in enumerate(runs): 
            #import ipdb; ipdb.set_trace()
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

            ID_dataset = config_list[i]['dataset']
            model_type = config_list[i]['model_type']
            # .name is the human-readable name of the run.dir
            name_list.append(run.name)
            keys = [key for key, value in summary_list[i].items() if desired_key in key.lower()]
            if ID == ID_dataset:
                for key in keys:
                
                    data_dir = summary_list[i][key]['path'] 
                    run_dir = root_dir + run_path

                    # Read dir is how to read the file
                    read_dir = run_dir + '/' + data_dir

                    with open(read_dir) as f: 
                        feature_entropy_json = json.load(f)

                    entropy_data = pd.DataFrame(feature_entropy_json['data'])
                    entropy_data.columns = ['Dimension', 'Feature Entropy']
                    datadict[model_type] = entropy_data
                    #plt.show()



        datadict['SupCLR'] = datadict['SupCon']
        del datadict['SupCon']
        plt.subplot(4, 1, subplot_num)

        plt.plot( 'Dimension', 'Feature Entropy', data=datadict['CE'], marker='', color='red', linewidth=2,label ='CE')
        plt.plot( 'Dimension', 'Feature Entropy', data=datadict['Moco'], marker='', color='olive', linewidth=2,label='Moco')
        plt.plot( 'Dimension', 'Feature Entropy', data=datadict['SupCLR'], marker='', color='skyblue', linewidth=2, linestyle='dashed', label="SupCLR")
        plt.ylim(-10,0)
        plt.xlabel('Feature Dimension')
        plt.title(f'{ID}', size=12)
        if desired_key == 'marginal feature entropy':
            plt.ylabel('Differential Entropy')
            plt.title(f'Differential Entropy for different feature dimensions for the {ID} Dataset', size=12)
        else:
            plt.ylabel('')
            plt.title(f'Class Conditional Differential Entropy for different feature dimensions for the {ID} Dataset', size=12)
        # show legend
        plt.legend()
        # replace the space with a value for the _
        subplot_num +=1
        print('subplot_num:',subplot_num)
    
    
    plt.subplots_adjust(left=0.15,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.8)
    saving_name = 'collated_'+ desired_key.replace(' ','_')
    plt.savefig(f'{saving_name}_{ID}.png')
#feature_entropy_saving()
if __name__ == '__main__':
    feature_entropy_subplot()