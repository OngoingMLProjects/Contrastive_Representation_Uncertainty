# Plotting the centroid distances against the confuson log probability

from re import S
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import general params
import json



def centroid_distance_plot():
    # Desired ID,OOD and Model
    #Models = ['Moco']
    #Models = ['CE','Moco','SupCon']
    Models = ['CE']
    desired_group = "OOD hierarchy baselines"
    desired_key = 'Centroid Distances Average vector_table'
    desired_key = desired_key.lower()
    
    all_ID = ['MNIST','FashionMNIST','KMNIST', 'CIFAR10','CIFAR100']
    all_OOD = {'MNIST':['FashionMNIST','KMNIST'],
    'FashionMNIST':['MNIST','KMNIST'],
    'KMNIST':['MNIST','FashionMNIST'],
    'CIFAR10':['SVHN','CIFAR100'],
    'CIFAR100':['SVHN','CIFAR10']}
    

    #all_ID = ['CIFAR100']
    #all_OOD = {'CIFAR100':['SVHN','CIFAR10']}
    for Model in Models:# Go through the different models
        for ID in all_ID: # Go through the different ID dataset
            
            api = wandb.Api()
            runs = api.runs(path="nerdk312/evaluation", filters={"config.group":desired_group,"config.dataset":ID,"config.model_type":Model, 'config.seed':42, 'config.epochs':300})

            summary_list, config_list, name_list = [], [], []
            root_dir = 'run_data/'
            for i, run in enumerate(runs): 
                    
                # .summary contains the output keys/values for metrics like accuracy.
                #  We call ._json_dict to omit large files 
                values = run.summary
                summary_list.append(run.summary._json_dict)
                print(f'ID {ID}, Model {Model}')

                keys = [key for key, value in summary_list[i].items() if desired_key in key.lower()]
                    
                #keys = [key for key in keys if 'table' not in key.lower()]    
                #keys = [key for key in keys if OOD.lower() in key.lower()]
                final_key = keys[0]

                run_path = '/'.join(runs[i].path)

                # .config contains the hyperparameters.
                #  We remove special values that start with _.
                config_list.append(
                    {k: v for k,v in run.config.items()
                        if not k.startswith('_')})

                # .name is the human-readable name of the run.dir
                name_list.append(run.name)

                group_name = config_list[i]['group']
                ID_dataset = config_list[i]['dataset']
                model_type = config_list[i]['model_type']

                seed =  str(config_list[i]['seed'])

                if model_type =='SupCon': #  Need to use it here as the model name 
                    model_name = 'SupCLR'
                else:
                    model_name = model_type

                path_list = runs[i].path
                path_list.insert(-1, group_name) # insert the group name in the location one before the last value (rather than the last value which is peculiar)
                path_list.insert(-1,model_type)
                path_list.insert(-1,ID_dataset)
                path_list.insert(-1,seed)

                run_path = '/'.join(path_list)
                run_dir = root_dir + run_path
                    
                data_dir =  summary_list[i][final_key]['path']
                run_dir = root_dir + run_path
                # Read dir is how to read the file
                    
                read_dir = run_dir + '/' + data_dir

                with open(read_dir) as f:
                    centroid_distance_data = json.load(f)
                    centroid_distance_values = centroid_distance_vector(centroid_distance_data)
                    class_values = np.arange(len(centroid_distance_values))
                    
                    collated_array = np.stack((class_values, centroid_distance_values),axis=1)
                    df = pd.DataFrame(collated_array)
                    columns = ['Class', 'Centroid Distance']
                    df.columns = columns
                    #plt.ylim(bottom=0)  # adjust the bottom leaving top unchanged)
                    #plt.ylim(0,7.5)
                    plt.ylim(0,15)
                    plt.title(f'Class Centroid Distance for {ID} using {model_name}')
                    sns.scatterplot(data=df, x="Class", y="Centroid Distance")

                    folder = f'Scatter_Plots/centroid_distance_plots/{model_name}'
                        
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    plt.savefig(f'{folder}/centroid_distance_{ID}_{model_name}.png')
                    plt.close()



def centroid_distance_vector(json_data):
    data = np.array(json_data['data'])
    data_list = data[:,1] # Take allof them beside the last one as that is all (and does not correspond to a particular class)
    centroid_distance_values = np.around(data_list,decimals=2)
    #ood_fraction_values = np.around((data[:,3]),decimals=2)
    return centroid_distance_values

centroid_distance_plot()