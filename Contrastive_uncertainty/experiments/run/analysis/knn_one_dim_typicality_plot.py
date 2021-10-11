# Plotting the AUROC as the value of K changes for the group typicality approach

from Contrastive_uncertainty.general.callbacks.ood_callbacks import Mahalanobis_OOD
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import general params
import json

from Contrastive_uncertainty.experiments.run.analysis.analysis_utils import dataset_dict, key_dict, ood_dataset_string

def knn_vector(json_data):
    data = np.array(json_data['data'])
    knn_values = np.around(data,decimals=3)
    return knn_values

def knn_auroc_plot():
    # Desired ID,OOD and Model  
    root_dir = 'run_data/'

    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py
    #runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.model_type": "SupCon","config.epochs": 300, 'state':'finished'})

    runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"New Model Testing","config.epochs":300})

    summary_list, config_list, name_list = [], [], []
    key_dict = {'dataset':{'MNIST':0, 'FashionMNIST':1,'KMNIST':2, 'CIFAR10':3, 'CIFAR100':4,'Caltech101':5,'Caltech256':6},
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
        desired_string = 'Different K Normalized One Dim Class Typicality KNN OOD'.lower()
        knn_keys = [key for key, value in summary_list[i].items() if desired_string in key.lower()]

        for key in knn_keys:
            
            OOD_dataset = ood_dataset_string(key, dataset_dict, ID_dataset)
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
            sns.regplot(x = df['K'], y = df['AUROC'],color='blue')
            plt.annotate('y={:.3f}+{:.4f}*x'.format(fit[1], fit[0]), xy=(0.05, 0.95), xycoords='axes fraction')
            #plt.text(3.2, -7.12, 'y={:.2f}+{:.2f}*x'.format(fit[1], fit[0]), color='darkblue', size=12)
            plt.title(f'AUROC for different K values  {ID_dataset}-{OOD_dataset} pair using {Model_name} model', size=12)
            # regression equations
            folder = f'Scatter_Plots/{Model_name}'

            if not os.path.exists(folder):
                os.makedirs(folder)
            
            plt.savefig(f'{folder}/AUROC for different K values  {ID_dataset}-{OOD_dataset} {Model_name}.png')
            plt.close()
                    



def knn_auroc_plot_v2():
    # Desired ID,OOD and Model  
    root_dir = 'run_data/'

    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py
    runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.model_type": "SupCon","config.epochs": 300, 'state':'finished'})

    #runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"New Model Testing","config.epochs":300})

    summary_list, config_list, name_list = [], [], []
    '''
    key_dict = {'dataset':{'MNIST':0, 'FashionMNIST':1,'KMNIST':2, 'CIFAR10':3, 'CIFAR100':4},
                'model_type':{'SupCon':0}}
    '''

    key_dict = {'dataset':{'CIFAR10':0, 'CIFAR100':1},
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
        desired_string = 'Different K Normalized One Dim Class Typicality KNN OOD'.lower()
        knn_keys = [key for key, value in summary_list[i].items() if desired_string in key.lower()]
        

        baseline_string = 'Mahalanobis AUROC OOD'.lower() #if group_name =='New Model Testing' else 'Mahalanobis AUROC: instance vector'.lower()
        mahalanobis_keys = [key for key, value in summary_list[i].items() if baseline_string in key.lower()]
        mahalanobis_keys = [key for key in mahalanobis_keys if 'relative' not in key.lower()]
        for key in knn_keys:
            OOD_dataset = ood_dataset_string(key, dataset_dict, ID_dataset)
            for mahalanobis_key in mahalanobis_keys:
                mahalanobis_OOD_dataset = ood_dataset_string(key, dataset_dict, ID_dataset)
                mahalanobis_AUROC = 0
                if mahalanobis_OOD_dataset == OOD_dataset:
                    mahalanobis_AUROC = summary_list[i][mahalanobis_key]
                            
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
                    folder = f'Scatter_Plots/{Model_name}'

                    if not os.path.exists(folder):
                        os.makedirs(folder)

                    plt.savefig(f'{folder}/AUROC for different K values  {ID_dataset}-{OOD_dataset} {Model_name}.png')
                    plt.close()


def knn_auroc_plot_v3():
    # Desired ID,OOD and Model  
    root_dir = 'run_data/'

    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py
    runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.model_type": "SupCon","config.epochs": 300})


    summary_list, config_list, name_list = [], [], []
    
    key_dict = {'dataset':{'MNIST':0, 'FashionMNIST':1,'KMNIST':2, 'CIFAR10':3, 'CIFAR100':4,'Caltech101':5,'Caltech256':6},
                'model_type':{'SupCon':0}}
    
    '''
    key_dict = {'dataset':{'CIFAR10':0, 'CIFAR100':1},
                'model_type':{'SupCon':0}}
    '''
    
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
        desired_string = 'Different K Normalized One Dim Class Typicality KNN OOD'.lower()
        knn_keys = [key for key, value in summary_list[i].items() if desired_string in key.lower()]
        
        
        baseline_string_1, baseline_string_2 = 'Mahalanobis AUROC OOD'.lower(), 'Mahalanobis AUROC: instance vector'.lower() 

        #baseline_string_1, baseline_string_2 = 'Mahalanobis AUROC: instance vector'.lower(), 'Mahalanobis AUROC: instance vector'.lower()

        baseline_mahalanobis_keys_1 = [key for key, value in summary_list[i].items() if baseline_string_1 in key.lower()]
        baseline_mahalanobis_keys_1 = [key for key in baseline_mahalanobis_keys_1 if 'relative' not in key.lower()]

        baseline_mahalanobis_keys_2 = [key for key, value in summary_list[i].items() if baseline_string_2 in key.lower()]
        baseline_mahalanobis_keys_2 = [key for key in baseline_mahalanobis_keys_2 if 'relative' not in key.lower()]
        baseline_mahalanobis_keys = [*baseline_mahalanobis_keys_1,*baseline_mahalanobis_keys_2] 
        
        # go through the different knn keys
        for key in knn_keys:
            OOD_dataset = ood_dataset_string(key, dataset_dict, ID_dataset)

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
                folder = f'Scatter_Plots/{Model_name}/{ID_dataset}'
                if not os.path.exists(folder):
                    os.makedirs(folder)
                plt.savefig(f'{folder}/AUROC for different K values  {ID_dataset}-{OOD_dataset} {Model_name}.png')
                plt.close()

if __name__== '__main__':
    knn_auroc_plot_v3()
    #knn_auroc_plot()