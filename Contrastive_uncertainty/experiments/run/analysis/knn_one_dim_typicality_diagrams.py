# Plotting the AUROC as the value of K changes for the group typicality approach
# Also works on making tables for the ID and OOD datasets

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

# Import general params
import json
import re

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
    key_dict = {'dataset':{'MNIST':0, 'FashionMNIST':1,'KMNIST':2, 'CIFAR10':3, 'CIFAR100':4,'Caltech101':5,'Caltech256':6,'TinyImageNet':7,'Cub200':8,'Dogs':9},
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
    
    key_dict = {'dataset':{'MNIST':0, 'FashionMNIST':1,'KMNIST':2, 'CIFAR10':3, 'CIFAR100':4,'Caltech101':5,'Caltech256':6,'TinyImageNet':7,'Cub200':8,'Dogs':9},
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



def knn_auroc_table():
    # Fixed value of k of interest
    fixed_k = 10
    # Desired ID,OOD and Model  
    root_dir = 'run_data/'

    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py
    runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.model_type": "SupCon","config.epochs": 300, 'state':'finished'})


    summary_list, config_list, name_list = [], [], []
    
    key_dict = {'dataset':{'MNIST':0, 'FashionMNIST':1,'KMNIST':2, 'CIFAR10':3, 'CIFAR100':4,'Caltech101':5,'Caltech256':6,'TinyImageNet':7,'Cub200':8,'Dogs':9},
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
        
        
        baseline_string_1, baseline_string_2 = 'Mahalanobis AUROC OOD'.lower(), 'Mahalanobis AUROC: instance vector'.lower() 

        #baseline_string_1, baseline_string_2 = 'Mahalanobis AUROC: instance vector'.lower(), 'Mahalanobis AUROC: instance vector'.lower()

        baseline_mahalanobis_keys_1 = [key for key, value in summary_list[i].items() if baseline_string_1 in key.lower()]
        baseline_mahalanobis_keys_1 = [key for key in baseline_mahalanobis_keys_1 if 'relative' not in key.lower()]

        baseline_mahalanobis_keys_2 = [key for key, value in summary_list[i].items() if baseline_string_2 in key.lower()]
        baseline_mahalanobis_keys_2 = [key for key in baseline_mahalanobis_keys_2 if 'relative' not in key.lower()]
        baseline_mahalanobis_keys = [*baseline_mahalanobis_keys_1,*baseline_mahalanobis_keys_2] 
        
        
        # Make a data array, where the number values are equal to the number of OOD classes present in the datamodule dict or equal to the number of keys
        
        # number of OOd datasets for this particular ID dataset
        num_ood = len(dataset_dict[ID_dataset])
        # data array
        data_array = np.empty((num_ood,3))
        data_array[:] = np.nan
        # name for the different rows of a table
        #row_names = []
        row_names = [None] * num_ood # Make an empty list to take into account all the different values 
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
                mahalanobis_AUROC = round(summary_list[i][OOD_dataset_specific_mahalanobis_key[0]],3)
                #print(f'ID: {ID_dataset}, OOD {OOD_dataset}:{round(mahalanobis_AUROC,3)}')
                data_dir = summary_list[i][key]['path']
                run_dir = root_dir + run_path
                read_dir = run_dir + '/' + data_dir
                with open(read_dir) as f: 
                    data = json.load(f)
                
                knn_values = knn_vector(data)

                #### obtain for specific k value
                indices = knn_values[:,0] # get all the values in the first column (get all the k values as indices)
                knn_df = pd.DataFrame(knn_values,index = indices)
                fixed_k_knn_value = knn_df.loc[fixed_k][1] # obtains the k value and the AUROC for the k value which is the fixed k, then takes index 1 which is the AUROC value
                ###### obtain optimal value ########
                optimal_knn_AUROC = np.max(knn_values[:,1])

                data_index = dataset_dict[ID_dataset][OOD_dataset]
                print(f'ID: {ID_dataset}, OOD {OOD_dataset}:{round(mahalanobis_AUROC,3)}:index {data_index}')
                data_array[data_index,0] = mahalanobis_AUROC 
                data_array[data_index,1] = fixed_k_knn_value
                data_array[data_index,2] = optimal_knn_AUROC

                row_names[data_index] = f'ID:{ID_dataset}, OOD:{OOD_dataset}' 
                #row_names.append(f'ID:{ID_dataset}, OOD:{OOD_dataset}')

        column_names = ['Baseline', f'{fixed_k} NN', 'Optimal KNN']
        auroc_df = pd.DataFrame(data_array,columns = column_names, index=row_names)
            
        latex_table = auroc_df.to_latex()
        
        latex_table = latex_table.replace('{}','{Datasets}')
        latex_table = latex_table.replace("lrrr","|p{3cm}|c|c|c|")
        latex_table = post_process_latex_table(latex_table)
        
        

        # Replacing the string with the max value so it can be seen
        #https://stackoverflow.com/questions/4703390/how-to-extract-a-floating-number-from-a-string
        #https://www.youtube.com/watch?v=sa-TUpSx1JA  - explanation, \s means space and + means 1 or more, whilst \d means decimal and \. means a . (\) used in front as . by itself is a special character 
        string = re.findall("&\s+\d+\.\d+\s+&\s+\d+\.\d+\s+&\s+\d+\.\d+", latex_table)
        updated_string = []
        for index in range(len(string)):
            numbers = re.findall("\d+\.\d+", string[index]) # fnd all the numbers in the substring (gets rid of the &)
            #max_number = max(numbers,key=lambda x:float(x))
            max_number = float(max(numbers,key=lambda x:float(x))) #  Need to get the output as a float
            
            #string[index] = string[index].replace(f'{max_number}',f'\textbf{ {max_number} }') # Need to put spaces around otherwise it just shows max number
            updated_string.append(string[index].replace(f'{max_number}',fr'\textbf{ {max_number} }')) # Need to put spaces around otherwise it just shows max number), also need to be place to make it so that \t does not act as space
            latex_table = latex_table.replace(f'{string[index]}',f'{updated_string[index]}') 

        print(latex_table)
            

# Same as version 1 but includes the results from the quadratic typicality
def knn_auroc_table_v2():
    # Fixed value of k of interest
    fixed_k = 10
    # Desired ID,OOD and Model  
    root_dir = 'run_data/'

    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py
    runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.model_type": "SupCon","config.epochs": 300, 'state':'finished'})


    summary_list, config_list, name_list = [], [], []
    
    key_dict = {'dataset':{'MNIST':0, 'FashionMNIST':1,'KMNIST':2, 'CIFAR10':3, 'CIFAR100':4,'Caltech101':5,'Caltech256':6,'TinyImageNet':7,'Cub200':8,'Dogs':9},
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
        
        
        baseline_string_1, baseline_string_2 = 'Mahalanobis AUROC OOD'.lower(), 'Mahalanobis AUROC: instance vector'.lower() 

        #baseline_string_1, baseline_string_2 = 'Mahalanobis AUROC: instance vector'.lower(), 'Mahalanobis AUROC: instance vector'.lower()

        baseline_mahalanobis_keys_1 = [key for key, value in summary_list[i].items() if baseline_string_1 in key.lower()]
        baseline_mahalanobis_keys_1 = [key for key in baseline_mahalanobis_keys_1 if 'relative' not in key.lower()]

        baseline_mahalanobis_keys_2 = [key for key, value in summary_list[i].items() if baseline_string_2 in key.lower()]
        baseline_mahalanobis_keys_2 = [key for key in baseline_mahalanobis_keys_2 if 'relative' not in key.lower()]
        baseline_mahalanobis_keys = [*baseline_mahalanobis_keys_1,*baseline_mahalanobis_keys_2] 

        quadratic_string = 'Normalized One Dim Class Quadratic Typicality KNN'.lower()
        quadratic_typicality_keys = [key for key, value in summary_list[i].items() if quadratic_string in key.lower()]
        # Make a data array, where the number values are equal to the number of OOD classes present in the datamodule dict or equal to the number of keys
        
        # number of OOd datasets for this particular ID dataset
        num_ood = len(dataset_dict[ID_dataset])
        # data array
        data_array = np.empty((num_ood,4))
        #data_array = np.empty((num_ood,3))
        data_array[:] = np.nan
        # name for the different rows of a table
        #row_names = []
        # https://stackoverflow.com/questions/10712002/create-an-empty-list-in-python-with-certain-size
        row_names = [None] * num_ood # Make an empty list to take into account all the different values 
        # go through the different knn keys

        for key in knn_keys:
            OOD_dataset = ood_dataset_string(key, dataset_dict, ID_dataset)
            # Get ood dataset specific key
            quadratic_ood_dataset_specific_key = [key for key in quadratic_typicality_keys if OOD_dataset.lower() in key.lower()]

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
                mahalanobis_AUROC = round(summary_list[i][OOD_dataset_specific_mahalanobis_key[0]],3)

                quadratic_auroc = round(summary_list[i][quadratic_ood_dataset_specific_key[0]],3)
                #print(f'ID: {ID_dataset}, OOD {OOD_dataset}:{round(mahalanobis_AUROC,3)}')
                data_dir = summary_list[i][key]['path']
                run_dir = root_dir + run_path
                read_dir = run_dir + '/' + data_dir
                with open(read_dir) as f: 
                    data = json.load(f)
                
                knn_values = knn_vector(data)

                #### obtain for specific k value
                indices = knn_values[:,0] # get all the values in the first column (get all the k values as indices)
                knn_df = pd.DataFrame(knn_values,index = indices)
                fixed_k_knn_value = knn_df.loc[fixed_k][1] # obtains the k value and the AUROC for the k value which is the fixed k, then takes index 1 which is the AUROC value
                ###### obtain optimal value ########
                optimal_knn_AUROC = np.max(knn_values[:,1])

                data_index = dataset_dict[ID_dataset][OOD_dataset]
                #print(f'ID: {ID_dataset}, OOD {OOD_dataset}:{round(mahalanobis_AUROC,3)}:index {data_index}')
                data_array[data_index,0] = mahalanobis_AUROC 
                data_array[data_index,1] = fixed_k_knn_value
                data_array[data_index,2] = optimal_knn_AUROC
                data_array[data_index,3] = quadratic_auroc

                row_names[data_index] = f'ID:{ID_dataset}, OOD:{OOD_dataset}' 
                #row_names.append(f'ID:{ID_dataset}, OOD:{OOD_dataset}')

        column_names = ['Baseline', f'{fixed_k} NN', 'Optimal KNN', f'Quadratic {fixed_k} NN',]
        #column_names = ['Baseline', f'{fixed_k} NN', f'Quadratic {fixed_k} NN',]
        auroc_df = pd.DataFrame(data_array,columns = column_names, index=row_names)
            
        latex_table = auroc_df.to_latex()
        
        
        latex_table = latex_table.replace('{}','{Datasets}')
        latex_table = latex_table.replace("lrrr","|p{3cm}|c|c|c|")
        latex_table = post_process_latex_table(latex_table)
        
        

        # Replacing the string with the max value so it can be seen
        #https://stackoverflow.com/questions/4703390/how-to-extract-a-floating-number-from-a-string
        #https://www.youtube.com/watch?v=sa-TUpSx1JA  - explanation, \s means space and + means 1 or more, whilst \d means decimal and \. means a . (\) used in front as . by itself is a special character 
        #string = re.findall("&\s+\d+\.\d+\s+&\s+\d+\.\d+\s+&\s+\d+\.\d+", latex_table)
        string = re.findall("&\s+\d+\.\d+\s+&\s+\d+\.\d+\s+&\s+\d+\.\d+\s+&\s+\d+\.\d+", latex_table)
        updated_string = []
        for index in range(len(string)):
            numbers = re.findall("\d+\.\d+", string[index]) # fnd all the numbers in the substring (gets rid of the &)
            #max_number = max(numbers,key=lambda x:float(x))
            max_number = float(max(numbers,key=lambda x:float(x))) #  Need to get the output as a float
            
            #string[index] = string[index].replace(f'{max_number}',f'\textbf{ {max_number} }') # Need to put spaces around otherwise it just shows max number
            updated_string.append(string[index].replace(f'{max_number}',fr'\textbf{ {max_number} }')) # Need to put spaces around otherwise it just shows max number), also need to be place to make it so that \t does not act as space
            latex_table = latex_table.replace(f'{string[index]}',f'{updated_string[index]}') 

        print(latex_table)
            




# Calculates the AUROC for the different datasets, then calculates the differences in the AUROC values to perform a wilcoxon test (does not use the optimal KNN value)
def knn_auroc_wilcoxon():
    # Fixed value of k of interest
    fixed_k = 10
    # Desired ID,OOD and Model  
    root_dir = 'run_data/'

    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py
    #runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.model_type": "SupCon","config.epochs": 300, 'state':'finished'})
    runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.model_type": "SupCon","config.epochs": 300})

    summary_list, config_list, name_list = [], [], []
    
    key_dict = {'dataset':{'MNIST':0, 'FashionMNIST':1,'KMNIST':2, 'CIFAR10':3, 'CIFAR100':4,'Caltech101':5,'Caltech256':6,'TinyImageNet':7,'Cub200':8,'Dogs':9},
                'model_type':{'SupCon':0}}
    
    # Makes array for the ranking of each of the dataset, as well as an empty list which should aggregate the scores from the datasets together 
    num_ID = len(key_dict['dataset'])
    collated_rank_score = np.empty((num_ID+1,2)) # + 1 to take into account an additional wilcoxon score which takes into account the entire dataset
    collated_rank_score[:] = np.nan
    collated_difference_linear = [None] * num_ID # Make an empty list to take into account all the different values
    collated_difference_quadratic = [None] * num_ID
    dataset_row_names = [None] * (num_ID+1) # empty list to take into account all the different dataset as well as the aggregate


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

        quadratic_string = 'Normalized One Dim Class Quadratic Typicality KNN'.lower()
        quadratic_typicality_keys = [key for key, value in summary_list[i].items() if quadratic_string in key.lower()]
        # Make a data array, where the number values are equal to the number of OOD classes present in the datamodule dict or equal to the number of keys
        
        # number of OOd datasets for this particular ID dataset
        num_ood = len(dataset_dict[ID_dataset])
        # data array
        data_array = np.empty((num_ood,3))
        data_array[:] = np.nan

        '''
        # Differences between the measurements of a particular dataset
        dataset_difference_array = np.empty((num_ood,2))
        dataset_difference_array = np.nan
        '''
        
        # name for the different rows of a table
        # https://stackoverflow.com/questions/10712002/create-an-empty-list-in-python-with-certain-size
        row_names = [None] * num_ood # Make an empty list to take into account all the different values 
        # go through the different knn keys

        
        for key in knn_keys:
            OOD_dataset = ood_dataset_string(key, dataset_dict, ID_dataset)
            # Get ood dataset specific key
            quadratic_ood_dataset_specific_key = [key for key in quadratic_typicality_keys if OOD_dataset.lower() in key.lower()]

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
                mahalanobis_AUROC = round(summary_list[i][OOD_dataset_specific_mahalanobis_key[0]],3)

                quadratic_auroc = round(summary_list[i][quadratic_ood_dataset_specific_key[0]],3)
                print(f'ID: {ID_dataset}, OOD {OOD_dataset}:{round(mahalanobis_AUROC,3)}')
                data_dir = summary_list[i][key]['path']
                run_dir = root_dir + run_path
                read_dir = run_dir + '/' + data_dir
                with open(read_dir) as f: 
                    data = json.load(f)
                
                knn_values = knn_vector(data)

                #### obtain for specific k value
                indices = knn_values[:,0] # get all the values in the first column (get all the k values as indices)
                knn_df = pd.DataFrame(knn_values,index = indices)
                fixed_k_knn_value = knn_df.loc[fixed_k][1] # obtains the k value and the AUROC for the k value which is the fixed k, then takes index 1 which is the AUROC value
                ###### obtain optimal value ########
                optimal_knn_AUROC = np.max(knn_values[:,1])

                data_index = dataset_dict[ID_dataset][OOD_dataset]
                #print(f'ID: {ID_dataset}, OOD {OOD_dataset}:{round(mahalanobis_AUROC,3)}:index {data_index}')
                data_array[data_index,0] = mahalanobis_AUROC 
                data_array[data_index,1] = fixed_k_knn_value
                data_array[data_index,2] = quadratic_auroc

                #row_names[data_index] = f'ID:{ID_dataset}, OOD:{OOD_dataset}' 
                

        # Calculate the differences in the results for a particular dataset

        linear_difference = data_array[:,0] - data_array[:,1]
        quadratic_difference = data_array[:,0] - data_array[:,2]
        
        # calculate scores for a particular dataest
        stat_linear, p_linear  = wilcoxon(linear_difference)
        stat_quadratic, p_quadratic  = wilcoxon(quadratic_difference)

        # Calculate P values for a particular dataset
        collated_rank_score[key_dict['dataset'][ID_dataset],0] = p_linear
        collated_rank_score[key_dict['dataset'][ID_dataset],1] = p_quadratic

        # Collate values into an array
        collated_difference_linear[key_dict['dataset'][ID_dataset]] = linear_difference
        collated_difference_quadratic[key_dict['dataset'][ID_dataset]] = quadratic_difference
        
        # Name for the dataset
        dataset_row_names[key_dict['dataset'][ID_dataset]] = f'ID:{ID_dataset}'

        '''
        column_names = [f'Linear {fixed_k} NN', f'Quadratic {fixed_k} NN',]
        
        #column_names = ['Baseline', f'{fixed_k} NN', f'Quadratic {fixed_k} NN',]
        auroc_df = pd.DataFrame(data_array,columns = column_names, index=row_names)
            
        latex_table = auroc_df.to_latex()
        
        
        latex_table = latex_table.replace('{}','{Datasets}')
        latex_table = latex_table.replace("lrrr","|p{3cm}|c|c|c|")
        latex_table = post_process_latex_table(latex_table)
        #import ipdb; ipdb.set_trace()

        
        column_names = ['Baseline', f'{fixed_k} NN', 'Optimal KNN', f'Quadratic {fixed_k} NN',]
        #column_names = ['Baseline', f'{fixed_k} NN', f'Quadratic {fixed_k} NN',]
        auroc_df = pd.DataFrame(data_array,columns = column_names, index=row_names)
        '''

    # Calculate the score for the aggregated score
    collated_difference_linear = np.concatenate(collated_difference_linear)
    collated_difference_quadratic = np.concatenate(collated_difference_quadratic)

    stat_linear, p_linear  = wilcoxon(collated_difference_linear)
    stat_quadratic, p_quadratic  = wilcoxon(collated_difference_quadratic)
    
    collated_rank_score[-1,0] = p_linear
    collated_rank_score[-1,1] = p_quadratic

    dataset_row_names[-1] = 'ID:Aggregate'
    column_names = [f'Linear {fixed_k} NN', f'Quadratic {fixed_k} NN']
    
    # Post pr
    auroc_df = pd.DataFrame(collated_rank_score, columns = column_names, index=dataset_row_names)
    latex_table = auroc_df.to_latex()
    latex_table = latex_table.replace('{}','{Datasets}')
    latex_table = latex_table.replace("lrr","|p{3cm}|c|c|")
    latex_table = post_process_latex_table(latex_table)

    print(latex_table)
    #negative_sum = np.sum(collated_difference_quadratic[collated_difference_quadratic<0])
    #positive_sum = np.sum(collated_difference_quadratic[collated_difference_quadratic>0])
    #import ipdb; ipdb.set_trace()
    


# Includes the results for the maximum softmax probability, odin , mahalanaobis as well as linear and quadratic typicality
def knn_auroc_table_v3():
    # Fixed value of k of interest
    fixed_k = 10
    # Desired ID,OOD and Model  
    root_dir = 'run_data/'

    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py


    key_dict = {'dataset':{'MNIST':0, 'FashionMNIST':1,'KMNIST':2, 'CIFAR10':3, 'CIFAR100':4,'Caltech101':5,'Caltech256':6,'TinyImageNet':7,'Cub200':8,'Dogs':9},
                'model_type':{'CE':0,'SupCon':1}}
    
    all_ID = ['MNIST','FashionMNIST','KMNIST', 'CIFAR10','CIFAR100','Caltech101','Caltech256','TinyImageNet','Cub200','Dogs']
    for ID_dataset in all_ID: # Go through the different ID dataset                
        runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.epochs": 300, 'state':'finished',"config.dataset": f"{ID_dataset}","$or": [{"config.model_type":"SupCon" }, {"config.model_type": "CE"}]})
        summary_list, config_list= [], []
        # number of OOd datasets for this particular ID dataset
        num_ood = len(dataset_dict[ID_dataset])
        # data array
        data_array = np.empty((num_ood,5)) # 5 different measurements
        data_array[:] = np.nan
    
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

            model_type = config_list[i]['model_type']
            Model_name = 'SupCLR' if model_type=='SupCon' else model_type
            # Make a data array, where the number values are equal to the number of OOD classes present in the datamodule dict or equal to the number of keys

            # name for the different rows of a table
            #row_names = []
            # https://stackoverflow.com/questions/10712002/create-an-empty-list-in-python-with-certain-size
            row_names = [None] * num_ood # Make an empty list to take into account all the different values 
            # go through the different knn keys

            if Model_name =='SupCLR':
                desired_string = 'Different K Normalized One Dim Class Typicality KNN OOD'.lower()
                knn_keys = [key for key, value in summary_list[i].items() if desired_string in key.lower()]

                baseline_string_1, baseline_string_2 = 'Mahalanobis AUROC OOD'.lower(), 'Mahalanobis AUROC: instance vector'.lower() 
                baseline_mahalanobis_keys_1 = [key for key, value in summary_list[i].items() if baseline_string_1 in key.lower()]
                baseline_mahalanobis_keys_1 = [key for key in baseline_mahalanobis_keys_1 if 'relative' not in key.lower()]

                baseline_mahalanobis_keys_2 = [key for key, value in summary_list[i].items() if baseline_string_2 in key.lower()]
                baseline_mahalanobis_keys_2 = [key for key in baseline_mahalanobis_keys_2 if 'relative' not in key.lower()]
                baseline_mahalanobis_keys = [*baseline_mahalanobis_keys_1,*baseline_mahalanobis_keys_2] 

                quadratic_string = 'Normalized One Dim Class Quadratic Typicality KNN'.lower()
                quadratic_typicality_keys = [key for key, value in summary_list[i].items() if quadratic_string in key.lower()]

                for key in knn_keys:
                    OOD_dataset = ood_dataset_string(key, dataset_dict, ID_dataset)
                    if OOD_dataset is None:
                        pass
                    else:
                        # Get ood dataset specific key
                        quadratic_ood_dataset_specific_key = [key for key in quadratic_typicality_keys if OOD_dataset.lower() in key.lower()]

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
                            mahalanobis_AUROC = round(summary_list[i][OOD_dataset_specific_mahalanobis_key[0]],3)

                            quadratic_auroc = round(summary_list[i][quadratic_ood_dataset_specific_key[0]],3)
                            #print(f'ID: {ID_dataset}, OOD {OOD_dataset}:{round(mahalanobis_AUROC,3)}')
                            data_dir = summary_list[i][key]['path']
                            run_dir = root_dir + run_path
                            read_dir = run_dir + '/' + data_dir
                            with open(read_dir) as f: 
                                data = json.load(f)
                            knn_values = knn_vector(data)
                            #### obtain for specific k value
                            indices = knn_values[:,0] # get all the values in the first column (get all the k values as indices)
                            knn_df = pd.DataFrame(knn_values,index = indices)
                            fixed_k_knn_value = knn_df.loc[fixed_k][1] # obtains the k value and the AUROC for the k value which is the fixed k, then takes index 1 which is the AUROC value
                            ###### obtain optimal value ########
                            data_index = dataset_dict[ID_dataset][OOD_dataset]
                            #print(f'ID: {ID_dataset}, OOD {OOD_dataset}:{round(mahalanobis_AUROC,3)}:index {data_index}')
                            data_array[data_index,2] = mahalanobis_AUROC 
                            data_array[data_index,3] = fixed_k_knn_value
                            data_array[data_index,4] = quadratic_auroc
                            row_names[data_index] = f'ID:{ID_dataset}, OOD:{OOD_dataset}' 
                            #row_names.append(f'ID:{ID_dataset}, OOD:{OOD_dataset}')
            else:
                baseline_max_softmax_string = 'Maximum Softmax Probability'.lower()
                baseline_odin_string = 'ODIN'.lower()
                
                baseline_max_softmax_keys = [key for key, value in summary_list[i].items() if baseline_max_softmax_string in key.lower()]
                baseline_odin_keys = [key for key, value in summary_list[i].items() if baseline_odin_string in key.lower()]
                
                for key in baseline_max_softmax_keys:
                    OOD_dataset = ood_dataset_string(key, dataset_dict, ID_dataset)
                    if OOD_dataset is None:
                        pass
                    else:
                        # Get ood dataset specific key
                        #print('ID dataset:',ID_dataset)
                        #print('OOD dataset:',OOD_dataset)

                        baseline_max_softmax_ood_specific_key = [key for key in baseline_max_softmax_keys if OOD_dataset.lower() in key.lower()]
                        baseline_odin_ood_specific_key = [key for key in baseline_odin_keys if OOD_dataset.lower() in key.lower()]
                        # get the specific mahalanobis keys for the specific OOD dataset
                        max_softmax_AUROC = round(summary_list[i][baseline_max_softmax_ood_specific_key[0]],3)
                        odin_AUROC = round(summary_list[i][baseline_odin_ood_specific_key[0]],3)
                        data_index = dataset_dict[ID_dataset][OOD_dataset]
                        data_array[data_index,0] = max_softmax_AUROC 
                        data_array[data_index,1] = odin_AUROC
                        row_names[data_index] = f'ID:{ID_dataset}, OOD:{OOD_dataset}'
            
        column_names = ['Maximum Softmax','ODIN', 'Mahalanobis', f'{fixed_k} NN', f'Quadratic {fixed_k} NN',]
        #column_names = ['Baseline', f'{fixed_k} NN', f'Quadratic {fixed_k} NN',]
        auroc_df = pd.DataFrame(data_array,columns = column_names, index=row_names)
        caption = ID_dataset + ' Dataset'
        label = f'tab:{ID_dataset}_Dataset'
        latex_table = full_post_process_latex_table(auroc_df, caption, label)
        #latex_table = full_post_process_latex_table(auroc_df,ID_dataset)
        
        print(latex_table)
        
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

if __name__== '__main__':
    #knn_auroc_table()
    #knn_auroc_table_v2()
    knn_auroc_table_v3()
    #knn_auroc_wilcoxon()
    
    #knn_auroc_plot_v3()
    #knn_auroc_plot()