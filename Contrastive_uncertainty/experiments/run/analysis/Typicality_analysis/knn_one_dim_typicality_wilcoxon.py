# Plotting the AUROC as the value of K changes for the group typicality approach
# Also works on making tables for the ID and OOD datasets

from pytorch_lightning.utilities.parsing import is_picklable
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
import copy
from Contrastive_uncertainty.experiments.run.analysis.analysis_utils import dataset_dict, key_dict, ood_dataset_string
from Contrastive_uncertainty.experiments.run.analysis.analysis_utils import full_post_process_latex_table,\
    replace_headings, bold_max_value, initial_table_info, add_caption, add_label, end_table_info,join_columns,join_different_columns,\
    replace_headings_collated_table, post_process_latex_table   
from Contrastive_uncertainty.experiments.run.analysis.Typicality_analysis.knn_one_dim_typicality_diagrams import knn_vector 
from Contrastive_uncertainty.experiments.run.analysis.Typicality_analysis.knn_one_dim_typicality_tables import obtain_ood_datasets,obtain_knn_value, obtain_baseline_mahalanobis, obtain_baseline

# Performs one sided-test to see the importance
def knn_auroc_wilcoxon_v2():
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

                data_index = dataset_dict[ID_dataset][OOD_dataset]
                #print(f'ID: {ID_dataset}, OOD {OOD_dataset}:{round(mahalanobis_AUROC,3)}:index {data_index}')
                data_array[data_index,0] = mahalanobis_AUROC 
                data_array[data_index,1] = fixed_k_knn_value
                data_array[data_index,2] = quadratic_auroc

                #row_names[data_index] = f'ID:{ID_dataset}, OOD:{OOD_dataset}' 
                

        # Calculate the differences in the results for a particular dataset

        linear_difference =  data_array[:,0] - data_array[:,1]
        quadratic_difference = data_array[:,0] - data_array[:,2] 
        
        # print('baseline',data_array[:,0])
        # print('quadratic',data_array[:,2])
        # print('quadratic difference',quadratic_difference)
        
        # calculate scores for a particular dataest
        stat_linear, p_linear  = wilcoxon(linear_difference,alternative='less')
        stat_quadratic, p_quadratic  = wilcoxon(quadratic_difference,alternative='less')
        #stat_linear, p_linear  = wilcoxon(linear_difference)
        #stat_quadratic, p_quadratic  = wilcoxon(quadratic_difference)
        # Calculate P values for a particular dataset
        collated_rank_score[key_dict['dataset'][ID_dataset],0] = p_linear
        collated_rank_score[key_dict['dataset'][ID_dataset],1] = p_quadratic

        # Collate values into an array
        collated_difference_linear[key_dict['dataset'][ID_dataset]] = linear_difference
        collated_difference_quadratic[key_dict['dataset'][ID_dataset]] = quadratic_difference
        
        # Name for the dataset
        dataset_row_names[key_dict['dataset'][ID_dataset]] = f'ID:{ID_dataset}'

    # Calculate the score for the aggregated score
    collated_difference_linear = np.concatenate(collated_difference_linear)
    collated_difference_quadratic = np.concatenate(collated_difference_quadratic)

    # To confirm that the differences can assume to be negative (baseline - typicality is negative), we use alternative is less. The null hypothesis is that the median difference is positive whilst the alternative is that the median difference is negative.

    stat_linear, p_linear  = wilcoxon(collated_difference_linear,alternative='less')     
    stat_quadratic, p_quadratic  = wilcoxon(collated_difference_quadratic,alternative='less')
    #stat_linear, p_linear  = wilcoxon(collated_difference_linear)
    #stat_quadratic, p_quadratic  = wilcoxon(collated_difference_quadratic)
    collated_rank_score[-1,0] = p_linear
    collated_rank_score[-1,1] = p_quadratic

    dataset_row_names[-1] = 'ID:Aggregate'
    column_names = [f'Linear {fixed_k} NN', f'Quadratic {fixed_k} NN']
        
    # Post processing latex table
    caption =  'Wilcoxon Signed Rank test - P values'
    label = f'tab:Wilcoxon_test'
    auroc_df = pd.DataFrame(collated_rank_score, columns = column_names, index=dataset_row_names)
    latex_table = auroc_df.to_latex()
    
    latex_table = replace_headings(auroc_df,latex_table)
    latex_table = post_process_latex_table(latex_table)
    latex_table = initial_table_info(latex_table)
    latex_table = add_caption(latex_table,caption)
    latex_table = add_label(latex_table,label) 
    latex_table = end_table_info(latex_table)
    
    print(latex_table)


# Calculates wilcoxon for the CE baselines as well
def knn_auroc_wilcoxon_v3(baseline):
    # Fixed value of k of interest
    fixed_k = 10
    # Desired ID,OOD and Model  
    root_dir = 'run_data/'

    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py


    key_dict = {'dataset':{'MNIST':0, 'FashionMNIST':1,'KMNIST':2, 'CIFAR10':3, 'CIFAR100':4,'Caltech101':5,'Caltech256':6,'TinyImageNet':7,'Cub200':8,'Dogs':9},
                'model_type':{'CE':0,'SupCon':1}}

    # Makes array for the ranking of each of the dataset, as well as an empty list which should aggregate the scores from the datasets together 
    num_ID = len(key_dict['dataset'])
    collated_rank_score = np.empty((num_ID+1,2)) # + 1 to take into account an additional wilcoxon score which takes into account the entire dataset
    collated_rank_score[:] = np.nan
    collated_difference_linear = [None] * num_ID # Make an empty list to take into account all the different values
    collated_difference_quadratic = [None] * num_ID
    dataset_row_names = [None] * (num_ID+1) # empty list to take into account all the different dataset as well as the aggregate
    
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
            # https://stackoverflow.com/questions/10712002/create-an-empty-list-in-python-with-certain-size
            row_names = [None] * num_ood # Make an empty list to take into account all the different values 
            # go through the different knn keys

            if Model_name =='SupCLR':
                desired_string = 'Different K Normalized One Dim Class Typicality KNN OOD'.lower()
                knn_keys = [key for key, value in summary_list[i].items() if desired_string in key.lower()]

                quadratic_string = 'Normalized One Dim Class Quadratic Typicality KNN'.lower()
                quadratic_typicality_keys = [key for key, value in summary_list[i].items() if quadratic_string in key.lower()]

                for key in knn_keys:
                    OOD_dataset = ood_dataset_string(key, dataset_dict, ID_dataset)
                    if OOD_dataset is None:
                        pass
                    else:
                        # Get ood dataset specific key
                        quadratic_ood_dataset_specific_key = [key for key in quadratic_typicality_keys if OOD_dataset.lower() in key.lower()]
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
                        data_array[data_index,1] = fixed_k_knn_value
                        data_array[data_index,2] = quadratic_auroc
                        #row_names[data_index] = f'ID:{ID_dataset}, OOD:{OOD_dataset}' 
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
                        if baseline =='Maximum-Probability':
                            data_array[data_index,0] = max_softmax_AUROC
                        elif baseline == 'ODIN': 
                            data_array[data_index,1] = odin_AUROC
                        else:
                            print('incorrect baseline')


        # Calculate the differences in the results for a particular dataset

        linear_difference =  data_array[:,0] - data_array[:,1]
        quadratic_difference = data_array[:,0] - data_array[:,2] 
        
        # print('baseline',data_array[:,0])
        # print('quadratic',data_array[:,2])
        # print('quadratic difference',quadratic_difference)
        
        # calculate scores for a particular dataest
        stat_linear, p_linear  = wilcoxon(linear_difference,alternative='less')
        stat_quadratic, p_quadratic  = wilcoxon(quadratic_difference,alternative='less')
        #stat_linear, p_linear  = wilcoxon(linear_difference)
        #stat_quadratic, p_quadratic  = wilcoxon(quadratic_difference)
        # Calculate P values for a particular dataset
        collated_rank_score[key_dict['dataset'][ID_dataset],0] = p_linear
        collated_rank_score[key_dict['dataset'][ID_dataset],1] = p_quadratic

        # Collate values into an array
        collated_difference_linear[key_dict['dataset'][ID_dataset]] = linear_difference
        collated_difference_quadratic[key_dict['dataset'][ID_dataset]] = quadratic_difference
        
        # Name for the dataset
        dataset_row_names[key_dict['dataset'][ID_dataset]] = f'ID:{ID_dataset}'    

    # Calculate the score for the aggregated score
    collated_difference_linear = np.concatenate(collated_difference_linear)
    collated_difference_quadratic = np.concatenate(collated_difference_quadratic)

    # To confirm that the differences can assume to be negative (baseline - typicality is negative), we use alternative is less. The null hypothesis is that the median difference is positive whilst the alternative is that the median difference is negative.

    stat_linear, p_linear  = wilcoxon(collated_difference_linear,alternative='less')     
    stat_quadratic, p_quadratic  = wilcoxon(collated_difference_quadratic,alternative='less')
    
    collated_rank_score[-1,0] = p_linear
    collated_rank_score[-1,1] = p_quadratic

    dataset_row_names[-1] = 'ID:Aggregate'
    column_names = [f'Linear {fixed_k} NN', f'Quadratic {fixed_k} NN']
        
    # Post processing latex table
    caption =  f'{baseline} Wilcoxon Signed Rank test - P values'
    label = f'tab:{baseline}_Wilcoxon_test'
    auroc_df = pd.DataFrame(collated_rank_score, columns = column_names, index=dataset_row_names)
    latex_table = auroc_df.to_latex()
    
    latex_table = replace_headings(auroc_df,latex_table)
    latex_table = post_process_latex_table(latex_table)
    latex_table = initial_table_info(latex_table)
    latex_table = add_caption(latex_table,caption)
    latex_table = add_label(latex_table,label) 
    latex_table = end_table_info(latex_table)
    
    print(latex_table)

# Calculates wilcoxon for the quadratic being compared to all the other dataset
def knn_auroc_wilcoxon_v4():
    num_baselines = 4
    # Fixed value of k of interest
    fixed_k = 10
    # Desired ID,OOD and Model  
    root_dir = 'run_data/'

    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py


    key_dict = {'dataset':{'MNIST':0, 'FashionMNIST':1,'KMNIST':2, 'CIFAR10':3, 'CIFAR100':4,'Caltech101':5,'Caltech256':6,'TinyImageNet':7,'Cub200':8,'Dogs':9},
                'model_type':{'CE':0,'SupCon':1}}

    # Makes array for the ranking of each of the dataset, as well as an empty list which should aggregate the scores from the datasets together 
    num_ID = len(key_dict['dataset'])
    collated_rank_score = np.empty((num_ID+1,num_baselines)) # + 1 to take into account an additional wilcoxon score which takes into account the entire dataset
    collated_rank_score[:] = np.nan
    # https://thispointer.com/how-to-create-and-initialize-a-list-of-lists-in-python/ -  Important to initialise list effectively (with separate lists rather than pointing to the same list)
    collated_difference = [[None] * num_ID for i in range(num_baselines)] 
    dataset_row_names = [None] * (num_ID+1) # empty list to take into account all the different dataset as well as the aggregate
 
    
    all_ID = ['MNIST','FashionMNIST','KMNIST', 'CIFAR10','CIFAR100','Caltech101','Caltech256','TinyImageNet','Cub200','Dogs']
    for ID_dataset in all_ID: # Go through the different ID dataset                
        runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.epochs": 300, "config.dataset": f"{ID_dataset}","$or": [{"config.model_type":"SupCon" }, {"config.model_type": "CE"}]})
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
                        baseline_max_softmax_ood_specific_key = [key for key in baseline_max_softmax_keys if OOD_dataset.lower() in key.lower()]
                        baseline_odin_ood_specific_key = [key for key in baseline_odin_keys if OOD_dataset.lower() in key.lower()]
                        # get the specific mahalanobis keys for the specific OOD dataset
                        max_softmax_AUROC = round(summary_list[i][baseline_max_softmax_ood_specific_key[0]],3)
                        odin_AUROC = round(summary_list[i][baseline_odin_ood_specific_key[0]],3)
                        data_index = dataset_dict[ID_dataset][OOD_dataset]
                        data_array[data_index,0] = max_softmax_AUROC 
                        data_array[data_index,1] = odin_AUROC
                        row_names[data_index] = f'ID:{ID_dataset}, OOD:{OOD_dataset}'
        print(f'ID:{ID_dataset}')
        print('data array',data_array)
        for i in range(num_baselines):

            difference = data_array[:,i] - data_array[:,-1] # Calculate the differences in the results for a particular dataset
            stat, p_value  = wilcoxon(difference,alternative='less') # calculate scores for a particular dataest
            # Calculate P values for a particular dataset
            collated_rank_score[key_dict['dataset'][ID_dataset],i] = p_value
            # Place the difference for a particular baseline for a particular ID dataset
            collated_difference[i][key_dict['dataset'][ID_dataset]] = difference
        
        dataset_row_names[key_dict['dataset'][ID_dataset]] = f'ID:{ID_dataset}'   

    # Calculate the score for the aggregated score
    
    aggregated_difference = [np.concatenate(collated_difference[i]) for i in range(num_baselines)] # Make a list for the different baseline
    # To confirm that the differences can assume to be negative (baseline - typicality is negative), we use alternative is less. The null hypothesis is that the median difference is positive whilst the alternative is that the median difference is negative.
    # Go through all the differentn baselines, perform wilcoxon test and then place into rank score array
    for i in range(num_baselines):
        stat, p_value = wilcoxon(aggregated_difference[i],alternative='less')
        collated_rank_score[-1,i] = p_value
    
    dataset_row_names[-1] = 'ID:Aggregate'
    column_names = ['Maximum Softmax', 'ODIN', 'Mahalanobis',f'Linear {fixed_k} NN']
        
    # Post processing latex table
    caption =  f'Wilcoxon Signed Rank test - P values'
    label = f'tab:Wilcoxon_test'
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.set_option.html
    pd.set_option('display.precision',3) 

    auroc_df = pd.DataFrame(collated_rank_score, columns = column_names, index=dataset_row_names)
    latex_table = auroc_df.to_latex()
    
    latex_table = replace_headings(auroc_df,latex_table)
    latex_table = post_process_latex_table(latex_table)
    latex_table = initial_table_info(latex_table)
    latex_table = add_caption(latex_table,caption)
    latex_table = add_label(latex_table,label) 
    latex_table = end_table_info(latex_table)
    
    print(latex_table)
    
# Includes the softmax mahalanobis as an additional baseline
def knn_auroc_wilcoxon_v5():
    num_baselines = 5
    # Fixed value of k of interest
    fixed_k = 10
    # Desired ID,OOD and Model  
    root_dir = 'run_data/'

    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py


    key_dict = {'dataset':{'MNIST':0, 'FashionMNIST':1,'KMNIST':2, 'CIFAR10':3, 'CIFAR100':4,'Caltech101':5,'Caltech256':6,'TinyImageNet':7,'Cub200':8,'Dogs':9},
                'model_type':{'CE':0,'SupCon':1}}

    # Makes array for the ranking of each of the dataset, as well as an empty list which should aggregate the scores from the datasets together 
    num_ID = len(key_dict['dataset'])
    collated_rank_score = np.empty((num_ID+1,num_baselines)) # + 1 to take into account an additional wilcoxon score which takes into account the entire dataset
    collated_rank_score[:] = np.nan
    # https://thispointer.com/how-to-create-and-initialize-a-list-of-lists-in-python/ -  Important to initialise list effectively (with separate lists rather than pointing to the same list)
    collated_difference = [[None] * num_ID for i in range(num_baselines)] 
    dataset_row_names = [None] * (num_ID+1) # empty list to take into account all the different dataset as well as the aggregate
 
    
    all_ID = ['MNIST','FashionMNIST','KMNIST', 'CIFAR10','CIFAR100','Caltech101','Caltech256','TinyImageNet','Cub200','Dogs']
    for ID_dataset in all_ID: # Go through the different ID dataset                
        runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.epochs": 300, "config.dataset": f"{ID_dataset}","$or": [{"config.model_type":"SupCon" }, {"config.model_type": "CE"}]})
        summary_list, config_list= [], []
        # number of OOd datasets for this particular ID dataset
        num_ood = len(dataset_dict[ID_dataset])
        # data array
        data_array = np.empty((num_ood,num_baselines+1)) # 6 different measurements
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
                            data_array[data_index,3] = mahalanobis_AUROC 
                            data_array[data_index,4] = fixed_k_knn_value
                            data_array[data_index,5] = quadratic_auroc
                            row_names[data_index] = f'ID:{ID_dataset}, OOD:{OOD_dataset}' 
                            #row_names.append(f'ID:{ID_dataset}, OOD:{OOD_dataset}')
            else:
                softmax_mahalanobis_baseline_string = 'Mahalanobis AUROC OOD'.lower()
                softmax_mahalanobis_keys = [key for key, value in summary_list[i].items() if softmax_mahalanobis_baseline_string in key.lower()]

                baseline_max_softmax_string = 'Maximum Softmax Probability'.lower()
                baseline_odin_string = 'ODIN'.lower()
                
                baseline_max_softmax_keys = [key for key, value in summary_list[i].items() if baseline_max_softmax_string in key.lower()]
                baseline_odin_keys = [key for key, value in summary_list[i].items() if baseline_odin_string in key.lower()]
                
                for key in baseline_max_softmax_keys:
                    OOD_dataset = ood_dataset_string(key, dataset_dict, ID_dataset)
                    if OOD_dataset is None:
                        pass
                    else:
                        baseline_max_softmax_ood_specific_key = [key for key in baseline_max_softmax_keys if OOD_dataset.lower() in key.lower()]
                        baseline_odin_ood_specific_key = [key for key in baseline_odin_keys if OOD_dataset.lower() in key.lower()]
                        baseline_softmax_mahalanobis_ood_specific_key = [key for key in softmax_mahalanobis_keys if OOD_dataset.lower() in key.lower()] 
                        #OOD_dataset_softmax_mahalanobis_key = [key for key in softmax_mahalanobis_keys if OOD_dataset.lower() in key.lower()]
                
                        # get the specific mahalanobis keys for the specific OOD dataset
                        max_softmax_AUROC = round(summary_list[i][baseline_max_softmax_ood_specific_key[0]],3)
                        odin_AUROC = round(summary_list[i][baseline_odin_ood_specific_key[0]],3)

                        softmax_mahalanobis_AUROC = round(summary_list[i][baseline_softmax_mahalanobis_ood_specific_key[0]],3)
                        data_index = dataset_dict[ID_dataset][OOD_dataset]
                        data_array[data_index,0] = max_softmax_AUROC 
                        data_array[data_index,1] = odin_AUROC
                        data_array[data_index,2] = softmax_mahalanobis_AUROC
                        row_names[data_index] = f'ID:{ID_dataset}, OOD:{OOD_dataset}'

        print(f'ID:{ID_dataset}')
        #print('data array',data_array)
        x = data_array[:,[index for index in range(6) if index not in [0,1,3,4]]]
        print('softmax mahalanobis and quadratic',x)
        for i in range(num_baselines):

            difference = data_array[:,i] - data_array[:,-1] # Calculate the differences in the results for a particular dataset
            stat, p_value  = wilcoxon(difference,alternative='less') # calculate scores for a particular dataest
            # Calculate P values for a particular dataset
            collated_rank_score[key_dict['dataset'][ID_dataset],i] = p_value
            # Place the difference for a particular baseline for a particular ID dataset
            collated_difference[i][key_dict['dataset'][ID_dataset]] = difference
        
        dataset_row_names[key_dict['dataset'][ID_dataset]] = f'ID:{ID_dataset}'   

    # Calculate the score for the aggregated score
    
    aggregated_difference = [np.concatenate(collated_difference[i]) for i in range(num_baselines)] # Make a list for the different baseline
    # To confirm that the differences can assume to be negative (baseline - typicality is negative), we use alternative is less. The null hypothesis is that the median difference is positive whilst the alternative is that the median difference is negative.
    # Go through all the differentn baselines, perform wilcoxon test and then place into rank score array
    for i in range(num_baselines):
        stat, p_value = wilcoxon(aggregated_difference[i],alternative='less')
        collated_rank_score[-1,i] = p_value
    
    dataset_row_names[-1] = 'ID:Aggregate'
    column_names = ['Maximum Softmax', 'ODIN','Softmax Mahalanobis', 'Contrastive Mahalanobis',f'Linear {fixed_k} NN']
        
    # Post processing latex table
    caption =  f'Wilcoxon Signed Rank test - P values'
    label = f'tab:Wilcoxon_test'
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.set_option.html
    pd.set_option('display.precision',3) 

    auroc_df = pd.DataFrame(collated_rank_score, columns = column_names, index=dataset_row_names)
    latex_table = auroc_df.to_latex()
    
    latex_table = replace_headings(auroc_df,latex_table)
    latex_table = post_process_latex_table(latex_table)
    latex_table = initial_table_info(latex_table)
    latex_table = add_caption(latex_table,caption)
    latex_table = add_label(latex_table,label) 
    latex_table = end_table_info(latex_table)
    
    print(latex_table)

# Comparing mahalanobis to the quadratic typicality only but with updated code
def knn_auroc_wilcoxon_v6():
    num_baselines = 1
    # Desired ID,OOD and Model  
    root_dir = 'run_data/'

    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py


    key_dict = {'dataset':{'MNIST':0, 'FashionMNIST':1,'KMNIST':2, 'CIFAR10':3, 'CIFAR100':4,'Caltech101':5,'Caltech256':6,'TinyImageNet':7,'Cub200':8,'Dogs':9},
                'model_type':{'CE':0,'SupCon':1}}

    # Makes array for the ranking of each of the dataset, as well as an empty list which should aggregate the scores from the datasets together 
    num_ID = len(key_dict['dataset'])
    collated_rank_score = np.empty((num_ID+1,num_baselines)) # + 1 to take into account an additional wilcoxon score which takes into account the entire dataset
    collated_rank_score[:] = np.nan
    # https://thispointer.com/how-to-create-and-initialize-a-list-of-lists-in-python/ -  Important to initialise list effectively (with separate lists rather than pointing to the same list)
    collated_difference = [[None] * num_ID for i in range(num_baselines)] 
    dataset_row_names = [None] * (num_ID+1) # empty list to take into account all the different dataset as well as the aggregate
 
    
    all_ID = ['MNIST','FashionMNIST','KMNIST', 'CIFAR10','CIFAR100','Caltech101','Caltech256','TinyImageNet','Cub200','Dogs']
    for ID_dataset in all_ID: # Go through the different ID dataset                
        runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.epochs": 300, "config.dataset": f"{ID_dataset}","config.model_type":"SupCon"})
        # number of OOd datasets for this particular ID dataset
        num_ood = len(dataset_dict[ID_dataset])
        # data array
        data_array = np.empty((num_ood,num_baselines+1)) # 6 different measurements
        data_array[:] = np.nan
    
        for i, run in enumerate(runs): 
            # .summary contains the output keys/values for metrics like accuracy.
            #  We call ._json_dict to omit large files 

            run_summary = run.summary._json_dict
            # .config contains the hyperparameters.
            #  We remove special values that start with _.
            run_config = {k: v for k,v in run.config.items()
                 if not k.startswith('_')} 

            group_name = run_config['group']
            path_list = run.path
            # include the group name in the run path
            path_list.insert(-1, group_name)
            run_path = '/'.join(path_list)

            model_type = run_config['model_type']
            Model_name = 'SupCLR' if model_type=='SupCon' else model_type
            # Make a data array, where the number values are equal to the number of OOD classes present in the datamodule dict or equal to the number of keys

            # name for the different rows of a table
            # https://stackoverflow.com/questions/10712002/create-an-empty-list-in-python-with-certain-size
            row_names = [None] * num_ood # Make an empty list to take into account all the different values 
            # go through the different knn keys
            desired_string = 'Normalized One Dim Class Quadratic Typicality KNN'.lower()

            # Obtain all the OOD datasets for a particular desired string
            all_OOD_datasets = obtain_ood_datasets(desired_string, run_summary,ID_dataset)
            for OOD_dataset in all_OOD_datasets:

                quadratic_auroc = obtain_knn_value(desired_string,run_summary,OOD_dataset)
                mahalanobis_auroc = obtain_baseline_mahalanobis(run_summary,OOD_dataset)
                data_index = dataset_dict[ID_dataset][OOD_dataset]
                data_array[data_index,0] = mahalanobis_auroc 
                data_array[data_index,1] = quadratic_auroc
                
                row_names[data_index] = f'ID:{ID_dataset}, OOD:{OOD_dataset}' 
            

        print(f'ID:{ID_dataset}')
        for i in range(num_baselines):
            difference = data_array[:,i] - data_array[:,-1] # Calculate the differences in the results for a particular dataset
            stat, p_value  = wilcoxon(difference,alternative='less') # calculate scores for a particular dataest
            # Calculate P values for a particular dataset
            collated_rank_score[key_dict['dataset'][ID_dataset],i] = p_value
            # Place the difference for a particular baseline for a particular ID dataset
            collated_difference[i][key_dict['dataset'][ID_dataset]] = difference
        
        dataset_row_names[key_dict['dataset'][ID_dataset]] = f'ID:{ID_dataset}'   

    # Calculate the score for the aggregated score
    
    aggregated_difference = [np.concatenate(collated_difference[i]) for i in range(num_baselines)] # Make a list for the different baseline
    # To confirm that the differences can assume to be negative (baseline - typicality is negative), we use alternative is less. The null hypothesis is that the median difference is positive whilst the alternative is that the median difference is negative.
    # Go through all the differentn baselines, perform wilcoxon test and then place into rank score array
    for i in range(num_baselines):
        stat, p_value = wilcoxon(aggregated_difference[i],alternative='less')
        collated_rank_score[-1,i] = p_value
    
    dataset_row_names[-1] = 'ID:Aggregate'
    column_names = ['Contrastive Mahalanobis']
        
    # Post processing latex table
    caption =  f'Wilcoxon Signed Rank test - P values'
    label = f'tab:Wilcoxon_test'
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.set_option.html
    pd.set_option('display.precision',3) 

    auroc_df = pd.DataFrame(collated_rank_score, columns = column_names, index=dataset_row_names)
    latex_table = auroc_df.to_latex()
    
    latex_table = replace_headings(auroc_df,latex_table)
    latex_table = post_process_latex_table(latex_table)
    latex_table = initial_table_info(latex_table)
    latex_table = add_caption(latex_table,caption)
    latex_table = add_label(latex_table,label) 
    latex_table = end_table_info(latex_table)
    
    print(latex_table)

# Calcualate the P value when a particular OOD dataset is fixed
def knn_auroc_wilcoxon_ood():

    reverse_dataset_dict = {'MNIST':[], 'FashionMNIST':[],'KMNIST':[], 'CIFAR10':[], 'CIFAR100':[],'Caltech101':[],'Caltech256':[],'TinyImageNet':[],'Cub200':[],'Dogs':[]} # Used to get the OOD dataset
    # Invert the dataset dict in order to get index and key for particular OOD datasets in reverse
    for k in dataset_dict: # go through the different original dicts
        reverse_dataset_dict[k] = {v: k for k, v in dataset_dict[k].items()}

    num_baselines = 5
    # Fixed value of k of interest
    fixed_k = 10
    # Desired ID,OOD and Model  
    root_dir = 'run_data/'

    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py

    key_dict = {'dataset':{'MNIST':0, 'FashionMNIST':1,'KMNIST':2, 'CIFAR10':3, 'CIFAR100':4,'Caltech101':5,'Caltech256':6,'TinyImageNet':7,'Cub200':8,'Dogs':9},
                'model_type':{'CE':0,'SupCon':1}}

    # Makes array for the ranking of each of the dataset, as well as an empty list which should aggregate the scores from the datasets together 

    # https://thispointer.com/how-to-create-and-initialize-a-list-of-lists-in-python/ -  Important to initialise list effectively (with separate lists rather than pointing to the same list)
    dataset_row_names = []# [None] * (num_ID+1) # empty list to take into account all the different dataset as well as the aggregate
 
    
    all_ID = ['MNIST','FashionMNIST','KMNIST', 'CIFAR10','CIFAR100','Caltech101','Caltech256','TinyImageNet','Cub200','Dogs']
    all_OOD = {'MNIST':[],'FashionMNIST':[],'KMNIST':[],'EMNIST':[],'CelebA':[],'WIDERFace':[],'VOC':[], 'Places365':[],'STL10':[],'CIFAR10':[],'CIFAR100':[],'SVHN':[],'Caltech101':[],'Caltech256':[],'TinyImageNet':[],'Cub200':[],'Dogs':[]}
    
    collated_rank_score = np.empty((len(all_OOD),num_baselines)) # + 1 to take into account an additional wilcoxon score which takes into account the entire dataset
    collated_rank_score[:] = np.nan

    for ID_dataset in all_ID: # Go through the different ID dataset                
        runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.epochs": 300, "config.dataset": f"{ID_dataset}","$or": [{"config.model_type":"SupCon" }, {"config.model_type": "CE"}]})
        summary_list, config_list= [], []
        # number of OOd datasets for this particular ID dataset
        num_ood = len(dataset_dict[ID_dataset])
        # data array
        data_array = np.empty((num_ood,num_baselines+1)) # 6 different measurements
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
                            data_array[data_index,3] = mahalanobis_AUROC 
                            data_array[data_index,4] = fixed_k_knn_value
                            data_array[data_index,5] = quadratic_auroc
                            row_names[data_index] = f'ID:{ID_dataset}, OOD:{OOD_dataset}' 
                            #row_names.append(f'ID:{ID_dataset}, OOD:{OOD_dataset}')
            else:
                softmax_mahalanobis_baseline_string = 'Mahalanobis AUROC OOD'.lower()
                softmax_mahalanobis_keys = [key for key, value in summary_list[i].items() if softmax_mahalanobis_baseline_string in key.lower()]

                baseline_max_softmax_string = 'Maximum Softmax Probability'.lower()
                baseline_odin_string = 'ODIN'.lower()
                
                baseline_max_softmax_keys = [key for key, value in summary_list[i].items() if baseline_max_softmax_string in key.lower()]
                baseline_odin_keys = [key for key, value in summary_list[i].items() if baseline_odin_string in key.lower()]
                
                for key in baseline_max_softmax_keys:
                    OOD_dataset = ood_dataset_string(key, dataset_dict, ID_dataset)
                    if OOD_dataset is None:
                        pass
                    else:
                        baseline_max_softmax_ood_specific_key = [key for key in baseline_max_softmax_keys if OOD_dataset.lower() in key.lower()]
                        baseline_odin_ood_specific_key = [key for key in baseline_odin_keys if OOD_dataset.lower() in key.lower()]
                        baseline_softmax_mahalanobis_ood_specific_key = [key for key in softmax_mahalanobis_keys if OOD_dataset.lower() in key.lower()] 
                        #OOD_dataset_softmax_mahalanobis_key = [key for key in softmax_mahalanobis_keys if OOD_dataset.lower() in key.lower()]
                
                        # get the specific mahalanobis keys for the specific OOD dataset
                        max_softmax_AUROC = round(summary_list[i][baseline_max_softmax_ood_specific_key[0]],3)
                        odin_AUROC = round(summary_list[i][baseline_odin_ood_specific_key[0]],3)

                        softmax_mahalanobis_AUROC = round(summary_list[i][baseline_softmax_mahalanobis_ood_specific_key[0]],3)
                        data_index = dataset_dict[ID_dataset][OOD_dataset]
                        data_array[data_index,0] = max_softmax_AUROC 
                        data_array[data_index,1] = odin_AUROC
                        data_array[data_index,2] = softmax_mahalanobis_AUROC
                        row_names[data_index] = f'ID:{ID_dataset}, OOD:{OOD_dataset}'
        
        
        # Place in the array
        for i in range(len(data_array)):
            OOD_difference = data_array[i,:-1] - data_array[i,-1] # subtract the quadtract score from the remianing baselines
            OOD_data_value = np.expand_dims(OOD_difference,axis=0)

            OOD = reverse_dataset_dict[ID_dataset][i] # Get the particualr OOD dataset
            all_OOD[OOD].append(OOD_data_value) # place the differences in the array 

    # Goes through all the different OOD datasets and collates the different measurements           
    for i, key in enumerate(all_OOD): # go through the different OOD datasets
        collated_OOD_differences = np.concatenate(all_OOD[key],axis=0) # Differnces for a particualr OOD dataset
        dataset_row_names.append(f'OOD:{key}') # Add the OOD dataset name
        
        for j in range(num_baselines):
            stat, p_value  = wilcoxon(collated_OOD_differences[:,j],alternative='less') # calculate scores for a particular OOD dataset
            collated_rank_score[i,j] = p_value
    

    column_names = ['Maximum Softmax', 'ODIN','Softmax Mahalanobis', 'Contrastive Mahalanobis',f'Linear {fixed_k} NN']
        
    # Post processing latex table
    caption =  f'Wilcoxon Signed Rank test - P values'
    label = f'tab:Wilcoxon_test'
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.set_option.html
    pd.set_option('display.precision',3) 

    auroc_df = pd.DataFrame(collated_rank_score, columns = column_names, index=dataset_row_names)
    latex_table = auroc_df.to_latex()
    
    latex_table = replace_headings(auroc_df,latex_table)
    latex_table = post_process_latex_table(latex_table)
    latex_table = initial_table_info(latex_table)
    latex_table = add_caption(latex_table,caption)
    latex_table = add_label(latex_table,label) 
    latex_table = end_table_info(latex_table)
    
    print(latex_table)




# Performing the wilcoxon ranked test in the case where there are repeat measurments
def knn_auroc_wilcoxon_repeated_runs_v1():
    num_baselines = 1
    # Desired ID,OOD and Model  
    root_dir = 'run_data/'

    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py


    key_dict = {'dataset':{'MNIST':0, 'FashionMNIST':1,'KMNIST':2, 'CIFAR10':3, 'CIFAR100':4,'Caltech101':5,'Caltech256':6,'TinyImageNet':7,'Cub200':8,'Dogs':9},
                'model_type':{'SupCon':0}}

    # Makes array for the ranking of each of the dataset, as well as an empty list which should aggregate the scores from the datasets together 
    '''
    num_ID = len(key_dict['dataset'])
    collated_rank_score = np.empty((num_ID+1,num_baselines)) # + 1 to take into account an additional wilcoxon score which takes into account the entire dataset
    collated_rank_score[:] = np.nan
    # https://thispointer.com/how-to-create-and-initialize-a-list-of-lists-in-python/ -  Important to initialise list effectively (with separate lists rather than pointing to the same list)
    collated_difference = [[None] * num_ID for i in range(num_baselines)] 
    dataset_row_names = [None] * (num_ID+1) # empty list to take into account all the different dataset as well as the aggregate
     '''

    all_ID = ['MNIST','FashionMNIST','KMNIST', 'CIFAR10','CIFAR100','Caltech101','Caltech256','TinyImageNet','Cub200','Dogs']
    for ID_dataset in all_ID: # Go through the different ID dataset                
        runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats","config.epochs": 300, "config.dataset": f"{ID_dataset}","config.model_type":"SupCon"})
        # number of OOd datasets for this particular ID dataset
        num_ood = len(dataset_dict[ID_dataset])
        # data array for each ID dataset 
        baseline_array = [[] for i in range(num_ood)] # Make an empty list to take into account the values for the baseline
        quadratic_array = [[] for i in range(num_ood)]

        # Get scores for each of the OOD datasets for this ID dataset
        collated_rank_score = np.empty((num_ood,num_baselines)) # + 1 to take into account an additional wilcoxon score which takes into account the entire dataset
        collated_rank_score[:] = np.nan
        
    
        for i, run in enumerate(runs): # go through the runs for a particular ID dataset
            # .summary contains the output keys/values for metrics like accuracy.
            #  We call ._json_dict to omit large files 

            run_summary = run.summary._json_dict
            # .config contains the hyperparameters.
            #  We remove special values that start with _.
            run_config = {k: v for k,v in run.config.items()
                 if not k.startswith('_')} 

            group_name = run_config['group']
            path_list = run.path
            # include the group name in the run path
            path_list.insert(-1, group_name)
            run_path = '/'.join(path_list)

            model_type = run_config['model_type']
            Model_name = 'SupCLR' if model_type=='SupCon' else model_type
            # Make a data array, where the number values are equal to the number of OOD classes present in the datamodule dict or equal to the number of keys

            # name for the different rows of a table
            # https://stackoverflow.com/questions/10712002/create-an-empty-list-in-python-with-certain-size
            row_names = [None] * num_ood # Make an empty list to take into account all the different values 
            # go through the different knn keys
            desired_string = 'Normalized One Dim Class Quadratic Typicality KNN - 10 OOD'.lower()

            # Obtain all the OOD datasets for a particular desired string
            all_OOD_datasets = obtain_ood_datasets(desired_string, run_summary,ID_dataset)
            for OOD_dataset in all_OOD_datasets:

                quadratic_auroc = obtain_knn_value(desired_string,run_summary,OOD_dataset)
                mahalanobis_auroc = obtain_baseline_mahalanobis(run_summary,OOD_dataset)
                data_index = dataset_dict[ID_dataset][OOD_dataset] # obtain an index for a particular ID and OOD dataset pair
                baseline_array[data_index].append(mahalanobis_auroc)
                quadratic_array[data_index].append(quadratic_auroc)
                
                
                row_names[data_index] = f'ID:{ID_dataset}, OOD:{OOD_dataset}' 
            
        print(f'ID:{ID_dataset}')
        
        difference = np.array(baseline_array) - np.array(quadratic_array) # shape (num ood, repeats)
        # Calculate the p values for a particular OOD dataset for this ID dataset
        for i in range(len(difference)): # go through all the different ID OOD dataset pairs
            stat, p_value  = wilcoxon(difference[i],alternative='less') # calculate the p value for a particular ID OOD dataset pair
            #stat, p_value  = wilcoxon(difference[i]) # calculate the p value for a particular ID OOD dataset pair
            collated_rank_score[i] = p_value # add the p_value to the rank score for this particular dataset

        
        column_names = ['Contrastive Mahalanobis']

        # Post processing latex table
        caption =  f'Wilcoxon Signed Rank test - {ID_dataset} P values'
        label = f'tab:Wilcoxon_test_{ID_dataset}'
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.set_option.html
        pd.set_option('display.precision',3) 

        auroc_df = pd.DataFrame(collated_rank_score, columns = column_names, index=row_names)
        latex_table = auroc_df.to_latex()

        latex_table = replace_headings(auroc_df,latex_table)
        latex_table = post_process_latex_table(latex_table)
        latex_table = initial_table_info(latex_table)
        latex_table = add_caption(latex_table,caption)
        latex_table = add_label(latex_table,label) 
        latex_table = end_table_info(latex_table)

        print(latex_table)


# Calculates repeated runs as before but also calculates for AUROC and FPR
def knn_auroc_wilcoxon_repeated_runs_v2():
    desired_string_AUROC = 'Normalized One Dim Class Quadratic Typicality KNN - 10 OOD'.lower() # Only get the key for the AUROC
    desired_string_AUPR= 'Normalized One Dim Class Quadratic Typicality KNN - 10 AUPR'.lower()
    desired_string_FPR = 'Normalized One Dim Class Quadratic Typicality KNN - 10 FPR'.lower()

    baseline_string_1_AUROC,baseline_string_2_AUROC = 'Mahalanobis AUROC OOD'.lower(),'Mahalanobis AUROC OOD'.lower()
    baseline_string_1_AUPR,baseline_string_2_AUPR = 'Mahalanobis AUPR'.lower(),'Mahalanobis AUPR'.lower()
    baseline_string_1_FPR,baseline_string_2_FPR = 'Mahalanobis FPR'.lower(),'Mahalanobis FPR'.lower()

    num_baselines = 1
    # Desired ID,OOD and Model  
    root_dir = 'run_data/'

    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py

    key_dict = {'dataset':{'MNIST':0, 'FashionMNIST':1,'KMNIST':2, 'CIFAR10':3, 'CIFAR100':4,'Caltech101':5,'Caltech256':6,'TinyImageNet':7,'Cub200':8,'Dogs':9},
                'model_type':{'SupCon':0}}

    all_ID = ['MNIST','FashionMNIST','KMNIST', 'CIFAR10','CIFAR100','Caltech101','Caltech256','TinyImageNet','Cub200','Dogs']
    for ID_dataset in all_ID: # Go through the different ID dataset                
        runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats","config.epochs": 300, "config.dataset": f"{ID_dataset}","config.model_type":"SupCon"})
        # number of OOd datasets for this particular ID dataset
        num_ood = len(dataset_dict[ID_dataset])
        # data array for each ID dataset 
        baseline_AUROC_values,baseline_AUPR_values, baseline_FPR_values = [[] for i in range(num_ood)], [[] for i in range(num_ood)], [[] for i in range(num_ood)] # Make an empty list to take into account the values for the baseline
        quadratic_AUROC_values, quadratic_AUPR_values, quadratic_FPR_values = [[] for i in range(num_ood)], [[] for i in range(num_ood)], [[] for i in range(num_ood)]

        # Get scores for each of the OOD datasets for this ID dataset
        collated_rank_score_auroc = np.empty((num_ood,num_baselines)) # + 1 to take into account an additional wilcoxon score which takes into account the entire dataset
        collated_rank_score_auroc[:] = np.nan
        
        collated_rank_score_aupr = np.empty((num_ood,num_baselines)) # + 1 to take into account an additional wilcoxon score which takes into account the entire dataset
        collated_rank_score_aupr[:] = np.nan

        collated_rank_score_fpr = np.empty((num_ood,num_baselines)) # + 1 to take into account an additional wilcoxon score which takes into account the entire dataset
        collated_rank_score_fpr[:] = np.nan
    
        for i, run in enumerate(runs): # go through the runs for a particular ID dataset
            # .summary contains the output keys/values for metrics like accuracy.
            #  We call ._json_dict to omit large files 

            run_summary = run.summary._json_dict
            # .config contains the hyperparameters.
            #  We remove special values that start with _.
            run_config = {k: v for k,v in run.config.items()
                 if not k.startswith('_')} 

            group_name = run_config['group']
            path_list = run.path
            # include the group name in the run path
            path_list.insert(-1, group_name)
            run_path = '/'.join(path_list)

            model_type = run_config['model_type']
            Model_name = 'SupCLR' if model_type=='SupCon' else model_type
            # Make a data array, where the number values are equal to the number of OOD classes present in the datamodule dict or equal to the number of keys

            # name for the different rows of a table
            # https://stackoverflow.com/questions/10712002/create-an-empty-list-in-python-with-certain-size
            row_names = [None] * num_ood # Make an empty list to take into account all the different values 
            # go through the different knn keys

            # Obtain all the OOD datasets for a particular desired string
            all_OOD_datasets = obtain_ood_datasets(desired_string_AUROC, run_summary,ID_dataset)
            for OOD_dataset in all_OOD_datasets:

                quadratic_auroc = obtain_knn_value(desired_string_AUROC,run_summary,OOD_dataset)
                mahalanobis_auroc = obtain_baseline_mahalanobis(run_summary,OOD_dataset,baseline_string_1_AUROC,baseline_string_2_AUROC)
                
                quadratic_aupr = obtain_knn_value(desired_string_AUPR,run_summary,OOD_dataset)
                mahalanobis_aupr = obtain_baseline_mahalanobis(run_summary,OOD_dataset,baseline_string_1_AUPR,baseline_string_2_AUPR)

                quadratic_fpr = obtain_knn_value(desired_string_FPR,run_summary,OOD_dataset)
                mahalanobis_fpr = obtain_baseline_mahalanobis(run_summary,OOD_dataset,baseline_string_1_FPR,baseline_string_2_FPR)

                data_index = dataset_dict[ID_dataset][OOD_dataset] # obtain an index for a particular ID and OOD dataset pair
                baseline_AUROC_values[data_index].append(mahalanobis_auroc)
                quadratic_AUROC_values[data_index].append(quadratic_auroc)

                baseline_AUPR_values[data_index].append(mahalanobis_aupr)
                quadratic_AUPR_values[data_index].append(quadratic_aupr)

                baseline_FPR_values[data_index].append(mahalanobis_fpr)
                quadratic_FPR_values[data_index].append(quadratic_fpr)
                
                row_names[data_index] = f'ID:{ID_dataset}, OOD:{OOD_dataset}' 
        
                
        print(f'ID:{ID_dataset}')
        
        difference_auroc = np.array(baseline_AUROC_values) - np.array(quadratic_AUROC_values) # shape (num ood, repeats)
        difference_aupr = np.array(baseline_AUPR_values) - np.array(quadratic_AUPR_values) # shape (num ood, repeats)
        # REVERSED THE DIRECTION FOR FPR DUE TO LOWER BEING BETTER FOR FPR, SO I DO NOT NEED TO REVERSE THE DIRECTION OF THE TEST STATISTIC
        difference_fpr = np.array(quadratic_FPR_values) - np.array(baseline_FPR_values) # shape (num ood, repeats)

        # Calculate the p values for a particular OOD dataset for this ID dataset
        for i in range(len(difference_auroc)): # go through all the different ID OOD dataset pairs
            #stat, p_value  = wilcoxon(difference[i],alternative='less') # calculate the p value for a particular ID OOD dataset pair
            stat, p_value_auroc  = wilcoxon(difference_auroc[i],alternative='less') # calculate the p value for a particular ID OOD dataset pair
            stat, p_value_aupr  = wilcoxon(difference_aupr[i],alternative='less') # calculate the p value for a particular ID OOD dataset pair
            stat, p_value_fpr  = wilcoxon(difference_fpr[i],alternative='less') # calculate the p value for a particular ID OOD dataset pair
            
            collated_rank_score_auroc[i] = p_value_auroc # add the p_value to the rank score for this particular dataset
            collated_rank_score_aupr[i] = p_value_aupr # add the p_value to the rank score for this particular dataset
            collated_rank_score_fpr[i] = p_value_fpr # add the p_value to the rank score for this particular dataset

        
        column_names_AUROC = ['AUROC']
        column_names_AUPR = ['AUPR']
        column_names_FPR = ['FPR']

        # Post processing latex table
        caption =  f'Wilcoxon Signed Rank test - {ID_dataset} P values'
        label = f'tab:Wilcoxon_test_{ID_dataset}'
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.set_option.html
        pd.set_option('display.precision',3) 

        auroc_df = pd.DataFrame(collated_rank_score_auroc,columns = column_names_AUROC, index=row_names)
        aupr_df = pd.DataFrame(collated_rank_score_aupr,columns = column_names_AUPR, index=row_names)
        fpr_df = pd.DataFrame(collated_rank_score_fpr,columns = column_names_FPR, index=row_names)
        latex_table = collated_wilcoxon_post_process_latex_table(auroc_df,aupr_df,fpr_df,caption,label)
        
        '''
        latex_table = replace_headings(auroc_df,latex_table)
        latex_table = post_process_latex_table(latex_table)
        latex_table = initial_table_info(latex_table)
        latex_table = add_caption(latex_table,caption)
        latex_table = add_label(latex_table,label) 
        latex_table = end_table_info(latex_table)
        '''
        
        
        print(latex_table)
        

def knn_auroc_wilcoxon_repeated_runs_v3(desired_approach = 'Quadratic_typicality', desired_model_type = 'SupCon', baseline_approach = 'Mahalanobis', baseline_model_type = 'CE'):
    if desired_approach == 'Quadratic_typicality':
        desired_string_AUROC = 'Normalized One Dim Class Quadratic Typicality KNN - 10 OOD'.lower() # Only get the key for the AUROC
        desired_string_AUPR= 'Normalized One Dim Class Quadratic Typicality KNN - 10 AUPR'.lower()
        desired_string_FPR = 'Normalized One Dim Class Quadratic Typicality KNN - 10 FPR'.lower()
        desired_function = obtain_knn_value # Used to calculate the value

    if baseline_approach =='Mahalanobis':
        baseline_string_AUROC = 'Mahalanobis AUROC OOD'.lower()
        baseline_string_AUPR = 'Mahalanobis AUPR'.lower()
        baseline_string_FPR = 'Mahalanobis FPR'.lower()
    elif baseline_approach =='Softmax':        
        baseline_string_AUROC = 'Maximum Softmax Probability AUROC OOD'.lower()
        baseline_string_AUPR = 'Maximum Softmax Probability AUPR OOD'.lower()
        baseline_string_FPR = 'Maximum Softmax Probability FPR OOD'.lower()
    else:
        assert baseline_approach == 'Mahalanobis' or baseline_approach =='Softmax', 'No other baselines implemented'
    
    baseline_function = obtain_baseline 
    num_baselines = 1
    # Desired ID,OOD and Model  
    root_dir = 'run_data/'

    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py


    all_ID = ['MNIST','FashionMNIST','KMNIST', 'CIFAR10','CIFAR100','Caltech101','Caltech256','TinyImageNet','Cub200','Dogs']
    for ID_dataset in all_ID: # Go through the different ID dataset                
        #runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats","config.epochs": 300, "config.dataset": f"{ID_dataset}","config.model_type":"SupCon"})
        runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats","config.epochs": 300, 'state':'finished',"config.dataset": f"{ID_dataset}","$or": [{"config.model_type":"SupCon" }, {"config.model_type": "CE"}]})
        # number of OOd datasets for this particular ID dataset
        num_ood = len(dataset_dict[ID_dataset])
        # data array for each ID dataset 
        baseline_AUROC_values,baseline_AUPR_values, baseline_FPR_values = [[] for i in range(num_ood)], [[] for i in range(num_ood)], [[] for i in range(num_ood)] # Make an empty list to take into account the values for the baseline
        desired_AUROC_values, desired_AUPR_values, desired_FPR_values = [[] for i in range(num_ood)], [[] for i in range(num_ood)], [[] for i in range(num_ood)]

        # Get scores for each of the OOD datasets for this ID dataset
        collated_rank_score_auroc = np.empty((num_ood,num_baselines)) # + 1 to take into account an additional wilcoxon score which takes into account the entire dataset
        collated_rank_score_auroc[:] = np.nan
        
        collated_rank_score_aupr = np.empty((num_ood,num_baselines)) # + 1 to take into account an additional wilcoxon score which takes into account the entire dataset
        collated_rank_score_aupr[:] = np.nan

        collated_rank_score_fpr = np.empty((num_ood,num_baselines)) # + 1 to take into account an additional wilcoxon score which takes into account the entire dataset
        collated_rank_score_fpr[:] = np.nan
        

        for i, run in enumerate(runs): # go through the runs for a particular ID dataset
            
            # .summary contains the output keys/values for metrics like accuracy.
            #  We call ._json_dict to omit large files 

            run_summary = run.summary._json_dict
            # .config contains the hyperparameters.
            #  We remove special values that start with _.
            run_config = {k: v for k,v in run.config.items()
                 if not k.startswith('_')} 
            
            group_name = run_config['group']
            path_list = run.path
            # include the group name in the run path
            path_list.insert(-1, group_name)
            run_path = '/'.join(path_list)

            model_type = run_config['model_type']
            Model_name = 'SupCLR' if model_type=='SupCon' else model_type
            # Make a data array, where the number values are equal to the number of OOD classes present in the datamodule dict or equal to the number of keys

            # name for the different rows of a table
            # https://stackoverflow.com/questions/10712002/create-an-empty-list-in-python-with-certain-size
            row_names = [None] * num_ood # Make an empty list to take into account all the different values 
            # go through the different knn keys

            # Obtain all the OOD datasets for a particular desired string
            all_OOD_datasets = obtain_ood_datasets(desired_string_AUROC, run_summary,ID_dataset)
            for OOD_dataset in all_OOD_datasets:
                data_index = dataset_dict[ID_dataset][OOD_dataset] # obtain an index for a particular ID and OOD dataset pair
                row_names[data_index] = f'ID:{ID_dataset}, OOD:{OOD_dataset}' 
                
                desired_AUROC_values = update_metric_list(desired_AUROC_values,data_index,desired_function,desired_string_AUROC,desired_model_type,model_type,run_summary,OOD_dataset,run_config['seed'])
                desired_AUPR_values = update_metric_list(desired_AUPR_values,data_index,desired_function,desired_string_AUPR,desired_model_type, model_type,run_summary,OOD_dataset,run_config['seed'])
                desired_FPR_values = update_metric_list(desired_FPR_values,data_index,desired_function,desired_string_FPR,desired_model_type, model_type,run_summary,OOD_dataset,run_config['seed'])

                baseline_AUROC_values = update_metric_list(baseline_AUROC_values,data_index,baseline_function,baseline_string_AUROC,desired_model_type,model_type,run_summary,OOD_dataset,run_config['seed'])
                baseline_AUPR_values = update_metric_list(baseline_AUPR_values,data_index,baseline_function,baseline_string_AUPR,desired_model_type, model_type,run_summary,OOD_dataset, run_config['seed'])
                baseline_FPR_values = update_metric_list(baseline_FPR_values,data_index,baseline_function,baseline_string_FPR,baseline_model_type, model_type,run_summary,OOD_dataset, run_config['seed'])
 
                
        print(f'ID:{ID_dataset}')
        
        difference_auroc = np.array(baseline_AUROC_values) - np.array(desired_AUROC_values) # shape (num ood, repeats)
        difference_aupr = np.array(baseline_AUPR_values) - np.array(desired_AUPR_values) # shape (num ood, repeats)
        import ipdb; ipdb.set_trace()

        # REVERSED THE DIRECTION FOR FPR DUE TO LOWER BEING BETTER FOR FPR, SO I DO NOT NEED TO REVERSE THE DIRECTION OF THE TEST STATISTIC
        difference_fpr = np.array(desired_FPR_values) - np.array(baseline_FPR_values) # shape (num ood, repeats)

        # Calculate the p values for a particular OOD dataset for this ID dataset
        for i in range(len(difference_auroc)): # go through all the different ID OOD dataset pairs
            #stat, p_value  = wilcoxon(difference[i],alternative='less') # calculate the p value for a particular ID OOD dataset pair
            stat, p_value_auroc  = wilcoxon(difference_auroc[i],alternative='less') # calculate the p value for a particular ID OOD dataset pair
            stat, p_value_aupr  = wilcoxon(difference_aupr[i],alternative='less') # calculate the p value for a particular ID OOD dataset pair
            stat, p_value_fpr  = wilcoxon(difference_fpr[i],alternative='less') # calculate the p value for a particular ID OOD dataset pair
            
            collated_rank_score_auroc[i] = p_value_auroc # add the p_value to the rank score for this particular dataset
            collated_rank_score_aupr[i] = p_value_aupr # add the p_value to the rank score for this particular dataset
            collated_rank_score_fpr[i] = p_value_fpr # add the p_value to the rank score for this particular dataset

        
        column_names_AUROC = ['AUROC']
        column_names_AUPR = ['AUPR']
        column_names_FPR = ['FPR']

        # Post processing latex table
        caption =  f'Wilcoxon Signed Rank test - {ID_dataset} P values'
        label = f'tab:Wilcoxon_test_{ID_dataset}'
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.set_option.html
        pd.set_option('display.precision',3) 

        auroc_df = pd.DataFrame(collated_rank_score_auroc,columns = column_names_AUROC, index=row_names)
        aupr_df = pd.DataFrame(collated_rank_score_aupr,columns = column_names_AUPR, index=row_names)
        fpr_df = pd.DataFrame(collated_rank_score_fpr,columns = column_names_FPR, index=row_names)
        latex_table = collated_wilcoxon_post_process_latex_table(auroc_df,aupr_df,fpr_df,caption,label)
        
        
        print(latex_table)
        
     

# Joins the different wilcoxon tables together
def collated_wilcoxon_post_process_latex_table(df_auroc, df_aupr, df_fpr,caption,label):
    
    latex_table_auroc = df_auroc.to_latex()
    latex_table_auroc = replace_headings(df_auroc,latex_table_auroc)
    
    latex_table_aupr = df_aupr.to_latex()
    latex_table_aupr = replace_headings(df_aupr,latex_table_aupr)
    
    latex_table_fpr = df_fpr.to_latex()
    latex_table_fpr = replace_headings(df_fpr,latex_table_fpr)
    
    latex_table = join_different_columns(latex_table_auroc,latex_table_aupr) # joins the auroc and aupr table together
    latex_table = join_different_columns(latex_table, latex_table_fpr) # joins the auroc+aupr table with the fpr table

    latex_table = replace_headings_collated_table(latex_table) # replaces the heading to take into account the collated readings
    
    latex_table = post_process_latex_table(latex_table)
    latex_table = initial_table_info(latex_table)
    latex_table = add_caption(latex_table,caption)
    latex_table = add_label(latex_table,label) 
    latex_table = end_table_info(latex_table)

    return latex_table


def update_metric_list(metric_list,data_index,metric_function, metric_string, metric_model_type,run_model_type, summary, OOD_dataset,seed):
    #print('metric model type:',metric_model_type)
    if metric_model_type == run_model_type and seed<200:
        metric_value = metric_function(metric_string,summary,OOD_dataset) # calculates the value
        metric_list[data_index].append(metric_value)
        return metric_list
    else:
        return metric_list# metric array with no changes


if __name__== '__main__':
    
    #knn_auroc_wilcoxon_v2()
    #knn_auroc_wilcoxon_v3('Maximum-Probability')
    #knn_auroc_wilcoxon_v3('ODIN')
    #knn_auroc_wilcoxon_v5()
    #knn_auroc_wilcoxon_v6()
    #knn_auroc_wilcoxon_repeated_runs_v1()
    #knn_auroc_wilcoxon_repeated_runs_v2()
    knn_auroc_wilcoxon_repeated_runs_v3()