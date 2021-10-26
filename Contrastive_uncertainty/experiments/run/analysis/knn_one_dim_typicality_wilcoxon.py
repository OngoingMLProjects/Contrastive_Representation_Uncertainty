# Plotting the AUROC as the value of K changes for the group typicality approach
# Also works on making tables for the ID and OOD datasets

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
from Contrastive_uncertainty.experiments.run.analysis.knn_one_dim_typicality_diagrams import knn_vector, full_post_process_latex_table,post_process_latex_table,\
    replace_headings, bold_max_value, initial_table_info, add_caption, add_label, end_table_info

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
    
    caption =  'Wilcoxon Signed Rank test - P values'
    label = f'tab:Wilcoxon_test'
    # Post pr
    auroc_df = pd.DataFrame(collated_rank_score, columns = column_names, index=dataset_row_names)
    latex_table = auroc_df.to_latex()
    
    latex_table = replace_headings(auroc_df,latex_table)
    latex_table = post_process_latex_table(latex_table)
    latex_table = initial_table_info(latex_table)
    latex_table = add_caption(latex_table,caption)
    latex_table = add_label(latex_table,label) 
    latex_table = end_table_info(latex_table)

    latex_table = latex_table.replace('{}','{Datasets}')
    latex_table = latex_table.replace("lrr","|p{3cm}|c|c|")
    latex_table = post_process_latex_table(latex_table)

    print(latex_table)

if __name__== '__main__':
    
    knn_auroc_wilcoxon_v2()
    
