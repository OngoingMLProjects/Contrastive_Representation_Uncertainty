# Plotting the AUROC as the value of K changes for the group typicality approach


from asyncore import read
from numpy.core.numeric import full
from torch.utils.data import dataset
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import wilcoxon
import matplotlib.patches as mpatches

import json
import re


from Contrastive_uncertainty.experiments.run.analysis.analysis_utils import dataset_dict, key_dict, ood_dataset_string
'''
This file is used to make plots for the KNN outlier fraction as well as the KNN class fraction
'''

def fraction_vector(json_data):
    data = np.array(json_data['data'])
    fraction_values = np.around(data,decimals=3)
    return fraction_values

# Nawid - New version of the code which makes the plot as a line plot rather than a scatter plot 
def thesis_outlier_fraction_plot():
    approach = 'Quadratic_Typicality'
    # Desired ID,OOD and Model  
    root_dir = 'run_data/'

    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py
    runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.model_type": "SupCon","config.epochs": 300})

    summary_list, config_list, name_list = [], [], []
    
    
    key_dict = {'dataset':{'CIFAR10':0, 'CIFAR100':1,'Caltech256':2,'TinyImageNet':3},
                'model_type':{'SupCon':0}}
    
    for i, run in enumerate(runs): 
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        config_list.append(
            {k: v for k,v in run.config.items()
             if not k.startswith('_')})


        ID_dataset = config_list[i]['dataset']
        model_type = config_list[i]['model_type']
        group_name = config_list[i]['group']
        seed_value = str(config_list[i]['seed'])
        summary_list.append(run.summary._json_dict)
        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        


        #updated path which includes the group of the dataset
        path_list = runs[i].path
        path_list.insert(-1, group_name) # insert the group name in the location one before the last value (rather than the last value which is peculiar)
        # Additional lines added compared to the previous file
        path_list.insert(-1,model_type)
        path_list.insert(-1,ID_dataset)
        path_list.insert(-1,seed_value)
        ###################################################
        run_path = '/'.join(path_list)
        
        #run_path = '/'.join(runs[i].path)
        # .name is the human-readable name of the run.dir
        name_list.append(run.name)

        Model_name = 'SupCLR' if model_type=='SupCon' else model_type
        # .name is the human-readable name of the run.dir
        desired_string = 'K:10 NN Outlier Percentage OOD'.lower()
        knn_keys = [key for key, value in summary_list[i].items() if desired_string in key.lower()]
        
        
        # go through the different knn keys
        for key in knn_keys:
            OOD_dataset = ood_dataset_string(key, dataset_dict, ID_dataset)
            if OOD_dataset is None:
                pass
            else:
                # get the specific mahalanobis keys for the specific OOD dataset  
                print(f'ID: {ID_dataset}, OOD {OOD_dataset}')
                data_dir = summary_list[i][key]['path']                    
                run_dir = root_dir + run_path
                read_dir = run_dir + '/' + data_dir
                print('read dir:',read_dir)
                isFile = os.path.isfile(read_dir)
                print('Is file:',isFile)
                    
                if isFile:
                    with open(read_dir) as f: 
                        data = json.load(f)
                        outlier_values = fraction_vector(data)
                        '''
                        Just to double check whether there were values for 0.3 and 0.7
                        thress_vals_1 = outlier_values[:,0] ==0.7
                        thress_vals_2 = outlier_values[:,0] ==0.7
                        print('three vals 1:',thress_vals_1.sum())
                        print('three vals 2:',thress_vals_2.sum())
                        '''  
                        df = pd.DataFrame(outlier_values)
                        columns = ['ID', 'OOD']
                        df.columns = columns
                        fig = plt.figure()
                        ax = plt.subplot(111)
                        # Hack - seems to be an issue with the values being below zero (remove)
                        
                        ID_values = outlier_values[:,0]
                        ID_values = ID_values[ID_values>=0]
                        
                        OOD_values = outlier_values[:,1]
                        OOD_values = OOD_values[OOD_values>=0]
        
                        ID_values, ID_counts = np.unique(ID_values*100, return_counts=True)
                        ID_values = np.asarray(ID_values, dtype = 'int')
                        OOD_values, OOD_counts = np.unique(OOD_values*100, return_counts=True)
                        OOD_values = np.asarray(OOD_values, dtype = 'int')

                        ID_data = {'Value': ID_values, 'Counts': ID_counts}
                        ID_df = pd.DataFrame(data=ID_data)
                        ID_df['Category']='ID'
                        OOD_data = {'Value': OOD_values, 'Counts': OOD_counts}
                        OOD_df = pd.DataFrame(data=OOD_data)
                        OOD_df['Category']='OOD'
                        
                        collated_df = pd.concat((ID_df,OOD_df))
                        #print('collated df:',collated_df)

                        # Based on https://stackoverflow.com/questions/38807895/seaborn-multiple-barplots
                        
                        g = sns.catplot(x='Value', y='Counts', hue='Category', data=collated_df, kind='bar',legend_out=True)
                        
                        # Changing figure height based on https://stackoverflow.com/questions/31594549/how-to-change-the-figure-size-of-a-seaborn-axes-or-figure-level-plot
                        g.fig.set_figwidth(7.27)
                        g.fig.set_figheight(7.27)
                        


                        '''
                        #sns.factorplot("Value", "Counts", col="Category", data=collated_df, kind="bar")
                        
                        plt.bar(*np.unique(outlier_values[:,0]*100, return_counts=True),alpha=0.7)
                        plt.bar(*np.unique(outlier_values[:,1]*100, return_counts=True),alpha=0.7) 
                        
                        #sns.histplot(outlier_values[:,0],binwidth=0.1,kde=False,alpha=0.5,color='b')
                        #sns.histplot(outlier_values[:,1],binwidth=0.1,kde=False,alpha=0.5,color='r')
                        
                        # Making colored labels based on https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html
                        blue_patch = mpatches.Patch(color='blue', label='ID')
                        red_patch = mpatches.Patch(color='orange', label='OOD')

                        # Changing position of legend Based on https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
                        #plt.legend(handles=[blue_patch,red_patch])
                        box = ax.get_position()
                        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                        ax.legend(loc='center left',handles=[blue_patch,red_patch], bbox_to_anchor=(1, 0.5))
                        '''
                        plt.xlabel('Outlier Fraction')
                        #plt.xlim((0,100))
                        plt.ylim((0,10000))
                        plt.title(f'KNN Outlier Fraction for the {ID_dataset}-{OOD_dataset} pair')
                        #plt.tight_layout()
                        
                        # Make a line plot for the data
                        #plt.text(3.2, -7.12, 'y={:.2f}+{:.2f}*x'.format(fit[1], fit[0]), color='darkblue', size=12)
                        folder = f'Scatter_Plots/Thesis_v2/Outlier_Fraction/{Model_name}/{ID_dataset}/{approach}'
                        if not os.path.exists(folder):
                            os.makedirs(folder)
                        plt.savefig(f'{folder}/Outlier_Fraction_{ID_dataset}_{OOD_dataset}.png')
                        plt.close()

def thesis_class_fraction_plot():
    approach = 'Quadratic_Typicality'
    # Desired ID,OOD and Model  
    root_dir = 'run_data/'

    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py
    runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.model_type": "SupCon","config.epochs": 300})

    summary_list, config_list, name_list = [], [], []

    for i, run in enumerate(runs): 
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        config_list.append(
            {k: v for k,v in run.config.items()
             if not k.startswith('_')})

        ID_dataset = config_list[i]['dataset']
        model_type = config_list[i]['model_type']
        group_name = config_list[i]['group']
        seed_value = str(config_list[i]['seed'])
        summary_list.append(run.summary._json_dict)
        # .config contains the hyperparameters.
        #  We remove special values that start with _.
    
        #updated path which includes the group of the dataset
        path_list = runs[i].path
        path_list.insert(-1, group_name) # insert the group name in the location one before the last value (rather than the last value which is peculiar)
        # Additional lines added compared to the previous file
        path_list.insert(-1,model_type)
        path_list.insert(-1,ID_dataset)
        path_list.insert(-1,seed_value)
        ###################################################
        run_path = '/'.join(path_list)
        
        #run_path = '/'.join(runs[i].path)
        # .name is the human-readable name of the run.dir
        name_list.append(run.name)

        Model_name = 'SupCLR' if model_type=='SupCon' else model_type
        # .name is the human-readable name of the run.dir
        desired_string = 'K:10 NN Class Fraction'.lower()
        knn_keys = [key for key, value in summary_list[i].items() if desired_string in key.lower()]
        
        
        # go through the different knn keys
        for key in knn_keys:
            # get the specific mahalanobis keys for the specific OOD dataset  
            print(f'ID: {ID_dataset}')
            data_dir = summary_list[i][key]['path']                    
            run_dir = root_dir + run_path
            read_dir = run_dir + '/' + data_dir
            print('read dir:',read_dir)
            isFile = os.path.isfile(read_dir)
            print('Is file:',isFile)
                    
            if isFile:
                with open(read_dir) as f: 
                    data = json.load(f)
                
                class_fraction_values = fraction_vector(data)
                
                sns.histplot(class_fraction_values,binwidth=0.1,kde=False,alpha=0.5,color='b')
                                
                plt.xlabel('Class Fraction')
                plt.xlim((0,1))
                plt.ylim((0,10000))
                plt.title(f'KNN Class Fraction for {ID_dataset}')
                # Removing legend based on https://stackoverflow.com/questions/5735208/remove-the-legend-on-a-matplotlib-figure
                plt.legend().set_visible(False)
                
                # Make a line plot for the data
                #plt.text(3.2, -7.12, 'y={:.2f}+{:.2f}*x'.format(fit[1], fit[0]), color='darkblue', size=12)
                folder = f'Scatter_Plots/Thesis_v2/Class_Fraction/{Model_name}/{ID_dataset}/{approach}'
                if not os.path.exists(folder):
                    os.makedirs(folder)
                plt.savefig(f'{folder}/Class_Fraction_{ID_dataset}.png')
                plt.close()

if __name__ =='__main__':
    #knn_auroc_plot_v4()
    #thesis_knn_auroc_plot()
    #thesis_class_fraction_plot()
    thesis_outlier_fraction_plot()
    
    
