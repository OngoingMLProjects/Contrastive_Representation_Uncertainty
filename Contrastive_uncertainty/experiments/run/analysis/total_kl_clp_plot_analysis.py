# Plotting the centroid distances against the confuson log probability

import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import general params
import json

# Make it so that the model can loop for the different datasets as well as different models
def total_kl_clp_plot():
    # Desired ID,OOD and Model
    Models = ['Moco','SupCon']
    #Models = ['SupCon']
    desired_key = 'KL Divergence(Total||Class)'

    all_ID = ['MNIST','FashionMNIST','KMNIST', 'CIFAR10','CIFAR100']
    all_OOD = {'MNIST':['FashionMNIST','KMNIST'],
    'FashionMNIST':['MNIST','KMNIST'],
    'KMNIST':['MNIST','FashionMNIST'],
    'CIFAR10':['SVHN','CIFAR100'],
    'CIFAR100':['SVHN','CIFAR10']}

    for Model in Models:# Go through the different models
        for ID in all_ID: # Go through the different ID dataset
            for OOD in all_OOD[ID]: # Go through the different OOD datasets for a particular ID dataset        
                api = wandb.Api()
                # Gets the runs corresponding to a specific filter
                # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py

                # Second part used of the filter used for the purpose of an or statement to calculate the different values (Shown in the github link above )
                #runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","$or": [{"config.model_type":"Moco" }, {"config.model_type": "SupCon"}]}) # only look at the runs related to Moco and SupCon
                #runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.seed":{"$ne": 26},"config.seed":{"$ne": 42}}) # only look at the runs related to Moco and SupCon
                #runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.seed":{"$ne": 26}, "config.seed":{"$ne": 42},"$or": [{"config.model_type":"Moco" }, {"config.model_type": "SupCon"}]})
                #runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.seed":{"$ne": 26},"config.seed":{"$ne": 42}})
                #runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines", "config.seed": { "$in": [25, 50] } })
                #runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines", "config.seed": {"$lt": 150} })
                
                #runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats", "config.seed": {"$lt": 150} })
                #runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats", "config.seed": {"$lt": 125} })
                #runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats", "config.seed":{"$ne": 26} })
                # Need to use an 'and' statement to combine the different conditions on the approach
                runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats","config.dataset":ID,"config.model_type":Model ,"$and": [{"config.seed":{"$ne": 26}},{"config.seed":{"$ne": 42}}]})


                #runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.seed":[{"$ne": 26},{"$ne": 42}]})
                #Filtering based on https://docs.mongodb.com/manual/reference/operator/query/ne/#mongodb-query-op.-ne
                summary_list, config_list, name_list = [], [], []

                # Dict to map distances of specific datasets and model types to the data array
                key_dict = {'dataset':{'MNIST':0, 'FashionMNIST':1,'KMNIST':2, 'CIFAR10':3, 'CIFAR100':4},
                            'model_type':{'Moco':0, 'SupCon':1}}

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
                    # seed = config_list[i]['seed']
                    # if seed == 26 or seed == 42:
                        # 
                        # pass
                    # else:

                    group_name = config_list[i]['group']
                    path_list = runs[i].path
                    path_list.insert(-1, group_name) # insert the group name in the location one before the last value (rather than the last value which is peculiar)
                    run_path = '/'.join(path_list)
                    run_dir = root_dir + run_path
                    data_dir =  summary_list[i][desired_key]['path']
                    run_dir = root_dir + run_path
                    # Read dir is how to read the file
                    read_dir = run_dir + '/' + data_dir


                    #data_kl_dir = summary_list[i]['KL Divergence(Total||Class)']['path']
                    # Obtain the dataset and the model type
                    #run_dir = root_dir + run_path
                    # Read dir is how to read the file
                    #read_dir = run_dir + '/' + data_kl_dir
                    
                    dataset = config_list[i]['dataset']
                    model_type = config_list[i]['model_type']
                    
                    if ID == dataset and Model == model_type:
                        with open(read_dir) as f:
                            total_kl_data = json.load(f)
                        # Calculate the mean distance
                        total_kl_values = total_kl_div_vector(total_kl_data)
                        collated_total_kl_values.append(total_kl_values)

                        #break # To stop the loop
                    
                    
                clp_runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Confusion Log Probability Evaluation"})
                #desired_key = 'Class Wise Confusion Log Probability'
                #desired_key = desired_key.lower()
                clp_summary_list, clp_config_list, clp_name_list = [], [], []
                root_dir = 'run_data/'
                for i, clp_run in enumerate(clp_runs): 

                    # .summary contains the output keys/values for metrics like accuracy.
                        #  We call ._json_dict to omit large files 
                    clp_summary_list.append(clp_run.summary._json_dict)

                    # .config contains the hyperparameters.
                    #  We remove special values that start with _.
                    clp_config_list.append(
                        {k: v for k,v in clp_run.config.items()
                         if not k.startswith('_')})

                    ID_dataset = clp_config_list[i]['dataset']
                    OOD_dataset = clp_config_list[i]['OOD_dataset'][0]
                    group_name = clp_config_list[i]['group']
                    if ID == ID_dataset and OOD == OOD_dataset:

                        # Take into account the difference in the name of the model for SupCon and SupCLR
                        Model_name = 'SupCLR' if Model=='SupCon' else Model
                        path_list = clp_runs[i].path
                        path_list.insert(-1, group_name) # insert the group name in the location one before the last value (rather than the last value which is peculiar)
                        run_path = '/'.join(path_list)
                        run_dir = root_dir + run_path
                        clp_values_dir = clp_summary_list[i]['Class Wise Confusion Log Probability']['path']
                        read_dir = run_dir + '/' + clp_values_dir

                        with open(read_dir) as f:
                            class_wise_clp_json = json.load(f)
                        
                        clp_values = class_confusion_log_probability_vector(class_wise_clp_json)
                        # Gets the number of repeat measurements
                        num_repeats = len(collated_total_kl_values)
                        collated_clp_values = np.tile(clp_values,num_repeats)
                        collated_total_kl_values = np.concatenate(collated_total_kl_values)

                        # Normalization
                        collated_total_kl_values= collated_total_kl_values / np.max(collated_total_kl_values)

                        #collated_total_kl_values = np.stack(collated_total_kl_values,axis=0)
                        collated_array = np.stack((collated_total_kl_values,collated_clp_values),axis=1)
                        #collated_array = np.stack((total_kl_values,clp_values),axis=1)  
                        df = pd.DataFrame(collated_array)
                        columns = ['KL(Overall||Class) (Nats)', 'CLP']
                        df.columns = columns

                        fit = np.polyfit(df['KL(Overall||Class) (Nats)'], df['CLP'], 1)
                        fig = plt.figure(figsize=(10, 7))
                        sns.regplot(x = df['KL(Overall||Class) (Nats)'], y = df['CLP'],color='blue')
                        plt.annotate('y={:.2f}+{:.2f}*x'.format(fit[1], fit[0]), xy=(0.80, 0.95), xycoords='axes fraction') # Used to control where the line is annotated
                        #plt.text(3.2, -7.12, 'y={:.2f}+{:.2f}*x'.format(fit[1], fit[0]), color='darkblue', size=12)
                        plt.title(f'KL Divergence and Confusion Log Probability for {ID}-{OOD} pair using {Model_name} model', size=12)
                        # regression equations
                        
                        folder = f'Scatter_Plots/KL_CLP_plots/{model_type}'
                        
                        if not os.path.exists(folder):
                            os.makedirs(folder)
                        plt.savefig(f'{folder}/KL_CLP_{ID}_{OOD}_{Model_name}.png')
                        plt.close()

def class_confusion_log_probability_vector(json_data):
    data = np.array(json_data['data'])
    clp_values = np.around(data[:,0],decimals=3)
    return clp_values

def total_kl_div_vector(json_data):
    data = np.array(json_data['data'])
    kl_values = np.around(data[:,0],decimals=3)
    return kl_values

if __name__ == '__main__':
    total_kl_clp_plot()