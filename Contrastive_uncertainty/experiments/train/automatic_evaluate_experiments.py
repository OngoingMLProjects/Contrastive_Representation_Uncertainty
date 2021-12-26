# Automatic version which checks the callbacks for the particular file
import wandb
import copy
from Contrastive_uncertainty.experiments.train.experimental_dict import model_dict
from Contrastive_uncertainty.experiments.train.repeat_callbacks_dict import callback_names, repeat_names, desired_key_dict

def evaluate(run_paths,update_dict):    
    # Dict for the model name, parameters and specific training loop

    # Iterate through the run paths
    for run_path in run_paths:
        api = wandb.Api()    
        # Obtain previous information such as the model type to be able to choose appropriate methods
        #print('run path',run_path)
        previous_run = api.run(path=run_path)
        previous_config = previous_run.config
        model_type = previous_config['model_type']
        # Filter the callbacks, amd OOD and then update the dict for evaluation
        # Choosing appropriate methods to resume the training        
        filtered_update_dict = copy.deepcopy(update_dict)
        filtered_OOD_datasets = OOD_dataset_filter(previous_config)
        filtered_update_dict['OOD_dataset'] = filtered_OOD_datasets

        # If finished, then filter the callbacks
        if previous_run.state =='finished':
            filtered_callbacks = callback_filter(previous_run.summary._json_dict, filtered_update_dict)
        # if failed or crashed, do not filter the callbacks
        elif previous_run.state =='failed' or previous_run.state =='crashed':
            filtered_callbacks = copy.deepcopy(update_dict['callbacks'])
        elif previous_run.state =='running':
            filtered_callbacks = [] # Manually set to zero to skip this run if it is already running

        # if state finished
        filtered_update_dict['callbacks'] = filtered_callbacks
        # if state crash:
        # filtered_callbacks = copy.deepcopy(update_dict['callbacks'])
        
        evaluate_method = model_dict[model_type]['evaluate']
        model_module = model_dict[model_type]['model_module'] 
        model_instance_method = model_dict[model_type]['model_instance']
        model_data_dict = model_dict[model_type]['data_dict']
        model_ood_dict = model_dict[model_type]['ood_dict']

        
        # Checks if there are any callbacks to perform, if there is,then evaluate, otherwise look at next run
        
        if len(filtered_callbacks) > 0:
            evaluate_method(run_path, filtered_update_dict, model_module, model_instance_method, model_data_dict,model_ood_dict)
        

def callback_filter(summary_info,evaluation_dict):
    callbacks = evaluation_dict['callbacks']
    
    # NEED TO MAKE A CALLBACK DICT WHICH DOES NOT TAKE INTO ACCOUNT OOD DATASETS
    # Need to check the summary information for GRADCAM heatmaps
    # 
    
    # Update to take into account different OOD datasets
    OOD_datasets = evaluation_dict['OOD_dataset']
    updated_desired_key_dict = {key:[] for key in desired_key_dict} # obtain a list for the dataset
    for key in updated_desired_key_dict: # go through all the keys in the updated_desired_key dict (which has same keys as desired key dict )  
        for value in desired_key_dict[key]: # go through the different values in a key
            updated_desired_key_dict[key].extend([value + f' {ood_dataset}' for ood_dataset in OOD_datasets]) # add list related to the OOD dataset
       
    filtered_callbacks = []
    # Make a dict connecting the callbacks and the inputs from the callbacks
    for callback in callbacks:
        repeat_callback = summary_info[repeat_names[callback]] # Boolean to check whether the callback should be repeated - need to change this to make it so that all the callbacks can be repeated or not repeated

        desired_strings = updated_desired_key_dict[callback] # get the summary strings related to the callback
        # iterate throguh the strings
        for desired_string in desired_strings:
            desired_keys = [key for key, value in summary_info.items() if desired_string.lower().replace(" ","") in key.lower().replace(" ","")] # check if the key has the desired string, remove the spaces to get rid of any issues with space
            if len(desired_keys) == 0 or repeat_callback: # if any of the strings in a callback is zero, append the callback to filtered callback and go to the next callback
                filtered_callbacks.append(callback)
                break
                
    '''
    filtered_callbacks = []
    # Make a dict connecting the callbacks and the inputs from the callbacks
    for callback in callbacks:
        desired_strings = desired_key_dict[callback] # get the summary strings related to the callback
        # iterate throguh the strings
        for desired_string in desired_strings:
            desired_keys = [key for key, value in summary_info.items() if desired_string.lower() in key.lower()] # check if the key has the desired string
            if len(desired_keys) == 0: # if any of the strings in a callback is zero, append the callback to filtered callback and go to the next callback
                filtered_callbacks.append(callback)
                break
    '''
    return filtered_callbacks



# Used to choose a specific OOD dataset based on the ID dataset
def OOD_dataset_filter(config):
    MNIST_variants = ['MNIST','FashionMNIST','KMNIST']
    # Checks if the ID dataset is an MNIST dataset
    if config['dataset'] in MNIST_variants:
        OOD_dataset = ['MNIST','FashionMNIST','KMNIST','EMNIST']
    else:
        OOD_dataset = ['STL10', 'CelebA','WIDERFace','SVHN', 'Caltech101','Caltech256','CIFAR10','CIFAR100', 'VOC', 'Places365','TinyImageNet','Cub200','Dogs', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST']
    return OOD_dataset
