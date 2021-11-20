# Automatic version which checks the callbacks for the particular file
import wandb
import copy
from Contrastive_uncertainty.experiments.train.experimental_dict import model_dict

#redo_callback = ['Maximum Softmax Probability']


desired_key_dict = {'Mahalanobis Distance':['Mahalanobis AUROC OOD','Mahalanobis AUPR OOD','Mahalanobis FPR OOD'],
'Nearest 10 Neighbours Class Quadratic 1D Typicality':['Normalized One Dim Class Quadratic Typicality KNN - 10 OOD','Normalized One Dim Class Quadratic Typicality KNN - 10 AUPR OOD','Normalized One Dim Class Quadratic Typicality KNN - 10 FPR OOD'],
'Nearest 10 Neighbours Class 1D Typicality': ['Normalized One Dim Class Typicality KNN - 10 OOD','Normalized One Dim Class Typicality KNN - 10 AUPR OOD','Normalized One Dim Class Typicality KNN - 10 FPR OOD'],
'Maximum Softmax Probability': ['Maximum Softmax Probability AUROC OOD','Maximum Softmax Probability AUPR OOD','Maximum Softmax Probability FPR OOD'],
'ODIN':['ODIN AUROC OOD']}

'''
# Need to redo the callback
for callback in desired_key_dict:
    if callback in redo_callback:
        desired_key_dict[callback] = ['Redo callback']
'''

def evaluate(run_paths,update_dict):    
    
    # Dict for the model name, parameters and specific training loop

    # Iterate through the run paths
    for run_path in run_paths:
        api = wandb.Api()    
        # Obtain previous information such as the model type to be able to choose appropriate methods
        
        previous_run = api.run(path=run_path)
        previous_config = previous_run.config
        model_type = previous_config['model_type']
        # Filter the callbacks, amd OOD and then update the dict for evaluation

        # If finished, then filter the callbacks
        if previous_run.state =='finished':
            filtered_callbacks = callback_filter(previous_run.summary._json_dict, update_dict)
        # if failed or crashed, do not filter the callbacks
        elif previous_run.state =='failed' or previous_run.state =='crashed':
            filtered_callbacks = copy.deepcopy(update_dict['callbacks'])
        elif previous_run.state =='running':
            filtered_callbacks = [] # Manually set to zero to skip this run if it is already running

        filtered_OOD_datasets = OOD_dataset_filter(previous_config)

        # Choosing appropriate methods to resume the training        
        filtered_update_dict = copy.deepcopy(update_dict)

        # if state finished
        filtered_update_dict['callbacks'] = filtered_callbacks
        # if state crash:
        # filtered_callbacks = copy.deepcopy(update_dict['callbacks'])
        
        filtered_update_dict['OOD_dataset'] = filtered_OOD_datasets


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
    return filtered_callbacks

    '''
    # Make a dict connecting the callbacks and the inputs from the callbacks
    for callback in callbacks:
        desired_string = desired_key_dict[callback].lower() 
        desired_keys = [key for key, value in summary_info.items() if desired_string in key.lower()]
        # if there are no keys already present, then the callback has not been used yet
        if len(desired_keys) == 0:
            filtered_callbacks.append(callback)

    return filtered_callbacks 
    '''

# Used to choose a specific OOD dataset based on the ID dataset
def OOD_dataset_filter(config):
    MNIST_variants = ['MNIST','FashionMNIST','KMNIST']
    # Checks if the ID dataset is an MNIST dataset
    if config['dataset'] in MNIST_variants:
        OOD_dataset = ['MNIST','FashionMNIST','KMNIST','EMNIST']
    else:
        OOD_dataset = ['STL10', 'CelebA','WIDERFace','SVHN', 'Caltech101','Caltech256','CIFAR10','CIFAR100', 'VOC', 'Places365','TinyImageNet','Cub200','Dogs', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST']
    return OOD_dataset
    