# Automatic version which checks the callbacks for the particular file
import wandb
import copy
from Contrastive_uncertainty.experiments.train.automatic_evaluate_experiments import desired_key_dict, callback_filter,OOD_dataset_filter

from Contrastive_uncertainty.basic_replica.basic_experiments.train.experimental_dict import model_dict

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
