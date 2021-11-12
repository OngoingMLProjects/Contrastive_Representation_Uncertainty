from os import remove
import sys
from graphql.language.parser import parse_definition
import wandb
from Contrastive_uncertainty.toy_replica.toy_experiments.train.experimental_dict import model_dict
from Contrastive_uncertainty.general.run.model_names import model_names_dict
from Contrastive_uncertainty.experiments.train.automatic_train_experiments import runs_present, remove_incomplete

def train(base_dict, trainer_dict):   
    # first obtain a base dict and a trainer dict
    model_types = [model_names_dict['SupCon']]
    seeds = [21]
    datasets = ['Blobs']

    # Update the parameters of each model
    # the seeds as well as updating the dataset
    

    # iterate through all items of the state dict, update the parameters related to the dict
    for base_k, base_v in base_dict.items():
        # Iterate through all the model dicts
        for model_k, model_v in model_dict.items():
            # Go through each dict one by one and check if base k in model params
            if base_k in model_dict[model_k]['params']:
                # update model key with base params
                model_dict[model_k]['params'][base_k] = base_v

    
    
    # Update the seeds as well as the datasets

    for model_k, model_v in model_dict.items():
        if model_k in model_types:
            train_method = model_dict[model_k]['train']
            model_module = model_dict[model_k]['model_module'] 
            model_instance_method = model_dict[model_k]['model_instance']
            model_data_dict = model_dict[model_k]['data_dict']
            
            base_params = model_dict[model_k]['params']
            # update dataset as well as model
            
            
            for seed in seeds:
                trainer_dict['seed'] = seed
                for dataset in datasets:
                    base_params['dataset'] = dataset
                    # remove incomplete runs related to a specific model and specific dataset
                    remove_incomplete(base_params)
                    already_present = runs_present(base_params, trainer_dict)
                    # if there is not a run present, then train a model
                    if not already_present:
                        # Try statement to allow the code to continue even if a single run fails
                        train_method(base_params,trainer_dict, model_module, model_instance_method,model_data_dict)




