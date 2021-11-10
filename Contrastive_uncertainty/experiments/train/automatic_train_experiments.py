import sys
import wandb
from Contrastive_uncertainty.experiments.train.experimental_dict import model_dict
from Contrastive_uncertainty.general.run.model_names import model_names_dict

def train(base_dict, trainer_dict):   
    # first obtain a base dict and a trainer dict
    model_types = [model_names_dict['SupCon']]
    seeds = [25,50,75,100,125,150,175,200]
    datasets = ['Caltech101','Caltech256','CIFAR10','CIFAR100','TinyImageNet','Cub200','Dogs'] # Need to perform MNIST experiments next

    

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
                    already_present = runs_present(base_params, trainer_dict)
                    # if there is not a run present, then train a model
                    if not already_present:
                        # Try statement to allow the code to continue even if a single run fails
                        train_method(base_params,trainer_dict, model_module, model_instance_method,model_data_dict)

# Checks if there is a run present for a particular run           
def runs_present(base_params, trainer_params):
    runs_present = False
    api = wandb.Api()
    params = {}
    # place base params and trainer params in the dictionary
    params.update(base_params)
    params.update(trainer_params)
    
    run_filter={"config.epochs":params['epochs'],"config.group":params['group'],"config.model_type":params['model_type'] ,"config.dataset": params['dataset'],"config.seed":params['seed']}
    project_path = 'nerdk312/' +base_params['project'] # Need to make sure I get the correct path for projects
    runs = api.runs(path=project_path, filters=run_filter)
    if len(runs)> 0:
        print('runs present:',len(runs))
        runs_present = True
    else:
        print('no runs present:',len(runs))
        runs_present = False

    return runs_present

    
