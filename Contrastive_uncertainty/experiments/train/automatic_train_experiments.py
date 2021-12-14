import sys
import wandb
from Contrastive_uncertainty.experiments.train.experimental_dict import model_dict
from Contrastive_uncertainty.general.run.model_names import model_names_dict

def train(base_dict, trainer_dict):   
    # first obtain a base dict and a trainer dict
    #model_types = [model_names_dict['SupCon']]
    model_types = [model_names_dict['CE']]
    seeds = [26,42]
    datasets = ['Caltech256','CIFAR10','CIFAR100','TinyImageNet']
    #datasets = ['Caltech101','Caltech256','CIFAR10','CIFAR100','TinyImageNet','Cub200','Dogs'] # Need to perform MNIST experiments next
    #datasets = ['Caltech101','Caltech256','CIFAR10','CIFAR100','TinyImageNet','Cub200','Dogs','MNIST','FashionMNIST','KMNIST']
    

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

    
# Remove runs which are incomplete (num of epochs does not match)
def remove_incomplete(base_dict):
    run_paths = []
    api = wandb.Api()
    project_path ="nerdk312/" +base_dict['project']
    # Obtain the runs for the particlar project
    runs = api.runs(path=project_path, filters={"config.group":base_dict['group'],"config.model_type": base_dict['model_type'],"config.dataset":base_dict['dataset'], "$or": [{'state':'finished'}, {'state':'crashed'},{'state':'failed'}]})
    # Make the run paths
    for i in range(len(runs)):
        # Joins together the path of the runs which are separated into different parts in a list
        run_path = '/'.join(runs[i].path)
        run_paths.append(run_path)
    # go through the different run paths
    for run_path in run_paths:
        previous_run = api.run(path=run_path)
        previous_config = previous_run.config
        
        summary_dict = previous_run.summary._json_dict

        if 'epoch' in summary_dict: 
            if previous_config['epochs']-1 <= summary_dict['epoch']: # Should be alright if the summary dict is more than config amount
                pass
            else:
                previous_config['group'] = 'to_delete'
                previous_config['notes'] = 'incomplete run as training epochs did not reach end'
                # Only update if the run is incomplete. During run, this is when the group, notes and config are able to change for the task
                
                run = wandb.init(id=previous_run.id,resume='allow',project=previous_config['project'],group=previous_config['group'], notes=previous_config['notes'])
                wandb.config.update(previous_config, allow_val_change=True) # Updates the config (particularly used to increase the number of epochs present)
                run.finish()
                
        # If epoch is not in the summary dict
        else: 
            previous_config['group'] = 'to_delete'
            previous_config['notes'] = 'incomplete run as training epochs did not reach end'

            # Only update if the run is incomplete. # During run, this is when the group, notes and config are able to change for the task
            run = wandb.init(id=previous_run.id,resume='allow',project=previous_config['project'],group=previous_config['group'], notes=previous_config['notes'])
            wandb.config.update(previous_config, allow_val_change=True) # Updates the config (particularly used to increase the number of epochs present)
            run.finish()