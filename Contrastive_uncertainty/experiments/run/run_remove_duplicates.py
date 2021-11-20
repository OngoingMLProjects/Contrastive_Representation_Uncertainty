import sys
import wandb
from Contrastive_uncertainty.experiments.train.experimental_dict import model_dict
from Contrastive_uncertainty.general.run.model_names import model_names_dict



run_paths = []
api = wandb.Api()
runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats","config.model_type": "CE","config.epochs":300, 'state':'finished'})

# Make the run paths for the different runs
for i in range(len(runs)):
    # Joins together the path of the runs which are separated into different parts in a list
    run_path = '/'.join(runs[i].path)
    run_paths.append(run_path)

# Go through all the runs
duplicate_run_paths = []
for run_path in run_paths:
    previous_run = api.run(path=run_path)
    previous_config = previous_run.config

    run_filter={"config.epochs":previous_config['epochs'],"config.group":previous_config['group'],"config.model_type":previous_config['model_type'] ,"config.dataset": previous_config['dataset'],"config.seed":previous_config['seed']}
    project_path = 'nerdk312/' +previous_config['project'] # Need to make sure I get the correct path for projects
    filtered_runs = api.runs(path=project_path, filters=run_filter) # get the filtered runs
    if len(filtered_runs)> 1:
        print('Duplicates present:',len(filtered_runs)) 
        duplicate_path = '/'.join(filtered_runs[-1].path)
        duplicate_run_paths.append(duplicate_path)
    else:
        print('No Duplicates present',len(filtered_runs))

    # Remove runs in duplicate path
    #     
    if run_path in duplicate_run_paths: 
        config = previous_config
        config['group'] = 'Additional unused runs'
        config['notes'] = 'Duplicate runs'
        run = wandb.init(id=previous_run.id,resume='allow',project=config['project'],group=config['group'], notes=config['notes'])
        wandb.config.update(config, allow_val_change=True) # Updates the config (particularly used to increase the number of epochs present)
        run.finish()
        
    # During run, this is when the group, notes and config are able to change for the task
