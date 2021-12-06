import sys
from numpy.core.fromnumeric import repeat
import wandb
from Contrastive_uncertainty.experiments.train.experimental_dict import model_dict
from Contrastive_uncertainty.general.run.model_names import model_names_dict
from Contrastive_uncertainty.experiments.train.repeat_callbacks_dict import callback_names,desired_key_dict

'''
# Choose quick callbacks to repeat
repeat_callbacks=[callback_names['NN Quadratic']]
repeat_bool = {key:False for key in desired_key_dict}

# Sets the particulat key to true
for key in repeat_bool:
    if key in repeat_callbacks:
        repeat_bool[key] = True
'''

# Choose quick callbacks to repeat
#repeat_callbacks=['NN Quadratic']
#repeat_callbacks=['MSP']
repeat_callbacks=[]
if len(repeat_callbacks)>0:
    for repeat_callback in repeat_callbacks:
        assert repeat_callback in callback_names, 'not in callback names'
    repeat_callbacks = [f'Repeat {key}'for key in repeat_callbacks]     

repeat_bool = {f'Repeat {key}':False for key in callback_names}
# Sets the particulat key to true
for key in repeat_bool:
    if key in repeat_callbacks:
        repeat_bool[key] = True

#import ipdb; ipdb.set_trace()
run_paths = []
api = wandb.Api()

#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"to_delete"})
#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats","config.model_type": "SupCon","config.epochs":300, 'state':'finished'})
runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats","config.epochs":300, 'state':'finished'})
# Make the run paths for the different runs
for i in range(len(runs)):
    # Joins together the path of the runs which are separated into different parts in a list
    run_path = '/'.join(runs[i].path)
    run_paths.append(run_path)

# Go through all the runs
for run_path in run_paths:
    previous_run = api.run(path=run_path)
    previous_config = previous_run.config

    config = previous_config
    
    run = wandb.init(id=previous_run.id,resume='allow',project=config['project'],group=config['group'], notes=config['notes'])
    wandb.log(repeat_bool)
    wandb.config.update(config, allow_val_change=True) # Updates the config (particularly used to increase the number of epochs present)
    run.finish()