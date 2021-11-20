import os 
import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from Contrastive_uncertainty.general.run.general_run_setup import train_run_name, eval_run_name,callback_dictionary, specific_callbacks, Datamodule_selection

'''
callback_names = {'MD':'Mahalanobis Distance',
'NN Quadratic':'Nearest 10 Neighbours Class Quadratic 1D Typicality',
'NN Linear':'Nearest 10 Neighbours Class 1D Typicality',
'MSP':'Maximum Softmax Probability'}


desired_key_dict = {callback_names['MD']:['Mahalanobis AUROC OOD','Mahalanobis AUPR OOD','Mahalanobis FPR OOD'],
callback_names['NN Quadratic']:['Normalized One Dim Class Quadratic Typicality KNN - 10 OOD -','Normalized One Dim Class Quadratic Typicality KNN - 10 AUPR OOD -','Normalized One Dim Class Quadratic Typicality KNN - 10 FPR OOD -'],
callback_names['NN Linear']: ['Normalized One Dim Class Typicality KNN - 10 OOD -','Normalized One Dim Class Typicality KNN - 10 AUPR OOD -','Normalized One Dim Class Typicality KNN - 10 FPR OOD -'],
callback_names['MSP']: ['Maximum Softmax Probability AUROC OOD','Maximum Softmax Probability AUPR OOD','Maximum Softmax Probability FPR OOD']}
'''

# Train takes in params, a particular training module as well a model_function to instantiate the model
def train(base_params,trainer_params, model_module,model_function,datamodule_dict):
    params = {}
    # place base params and trainer params in the dictionary
    params.update(base_params)
    params.update(trainer_params)

    run = wandb.init(entity="nerdk312",config = params, project= params['project'], reinit=True,group=params['group'], notes=params['notes'])  # Required to have access to wandb config, which is needed to set up a sweep
    wandb_logger = WandbLogger(log_model=True, sync_step=False, commit=False)
    config = wandb.config
    
    # Gets the path which could be used for evaluation
    run_path = wandb.run.path
    
    '''
    # Choose quick callbacks to repeat
    repeat_callbacks=['NN Quadratic']
    if len(repeat_callbacks)>0:
        for repeat_callback in repeat_callbacks:
            assert repeat_callback in callback_names, 'not in callback names'
        repeat_callbacks = [f'Repeat {key}'for key in repeat_callbacks]     
    
    repeat_bool = {f'Repeat {key}':False for key in callback_names}
    # Sets the particulat key to true
    for key in repeat_bool:
        if key in repeat_callbacks:
            repeat_bool[key] = True

    wandb.log(repeat_bool)
    '''
    

    folder = 'Images'
    if not os.path.exists(folder):
        os.mkdir(folder)
        
    # Run setup
    #wandb.run.name = run_name(config)
    pl.seed_everything(config['seed'])
    
    datamodule = Datamodule_selection(datamodule_dict,config['dataset'],config)
    callback_dict = callback_dictionary(datamodule, config,datamodule_dict)
    desired_callbacks = specific_callbacks(callback_dict, config['callbacks'])
    
                        
    # model_function takes in the model module and the config and uses it to instantiate the model
    model = model_function(model_module,config,datamodule)
    wandb_logger.watch(model, log='gradients', log_freq=100) # logs the gradients

    trainer = pl.Trainer(precision=16,fast_dev_run = config['fast_run'],progress_bar_refresh_rate=20,benchmark=True,
                        limit_train_batches = config['training_ratio'],limit_val_batches=config['validation_ratio'],limit_test_batches = config['test_ratio'],
                        max_epochs = config['epochs'],check_val_every_n_epoch = config['val_check'],
                        gpus=1,logger=wandb_logger,checkpoint_callback = False,deterministic =True,callbacks = desired_callbacks)#,auto_lr_find = True)
    '''
    trainer.tune(model)
    # Updates new learning rate from the learning rate finder for the saving of the config as well as the run name
    wandb.config.update({"learning_rate": model.hparams.learning_rate},allow_val_change=True)
    '''
    wandb.run.name = train_run_name(model.name,config)

    trainer.fit(model,datamodule)
    trainer.test(datamodule=datamodule,
            ckpt_path=None)  # uses last-saved model , use test set to call the reliability diagram only at the end of the training process
    
    run.finish()
    return run_path # Output run path for use if I want to perform subsequent evaluation
