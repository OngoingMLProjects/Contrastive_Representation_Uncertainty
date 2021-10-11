# Import parameters for different training methods
from Contrastive_uncertainty.experiments.train.experimental_dict import model_dict
from Contrastive_uncertainty.general.run.model_names import model_names_dict

def train(batch_base_dict, batch_trainer_dict):   
    acceptable_single_models = ['Baselines',
    #model_names_dict['CE'],
    #model_names_dict['Moco'],
    model_names_dict['SupCon'],
    #model_names_dict['Centroid_VicReg'],
    #model_names_dict['NNCLR']
    ]

    # Dict for the model name, parameters and specific training loop
    

    num_experiments = len(batch_base_dict)
    for i in range(num_experiments):
        base_dict = batch_base_dict[i]
        trainer_dict = batch_trainer_dict[i]
        # Update the parameters of each model

        # Update the parameters of each model
    
        # iterate through all items of the state dict
        for base_k, base_v in base_dict.items():
            # Iterate through all the model dicts
            for model_k, model_v in model_dict.items():
                # Go through each dict one by one and check if base k in model params
                if base_k in model_dict[model_k]['params']:
                    # update model key with base params
                    model_dict[model_k]['params'][base_k] = base_v
    
    
        # Checks whether base_dict single model is present in the list
        assert base_dict['single_model'] in acceptable_single_models, 'single model response not in list of acceptable responses'
        
        datasets = ['MNIST','FashionMNIST','KMNIST','CIFAR10', 'CIFAR100']
        ood_datasets = [['FashionMNIST'],['MNIST'],['SVHN'],['SVHN']]
        
        # BASELINES
        # Go through all the models in the current dataset and current OOD dataset
        if base_dict['single_model']== 'Baselines':
            # Go through all the models present    
            for model_k, model_v in model_dict.items():
                # Checks if model is present in the acceptable single models
                if model_k in acceptable_single_models:
                    base_params = model_dict[model_k]['params']
                    train_method = model_dict[model_k]['train']
                    model_module = model_dict[model_k]['model_module'] 
                    model_instance_method = model_dict[model_k]['model_instance']
                    model_data_dict = model_dict[model_k]['data_dict']
                    # Try statement to allow the code to continue even if a single run fails
                    train_method(base_params,trainer_dict, model_module, model_instance_method,model_data_dict)
    
        ## SINGLE MODEL
        # Go through a single model on all different datasets
        else:
            # Name of the chosen model
            chosen_model = base_dict['single_model']
            # Specific model dictionary chosen
            model_info = model_dict[chosen_model]
            train_method = model_info['train']
            params = model_info['params']
            model_module = model_info['model_module'] 
            model_instance_method = model_info['model_instance']
            model_data_dict = model_info['data_dict']
            # Loop through the different datasets and OOD datasets and examine if the model is able to train for the task
            for dataset, ood_dataset in zip(datasets, ood_datasets):
                params['dataset'] = dataset
                params['OOD_dataset'] = ood_dataset
                train_method(params, model_module, model_instance_method, model_data_dict)