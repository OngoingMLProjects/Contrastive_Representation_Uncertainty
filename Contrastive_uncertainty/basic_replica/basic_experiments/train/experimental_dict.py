# Import parameters for different training methods
from Contrastive_uncertainty.basic_replica.cross_entropy.config.cross_entropy_params import cross_entropy_hparams
from Contrastive_uncertainty.basic_replica.moco.config.moco_params import moco_hparams
from Contrastive_uncertainty.basic_replica.supcon.config.sup_con_params import sup_con_hparams


# Importing the different lightning modules for the baselines
from Contrastive_uncertainty.basic_replica.cross_entropy.models.cross_entropy_module import CrossEntropyModule
from Contrastive_uncertainty.basic_replica.moco.models.moco_module import MocoModule
from Contrastive_uncertainty.basic_replica.supcon.models.sup_con_module import SupConModule


# Model instances for the different methods
from Contrastive_uncertainty.basic_replica.cross_entropy.models.cross_entropy_model_instance import ModelInstance as CEModelInstance
from Contrastive_uncertainty.basic_replica.moco.models.moco_model_instance import ModelInstance as MocoModelInstance
from Contrastive_uncertainty.basic_replica.supcon.models.sup_con_model_instance import ModelInstance as SupConModelInstance


# Import evaluate
from Contrastive_uncertainty.general.run.model_names import model_names_dict
from Contrastive_uncertainty.general.train.train_general import train as general_training
from Contrastive_uncertainty.general.train.evaluate_general import evaluation as general_evaluation

# Import datamodule info
from Contrastive_uncertainty.basic_replica.basic_general.datamodules.datamodule_dict import dataset_dict as general_dataset_dict, OOD_dict as general_OOD_dict

model_dict = {model_names_dict['CE']:{'params':cross_entropy_hparams,'model_module':CrossEntropyModule, 
                    'model_instance':CEModelInstance, 
                    'train':general_training,'evaluate':general_evaluation,
                    'data_dict':general_dataset_dict, 'ood_dict':general_OOD_dict},  

                    model_names_dict['Moco']:{'params':moco_hparams,'model_module':MocoModule, 
                    'model_instance':MocoModelInstance, 
                    'train':general_training,'evaluate':general_evaluation,
                    'data_dict':general_dataset_dict, 'ood_dict':general_OOD_dict},

                    model_names_dict['SupCon']:{'params':sup_con_hparams,'model_module':SupConModule, 
                    'model_instance':SupConModelInstance, 
                    'train':general_training,'evaluate':general_evaluation,
                    'data_dict':general_dataset_dict, 'ood_dict':general_OOD_dict},             
    }