# Import parameters for different training methods
from Contrastive_uncertainty.cross_entropy.config.cross_entropy_params import cross_entropy_hparams
from Contrastive_uncertainty.moco.config.moco_params import moco_hparams
from Contrastive_uncertainty.supcon.config.sup_con_params import sup_con_hparams
from Contrastive_uncertainty.centroid_vicreg.config.centroid_vicreg_params import centroid_vicreg_hparams
from Contrastive_uncertainty.centroid_class_vicreg.config.centroid_class_vicreg_params import centroid_class_vicreg_hparams
from Contrastive_uncertainty.nnclr.config.nnclr_params import nnclr_hparams
from Contrastive_uncertainty.ensemble.config.cross_entropy_ensemble_params import cross_entropy_ensemble_hparams

# Importing the different lightning modules for the baselines
from Contrastive_uncertainty.cross_entropy.models.cross_entropy_module import CrossEntropyModule
from Contrastive_uncertainty.moco.models.moco_module import MocoModule
from Contrastive_uncertainty.supcon.models.sup_con_module import SupConModule
from Contrastive_uncertainty.centroid_vicreg.models.centroid_vicreg_module import CentroidVICRegModule
from Contrastive_uncertainty.centroid_class_vicreg.models.centroid_class_vicreg_module import CentroidClassVICRegModule
from Contrastive_uncertainty.nnclr.models.nnclr_module import NNCLRModule
from Contrastive_uncertainty.ensemble.models.cross_entropy_ensemble_module import CrossEntropyEnsembleModule

# Model instances for the different methods
from Contrastive_uncertainty.cross_entropy.models.cross_entropy_model_instance import ModelInstance as CEModelInstance
from Contrastive_uncertainty.moco.models.moco_model_instance import ModelInstance as MocoModelInstance
from Contrastive_uncertainty.supcon.models.sup_con_model_instance import ModelInstance as SupConModelInstance
from Contrastive_uncertainty.centroid_vicreg.models.centroid_vicreg_model_instance import ModelInstance as CentroidVICRegModelInstance
from Contrastive_uncertainty.centroid_class_vicreg.models.centroid_class_vicreg_model_instance import ModelInstance as CentroidClassVICRegModelInstance
from Contrastive_uncertainty.nnclr.models.nnclr_model_instance import ModelInstance as NNCLRModelInstance
from Contrastive_uncertainty.ensemble.models.cross_entropy_ensemble_model_instance import ModelInstance as CEEnsembleModelInstance

# Import evaluate
from Contrastive_uncertainty.general.run.model_names import model_names_dict
from Contrastive_uncertainty.general.train.train_general import train as general_training
from Contrastive_uncertainty.general.train.train_general_confusion import train as general_confusion_training
from Contrastive_uncertainty.general.train.evaluate_general import evaluation as general_evaluation

from Contrastive_uncertainty.general.train.evaluate_general_confusion import evaluation as general_confusion_evaluation

# Import datamodule info
from Contrastive_uncertainty.general.datamodules.datamodule_dict import dataset_dict as general_dataset_dict, OOD_dict as general_OOD_dict

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

                    model_names_dict['Centroid_VicReg']:{'params':centroid_vicreg_hparams,'model_module':CentroidVICRegModule, 
                    'model_instance':CentroidVICRegModelInstance,
                    'train':general_training,'evaluate':general_evaluation, 
                    'data_dict':general_dataset_dict,'ood_dict':general_OOD_dict},

                    model_names_dict['NNCLR']:{'params':nnclr_hparams,'model_module':NNCLRModule,
                    'model_instance':NNCLRModelInstance,
                    'train': general_training,'evaluate':general_evaluation,
                    'data_dict':general_dataset_dict, 'ood_dict':general_OOD_dict},

                    'CentroidClassVICReg':{'params':centroid_class_vicreg_hparams,'model_module':CentroidClassVICRegModule,
                    'model_instance':CentroidClassVICRegModelInstance,
                    'train': general_training,'evaluate':general_evaluation,
                    'data_dict':general_dataset_dict, 'ood_dict':general_OOD_dict},

                    model_names_dict['CEEnsemble']: {'params':cross_entropy_ensemble_hparams,'model_module':CrossEntropyEnsembleModule, 
                    'model_instance':CEEnsembleModelInstance, 
                    'train':general_confusion_training, 'evaluate':general_confusion_evaluation,
                    'data_dict':general_dataset_dict, 'ood_dict':general_OOD_dict},
                    
    }
