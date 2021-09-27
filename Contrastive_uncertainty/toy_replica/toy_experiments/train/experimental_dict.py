# Import parameters for different training methods
from Contrastive_uncertainty.toy_replica.cross_entropy.config.cross_entropy_params import cross_entropy_hparams
from Contrastive_uncertainty.toy_replica.moco.config.moco_params import moco_hparams
from Contrastive_uncertainty.toy_replica.centroid_vicreg.config.centroid_vicreg_params import centroid_vicreg_hparams
from Contrastive_uncertainty.toy_replica.centroid_class_vicreg.config.centroid_class_vicreg_params import centroid_class_vicreg_hparams
from Contrastive_uncertainty.toy_replica.nnclr.config.nnclr_params import nnclr_hparams

#from Contrastive_uncertainty.toy_replica.nnclr.config.nnclr_params import 
# Importing the different lightning modules for the baselines
from Contrastive_uncertainty.toy_replica.cross_entropy.models.cross_entropy_module import CrossEntropyToy
from Contrastive_uncertainty.toy_replica.moco.models.moco_module import MocoToy
from Contrastive_uncertainty.toy_replica.centroid_vicreg.models.centroid_vicreg_module import CentroidVICRegToy
from Contrastive_uncertainty.toy_replica.centroid_class_vicreg.models.centroid_class_vicreg_module import CentroidClassVICRegToy
from Contrastive_uncertainty.toy_replica.nnclr.models.nnclr_module import NNCLRToy

# Model instances for the different methods
from Contrastive_uncertainty.toy_replica.cross_entropy.models.cross_entropy_model_instance import ModelInstance as CEModelInstance
from Contrastive_uncertainty.toy_replica.moco.models.moco_model_instance import ModelInstance as MocoModelInstance
from Contrastive_uncertainty.toy_replica.centroid_vicreg.models.centroid_vicreg_model_instance import ModelInstance as CentroidVICRegModelInstance
from Contrastive_uncertainty.toy_replica.centroid_class_vicreg.models.centroid_class_vicreg_model_instance import ModelInstance as CentroidClassVICRegModelInstance
from Contrastive_uncertainty.toy_replica.nnclr.models.nnclr_model_instance import ModelInstance as NNCLRModelInstance

# Import training methods 
from Contrastive_uncertainty.general.run.general_run_setup import model_names_dict

from Contrastive_uncertainty.general.train.train_general import train as general_training
from Contrastive_uncertainty.general.train.evaluate_general import evaluation as general_evaluation
from Contrastive_uncertainty.toy_replica.toy_general.datamodules.datamodule_dict import dataset_dict as general_dataset_dict, OOD_dict as general_OOD_dict


model_dict = {model_names_dict['CE']:{'params':cross_entropy_hparams,'model_module':CrossEntropyToy, 
                    'model_instance':CEModelInstance, 
                    'train': general_training,'evaluate':general_evaluation, 
                    'data_dict':general_dataset_dict, 'ood_dict':general_OOD_dict},

                    model_names_dict['Moco']:{'params':moco_hparams,'model_module':MocoToy, 
                    'model_instance':MocoModelInstance, 
                    'train': general_training,'evaluate':general_evaluation,
                    'data_dict':general_dataset_dict, 'ood_dict':general_OOD_dict},

                    model_names_dict['Centroid_VicReg']:{'params':centroid_vicreg_hparams,'model_module':CentroidVICRegToy,
                    'model_instance':CentroidVICRegModelInstance,
                    'train': general_training,'evaluate':general_evaluation,
                    'data_dict':general_dataset_dict, 'ood_dict':general_OOD_dict},

                    model_names_dict['NNCLR']:{'params':nnclr_hparams,'model_module':NNCLRToy,
                    'model_instance':NNCLRModelInstance,
                    'train': general_training,'evaluate':general_evaluation,
                    'data_dict':general_dataset_dict, 'ood_dict':general_OOD_dict},

                    'CentroidClassVICReg':{'params':centroid_class_vicreg_hparams,'model_module':CentroidClassVICRegToy,
                    'model_instance':CentroidClassVICRegModelInstance,
                    'train': general_training,'evaluate':general_evaluation,
                    'data_dict':general_dataset_dict, 'ood_dict':general_OOD_dict}
    }
