from re import search

from pytorch_lightning.core import datamodule
from Contrastive_uncertainty.general.callbacks.general_callbacks import  ModelSaving
from Contrastive_uncertainty.general.callbacks.visualisation_callback import Visualisation
from Contrastive_uncertainty.general.callbacks.metrics.metric_callback import MetricLogger, evaluation_metrics, evaltypes

from Contrastive_uncertainty.general.callbacks.ood_callbacks import Mahalanobis_OOD, Class_Mahalanobis_OOD, Mahalanobis_OOD_Fractions
from Contrastive_uncertainty.general.callbacks.odin_callbacks import ODIN
from Contrastive_uncertainty.general.callbacks.nearest_neighbours_callbacks import NearestNeighbours, NearestClassNeighbours,\
NearestNeighbours1DTypicality, NearestNeighboursClass1DTypicality,NearestNeighboursQuadraticClass1DTypicality, OracleNearestNeighboursClass1DTypicality, DifferentKNNClass1DTypicality,\
DifferentKNNMarginal1DTypicality, DifferentKNNMarginalTypicality, DifferentKNNClassTypicality, DifferentKNNQuadraticClass1DTypicality, KNNClassTypicality,\
NearestNeighboursQuadraticMarginal1DTypicality, NearestNeighboursQuadraticClass1DScoresTypicality
from Contrastive_uncertainty.general.callbacks.nearest_neighbours_analysis_callbacks import AnalysisQuadraticClass1DScoresTypicality

from Contrastive_uncertainty.general.callbacks.softmax_probability_callbacks import Max_Softmax_Probability
from Contrastive_uncertainty.general.callbacks.cem_callbacks import ContrastiveExplanationMethod, ContrastiveExplanationDistance
from Contrastive_uncertainty.general.callbacks.gram_ood_callback import Gram_OOD
from Contrastive_uncertainty.general.callbacks.kde_ood_callback import KDE_OOD
from Contrastive_uncertainty.general.callbacks.gradcam.pytorch_grad_cam.cam_visualization_callback import Cam_Visualization
from Contrastive_uncertainty.general.callbacks.total_centroid_similarity_callback import Total_Centroid_KL, Class_Centroid_Radii_Overlap
from Contrastive_uncertainty.general.callbacks.feature_entropy_callback import Feature_Entropy
from Contrastive_uncertainty.general.callbacks.kl_distance_callbacks import KLDistanceOOD
from Contrastive_uncertainty.general.callbacks.confusion_log_probability_callback import ConfusionLogProbability

def train_run_name(model_name, config, group=None):
    run_name = 'Train_' + model_name + '_DS:'+str(config['dataset']) +'_Epochs:'+ str(config['epochs']) + '_seed:' +str(config['seed'])  
    if group is not None:
        run_name = group + '_' + run_name
    return run_name

def eval_run_name(model_name,config, group=None):
    run_name = 'Eval_' + model_name + '_DS:'+str(config['dataset']) +'_Epochs:'+ str(config['epochs']) + '_seed:' +str(config['seed'])   
    if group is not None:
        run_name = group + '_' + run_name
    return run_name

def Datamodule_selection(data_dict, dataset, config):
    # Information regarding the configuration of the data module for the specific task
    datamodule_info =  data_dict[dataset] # Specific module
    Datamodule = datamodule_info['module'](data_dir= './',batch_size = config['bsz'],seed = config['seed'])
    Datamodule.train_transforms = datamodule_info['train_transform']
    Datamodule.val_transforms = datamodule_info['val_transform']
    Datamodule.test_transforms = datamodule_info['test_transform']
    
    Datamodule.prepare_data()
    Datamodule.setup()
    return Datamodule

def callback_dictionary(Datamodule,config,data_dict):
    quick_callback = config['quick_callback']
    # Manually added callbacks
    callback_dict = {'Model_saving':ModelSaving(config['model_saving'],'Models'),
                    'Metrics':MetricLogger(evaluation_metrics,Datamodule,evaltypes, quick_callback=quick_callback),
                    'Visualisation': Visualisation(Datamodule, quick_callback=quick_callback),
                    'Nearest 10 Class Neighbours':NearestClassNeighbours(Datamodule, quick_callback=quick_callback,K=10),
                    'Nearest 5 Class Neighbours':NearestClassNeighbours(Datamodule, quick_callback=quick_callback,K=5),
                    'CAM':Cam_Visualization(Datamodule,quick_callback),
                    f'Total Centroid KL': Total_Centroid_KL(Datamodule, quick_callback=quick_callback),
                    f'Feature Entropy': Feature_Entropy(Datamodule,quick_callback)}

    for ood_dataset in config['OOD_dataset']:
        OOD_Datamodule = Datamodule_selection(data_dict, ood_dataset, config)
        OOD_callback = {                                                
                f'Mahalanobis Distance {ood_dataset}': Mahalanobis_OOD(Datamodule,OOD_Datamodule,quick_callback=quick_callback),
                f'Class Mahalanobis {ood_dataset}': Class_Mahalanobis_OOD(Datamodule,OOD_Datamodule,quick_callback=quick_callback),
                f'Maximum Softmax Probability {ood_dataset}': Max_Softmax_Probability(Datamodule,OOD_Datamodule,quick_callback=quick_callback), # softmax probability baseline (need to train cross entropy models for the case of this baseline)

                f'ODIN {ood_dataset}': ODIN(Datamodule,OOD_Datamodule,quick_callback=quick_callback),

                f'KDE {ood_dataset}': KDE_OOD(Datamodule, OOD_Datamodule, quick_callback= quick_callback),

                'Contrastive Explanation Method':ContrastiveExplanationMethod(Datamodule,quick_callback=quick_callback),
                'Contrastive Explanation Distance':ContrastiveExplanationDistance(Datamodule,quick_callback=quick_callback),

                f'Mahalanobis OOD Fractions {ood_dataset}': Mahalanobis_OOD_Fractions(Datamodule,OOD_Datamodule,quick_callback=quick_callback),
                f'Nearest Neighbours {ood_dataset}': NearestNeighbours(Datamodule,OOD_Datamodule,quick_callback=quick_callback),
                f'Nearest 10 Neighbours 1D Typicality {ood_dataset}':NearestNeighbours1DTypicality(Datamodule,OOD_Datamodule,quick_callback=quick_callback,K=10),
                f'Nearest 10 Neighbours Class 1D Typicality {ood_dataset}':NearestNeighboursClass1DTypicality(Datamodule,OOD_Datamodule,quick_callback=quick_callback,K=10),

                f'Nearest 10 Neighbours Class Quadratic 1D Typicality {ood_dataset}':NearestNeighboursQuadraticClass1DTypicality(Datamodule,OOD_Datamodule,quick_callback=quick_callback,K=10),

                f'Nearest 10 Neighbours Class Typicality {ood_dataset}': KNNClassTypicality(Datamodule, OOD_Datamodule, quick_callback= quick_callback, K=10), # Does not use the different individual dimensions
                
                f'Nearest 5 Neighbours 1D Typicality {ood_dataset}':NearestNeighbours1DTypicality(Datamodule,OOD_Datamodule,quick_callback=quick_callback,K=5),
                f'Nearest 5 Neighbours Class 1D Typicality {ood_dataset}':NearestNeighboursClass1DTypicality(Datamodule,OOD_Datamodule,quick_callback=quick_callback,K=5),
                #f'Oracle Nearest 10 Neighbours Class 1D Typicality {ood_dataset}': OracleNearestNeighboursClass1DTypicality(Datamodule,OOD_Datamodule,quick_callback=quick_callback,K=10),

                # Marginal Quadratic ablation
                f'Nearest 10 Neighbours Marginal Quadratic 1D Typicality {ood_dataset}': NearestNeighboursQuadraticMarginal1DTypicality(Datamodule,OOD_Datamodule,quick_callback=quick_callback,K=10),
                
                # Single data point ablation
                f'Nearest 1 Neighbours Class Quadratic 1D Typicality {ood_dataset}':NearestNeighboursQuadraticClass1DTypicality(Datamodule,OOD_Datamodule,quick_callback=quick_callback,K=1),

                # Obtain the scores for the approach in comparison to the OOD dataset
                f'Nearest 10 Neighbours Class 1D Scores Typicality {ood_dataset}':NearestNeighboursQuadraticClass1DScoresTypicality(Datamodule,OOD_Datamodule,quick_callback=quick_callback,K=10),
                # Used to obtain the scores for indivudal classes fpr the 1D Typicality approach
                f'Nearest 10 Neighbours Analysis Class 1D Scores Typicality {ood_dataset}':AnalysisQuadraticClass1DScoresTypicality(Datamodule,OOD_Datamodule,quick_callback=quick_callback,K=10),
                
                #f'Nearest 10 Neighbours Analysis Worst Class 1D Scores Typicality {ood_dataset}':AnalysisQuadraticWorstClass1DScoresTypicality(Datamodule,OOD_Datamodule,quick_callback=quick_callback,K=10),

                f'KL Distance OOD {ood_dataset}': KLDistanceOOD(Datamodule,OOD_Datamodule,quick_callback=quick_callback),
                f'Confusion Log Probability': ConfusionLogProbability(Datamodule,quick_callback),

                f'Different K Nearest Neighbours Class 1D Typicality {ood_dataset}':DifferentKNNClass1DTypicality(Datamodule,OOD_Datamodule,quick_callback=quick_callback),
                f'Different K Nearest Neighbours Quadratic Class 1D Typicality {ood_dataset}':DifferentKNNQuadraticClass1DTypicality(Datamodule,OOD_Datamodule,quick_callback=quick_callback),
                f'Gram OOD Detection {ood_dataset}': Gram_OOD(Datamodule,OOD_Datamodule,quick_callback=quick_callback)}
                
                # Ablation studies
                #f'Different K Nearest Neighbours Marginal 1D Typicality {ood_dataset}':DifferentKNNMarginal1DTypicality(Datamodule,OOD_Datamodule,quick_callback=quick_callback),
                #f'Different K Nearest Neighbours Marginal Typicality {ood_dataset}':DifferentKNNMarginalTypicality(Datamodule,OOD_Datamodule,quick_callback=quick_callback),
                #f'Different K Nearest Neighbours Class Typicality {ood_dataset}':DifferentKNNClassTypicality(Datamodule,OOD_Datamodule,quick_callback=quick_callback)}
                
        callback_dict.update(OOD_callback)
    
    return callback_dict

def specific_callbacks(callback_dict, names):
    desired_callbacks = []
    # Obtain all the different callback keys
    callback_keys = callback_dict.keys()
    
    # Iterate through all the different names which I specify
    for index, name in enumerate(names):
        for key in callback_keys: # Goes through all the different keys
            if search(name, key): # Checks if name is part of the substring of key 
                desired_callbacks.append(callback_dict[key]) # Add the specific callback
    
    return desired_callbacks
