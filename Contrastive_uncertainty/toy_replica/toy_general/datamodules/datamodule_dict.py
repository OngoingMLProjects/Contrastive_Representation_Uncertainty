from Contrastive_uncertainty.toy_replica.toy_general.datamodules.two_moons_datamodule import TwoMoonsDataModule
from Contrastive_uncertainty.toy_replica.toy_general.datamodules.blobs_datamodule import BlobsDataModule
from Contrastive_uncertainty.toy_replica.toy_general.datamodules.toy_transforms import ToyTrainTwoMoonsTransforms, ToyEvalTwoMoonsTransforms,\
                                                                        ToyTrainBlobsTransforms, ToyEvalBlobsTransforms
                                                                    

# Nested dict to hold the name of the dataset aswell as the different transforms for the dataset : https://www.programiz.com/python-programming/nested-dictionary

# Changed the multi_transform to not use () as I will provide it an input which controls the number of data augmentations used
dataset_dict = {'TwoMoons':{'module':TwoMoonsDataModule,'train_transform':ToyTrainTwoMoonsTransforms(),'val_transform':ToyEvalTwoMoonsTransforms(),'test_transform':ToyEvalTwoMoonsTransforms()},
                
                'Blobs':{'module':BlobsDataModule,'train_transform':ToyTrainBlobsTransforms(),'val_transform':ToyEvalBlobsTransforms(),'test_transform':ToyEvalBlobsTransforms()}}

OOD_dict = {'TwoMoons':['Blobs'],
            'Blobs':['TwoMoons']}