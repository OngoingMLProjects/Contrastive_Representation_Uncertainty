from Contrastive_uncertainty.general.datamodules.fashionmnist_datamodule import FashionMNISTDataModule
from Contrastive_uncertainty.general.datamodules.mnist_datamodule import MNISTDataModule
from Contrastive_uncertainty.general.datamodules.kmnist_datamodule import KMNISTDataModule
from Contrastive_uncertainty.general.datamodules.emnist_datamodule import EMNISTDataModule




from Contrastive_uncertainty.basic_replica.basic_general.datamodules.datamodule_transforms import  Moco2TrainFashionMNISTTransforms,Moco2EvalFashionMNISTTransforms, Moco2MultiFashionMNISTTransforms, \
Moco2TrainMNISTTransforms, Moco2EvalMNISTTransforms,Moco2MultiMNISTTransforms,\
Moco2TrainKMNISTTransforms,Moco2EvalKMNISTTransforms,Moco2MultiKMNISTTransforms, \
Moco2TrainEMNISTTransforms, Moco2EvalEMNISTTransforms, Moco2MultiEMNISTTransforms


# Nested dict to hold the name of the dataset aswell as the different transforms for the dataset : https://www.programiz.com/python-programming/nested-dictionary

dataset_dict = {'MNIST':{'module':MNISTDataModule,'train_transform':Moco2TrainMNISTTransforms(),'val_transform':Moco2EvalMNISTTransforms(),'test_transform':Moco2EvalMNISTTransforms(),'multi_transform':Moco2MultiMNISTTransforms},
                
                'KMNIST':{'module':KMNISTDataModule,'train_transform':Moco2TrainKMNISTTransforms(),
                'val_transform':Moco2EvalKMNISTTransforms(),'test_transform':Moco2EvalKMNISTTransforms(), 'multi_transform':Moco2MultiKMNISTTransforms},

                'FashionMNIST':{'module':FashionMNISTDataModule,'train_transform':Moco2TrainFashionMNISTTransforms(),
                'val_transform':Moco2EvalFashionMNISTTransforms(),'test_transform':Moco2EvalFashionMNISTTransforms(), 'multi_transform':Moco2MultiFashionMNISTTransforms},

                'EMNIST':{'module':EMNISTDataModule,'train_transform':Moco2TrainEMNISTTransforms(),
                'val_transform':Moco2EvalEMNISTTransforms(),'test_transform':Moco2EvalEMNISTTransforms(), 'multi_transform':Moco2MultiEMNISTTransforms},

                }

OOD_dict = {'MNIST':['MNIST','FashionMNIST','KMNIST','EMNIST'],
            'FashionMNIST':['MNIST', 'FashionMNIST','KMNIST','EMNIST'],
            'KMNIST':['MNIST', 'FashionMNIST','KMNIST','EMNIST'],
            'EMNIST':['MNIST', 'FashionMNIST','KMNIST','EMNIST']}