from Contrastive_uncertainty.general.datamodules.cifar10_datamodule import CIFAR10DataModule
from Contrastive_uncertainty.general.datamodules.cifar100_datamodule import CIFAR100DataModule
from Contrastive_uncertainty.general.datamodules.fashionmnist_datamodule import FashionMNISTDataModule
from Contrastive_uncertainty.general.datamodules.mnist_datamodule import MNISTDataModule
from Contrastive_uncertainty.general.datamodules.kmnist_datamodule import KMNISTDataModule
from Contrastive_uncertainty.general.datamodules.emnist_datamodule import EMNISTDataModule
from Contrastive_uncertainty.general.datamodules.svhn_datamodule import SVHNDataModule
from Contrastive_uncertainty.general.datamodules.stl10_datamodule import STL10DataModule
from Contrastive_uncertainty.general.datamodules.caltech101_datamodule import Caltech101DataModule
from Contrastive_uncertainty.general.datamodules.caltech256_datamodule import Caltech256DataModule
from Contrastive_uncertainty.general.datamodules.imagenet_datamodule import ImageNetDataModule
from Contrastive_uncertainty.general.datamodules.tinyimagenet_datamodule import TinyImageNetDataModule
from Contrastive_uncertainty.general.datamodules.celeba_datamodule import CelebADataModule
from Contrastive_uncertainty.general.datamodules.cub200_datamodule import CUB200DataModule
from Contrastive_uncertainty.general.datamodules.dogs_datamodule import DogsDataModule
from Contrastive_uncertainty.general.datamodules.widerface_datamodule import WIDERFaceDataModule
from Contrastive_uncertainty.general.datamodules.places365_datamodule import Places365DataModule
from Contrastive_uncertainty.general.datamodules.voc_datamodule import VOCDataModule
from Contrastive_uncertainty.general.datamodules.emnist_datamodule import EMNISTDataModule



from Contrastive_uncertainty.general.datamodules.datamodule_transforms import  Moco2TrainCIFAR10Transforms, Moco2EvalCIFAR10Transforms,Moco2MultiCIFAR10Transforms,\
Moco2TrainCIFAR100Transforms, Moco2EvalCIFAR100Transforms,Moco2MultiCIFAR100Transforms,\
Moco2TrainFashionMNISTTransforms,Moco2EvalFashionMNISTTransforms, Moco2MultiFashionMNISTTransforms, \
Moco2TrainMNISTTransforms, Moco2EvalMNISTTransforms,Moco2MultiMNISTTransforms,\
Moco2TrainSVHNTransforms, Moco2EvalSVHNTransforms,Moco2MultiSVHNTransforms,\
Moco2TrainCaltech101Transforms, Moco2EvalCaltech101Transforms,\
Moco2TrainCaltech256Transforms, Moco2EvalCaltech256Transforms,\
Moco2TrainImageNetTransforms, Moco2EvalImageNetTransforms,\
Moco2TrainTinyImageNetTransforms, Moco2EvalTinyImageNetTransforms,\
Moco2TrainCub200Transforms, Moco2EvalCub200Transforms, \
Moco2TrainDogsTransforms, Moco2EvalDogsTransforms, \
Moco2TrainCelebATransforms, Moco2EvalCelebATransforms,\
Moco2TrainWIDERFaceTransforms, Moco2EvalWIDERFaceTransforms,\
Moco2TrainPlaces365Transforms, Moco2EvalPlaces365Transforms,\
Moco2TrainVOCTransforms, Moco2EvalVOCTransforms,\
Moco2TrainKMNISTTransforms,Moco2EvalKMNISTTransforms,Moco2MultiKMNISTTransforms, \
Moco2TrainSTL10Transforms, Moco2EvalSTL10Transforms, Moco2TrainEMNISTTransforms, Moco2EvalEMNISTTransforms, Moco2MultiEMNISTTransforms


# Nested dict to hold the name of the dataset aswell as the different transforms for the dataset : https://www.programiz.com/python-programming/nested-dictionary

dataset_dict = {'MNIST':{'module':MNISTDataModule,'train_transform':Moco2TrainMNISTTransforms(),'val_transform':Moco2EvalMNISTTransforms(),'test_transform':Moco2EvalMNISTTransforms(),'multi_transform':Moco2MultiMNISTTransforms},
                
                'KMNIST':{'module':KMNISTDataModule,'train_transform':Moco2TrainKMNISTTransforms(),
                'val_transform':Moco2EvalKMNISTTransforms(),'test_transform':Moco2EvalKMNISTTransforms(), 'multi_transform':Moco2MultiKMNISTTransforms},

                'FashionMNIST':{'module':FashionMNISTDataModule,'train_transform':Moco2TrainFashionMNISTTransforms(),
                'val_transform':Moco2EvalFashionMNISTTransforms(),'test_transform':Moco2EvalFashionMNISTTransforms(), 'multi_transform':Moco2MultiFashionMNISTTransforms},

                'EMNIST':{'module':EMNISTDataModule,'train_transform':Moco2TrainEMNISTTransforms(),
                'val_transform':Moco2EvalEMNISTTransforms(),'test_transform':Moco2EvalEMNISTTransforms(), 'multi_transform':Moco2MultiEMNISTTransforms},

                'CIFAR10':{'module':CIFAR10DataModule,'train_transform':Moco2TrainCIFAR10Transforms(),
                'val_transform':Moco2EvalCIFAR10Transforms(),'test_transform':Moco2EvalCIFAR10Transforms(),'multi_transform':Moco2MultiCIFAR10Transforms},
                
                'CIFAR100':{'module':CIFAR100DataModule,'train_transform':Moco2TrainCIFAR100Transforms(),
                'val_transform':Moco2EvalCIFAR100Transforms(),'test_transform':Moco2EvalCIFAR100Transforms(), 'multi_transform':Moco2MultiCIFAR100Transforms},

                'STL10':{'module': STL10DataModule,'train_transform':Moco2TrainSTL10Transforms(),
                'val_transform':Moco2EvalSTL10Transforms(),'test_transform':Moco2EvalSTL10Transforms()},

                'SVHN':{'module':SVHNDataModule,'train_transform':Moco2TrainSVHNTransforms(),
                'val_transform':Moco2EvalSVHNTransforms(),'test_transform':Moco2EvalSVHNTransforms(), 'multi_transform':Moco2MultiSVHNTransforms},
                
                'Caltech101':{'module': Caltech101DataModule,'train_transform':Moco2TrainCaltech101Transforms(),
                'val_transform':Moco2EvalCaltech101Transforms(),'test_transform':Moco2EvalCaltech101Transforms()},

                'Caltech256':{'module': Caltech256DataModule,'train_transform':Moco2TrainCaltech256Transforms(),
                'val_transform':Moco2EvalCaltech256Transforms(),'test_transform':Moco2EvalCaltech256Transforms()},

                'ImageNet':{'module': ImageNetDataModule,'train_transform':Moco2TrainImageNetTransforms(),
                'val_transform':Moco2EvalImageNetTransforms(),'test_transform':Moco2EvalImageNetTransforms()},

                'TinyImageNet':{'module': TinyImageNetDataModule,'train_transform':Moco2TrainTinyImageNetTransforms(),
                'val_transform':Moco2EvalTinyImageNetTransforms(),'test_transform':Moco2EvalTinyImageNetTransforms()},

                'Cub200':{'module': CUB200DataModule,'train_transform':Moco2TrainCub200Transforms(),
                'val_transform':Moco2EvalCub200Transforms(),'test_transform':Moco2EvalCub200Transforms()},

                'Dogs':{'module': DogsDataModule,'train_transform':Moco2TrainDogsTransforms(),
                'val_transform':Moco2EvalDogsTransforms(),'test_transform':Moco2EvalDogsTransforms()},

                'CelebA':{'module': CelebADataModule,'train_transform':Moco2TrainCelebATransforms(),
                'val_transform':Moco2EvalCelebATransforms(),'test_transform':Moco2EvalCelebATransforms()},

                'WIDERFace':{'module': WIDERFaceDataModule,'train_transform':Moco2TrainWIDERFaceTransforms(),
                'val_transform':Moco2EvalWIDERFaceTransforms(),'test_transform':Moco2EvalWIDERFaceTransforms()},

                'Places365':{'module': Places365DataModule,'train_transform':Moco2TrainPlaces365Transforms(),
                'val_transform':Moco2EvalPlaces365Transforms(),'test_transform':Moco2EvalPlaces365Transforms()},

                'VOC':{'module': VOCDataModule,'train_transform':Moco2TrainVOCTransforms(),
                'val_transform':Moco2EvalVOCTransforms(),'test_transform':Moco2EvalVOCTransforms()},
                
                }



'''
OOD_dict = {'MNIST':['FashionMNIST','KMNIST','EMNIST'],
            'FashionMNIST':['MNIST','KMNIST','EMNIST'],
            'KMNIST':['MNIST','FashionMNIST','EMNIST'],
            'EMNIST':['MNIST','FashionMNIST','KMNIST'],
            
            'CIFAR10':['STL10','Caltech101', 'CelebA','WIDERFace','SVHN', 'CIFAR100', 'VOC', 'Places365', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
            'CIFAR100':['STL10','Caltech101', 'CelebA','WIDERFace','SVHN', 'CIFAR10', 'VOC', 'Places365', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],

            'Caltech101':['STL10', 'CelebA','WIDERFace','SVHN', 'CIFAR10','CIFAR100', 'VOC', 'Places365', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
            'Caltech256':['STL10', 'CelebA','WIDERFace','SVHN','Caltech101', 'CIFAR10','CIFAR100', 'VOC', 'Places365', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
            'TinyImageNet':['STL10', 'CelebA','WIDERFace','SVHN','Caltech101', 'Caltech256','CIFAR10','CIFAR100', 'VOC', 'Places365', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
            'Cub200':['STL10', 'CelebA','WIDERFace','SVHN','Caltech101', 'Caltech256','CIFAR10','CIFAR100', 'VOC', 'Places365','TinyImageNet','Dogs', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
            'Dogs':['STL10', 'CelebA','WIDERFace','SVHN','Caltech101', 'Caltech256','CIFAR10','CIFAR100', 'VOC', 'Places365','TinyImageNet','Cub200', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
            #'CIFAR10':['CIFAR100','SVHN'],
            #'CIFAR100':['CIFAR10','SVHN'],
            'SVHN':['CIFAR10','CIFAR100']}
'''



OOD_dict = {'MNIST':['MNIST','FashionMNIST','KMNIST','EMNIST'],
            'FashionMNIST':['MNIST', 'FashionMNIST','KMNIST','EMNIST'],
            'KMNIST':['MNIST', 'FashionMNIST','KMNIST','EMNIST'],
            'EMNIST':['MNIST', 'FashionMNIST','KMNIST','EMNIST'],
            
            'CIFAR10':['STL10', 'CelebA','WIDERFace','SVHN', 'Caltech101','Caltech256','CIFAR10','CIFAR100', 'VOC', 'Places365','TinyImageNet','Cub200','Dogs', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
            'CIFAR100':['STL10', 'CelebA','WIDERFace','SVHN', 'Caltech101','Caltech256','CIFAR10','CIFAR100', 'VOC', 'Places365','TinyImageNet','Cub200','Dogs', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
            'Caltech101':['STL10', 'CelebA','WIDERFace','SVHN', 'Caltech101','Caltech256','CIFAR10','CIFAR100', 'VOC', 'Places365','TinyImageNet','Cub200','Dogs', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],

            
            'Caltech256':['STL10', 'CelebA','WIDERFace','SVHN', 'Caltech101','Caltech256','CIFAR10','CIFAR100', 'VOC', 'Places365','TinyImageNet','Cub200','Dogs', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],            
            'TinyImageNet':['STL10', 'CelebA','WIDERFace','SVHN', 'Caltech101','Caltech256','CIFAR10','CIFAR100', 'VOC', 'Places365','TinyImageNet','Cub200','Dogs', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
            'Cub200':['STL10', 'CelebA','WIDERFace','SVHN', 'Caltech101','Caltech256','CIFAR10','CIFAR100', 'VOC', 'Places365','TinyImageNet','Cub200','Dogs', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
            'Dogs':['STL10', 'CelebA','WIDERFace','SVHN', 'Caltech101','Caltech256','CIFAR10','CIFAR100', 'VOC', 'Places365','TinyImageNet','Cub200','Dogs', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
            #'CIFAR10':['CIFAR100','SVHN'],
            #'CIFAR100':['CIFAR10','SVHN'],
            'SVHN':['STL10', 'CelebA','WIDERFace','SVHN', 'Caltech101','Caltech256','CIFAR10','CIFAR100', 'VOC', 'Places365','TinyImageNet','Cub200','Dogs', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST']}