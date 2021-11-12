#from PIL.Image import Image
import numpy as np
import random
from warnings import warn
from PIL import Image

from numpy.lib.type_check import imag
from Contrastive_uncertainty.general.datamodules.dataset_normalizations import fashionmnist_normalization, mnist_normalization, kmnist_normalization,\
    emnist_normalization
    
import torch

from PIL import ImageFilter
from torchvision import transforms
from torch.utils.data import DataLoader, random_split,  Dataset, Subset


    
# MNIST Coarse labels
MNIST_coarse_labels = np.array([ 0, 2, 1,  4,  3,  4,  0,  2, 1, 3])


# FashionMNIST Coarse labels
# {0: '0 - T-shirt/top', 1: '1 - Trouser', 2: '2 - Pullover', 3: '3 - Dress', 4: '4 - Coat', 5: '5 - Sandal', 6: '6 - Shirt', 7: '7 - Sneaker', 8: '8 - Bag', 9: '9 - Ankle boot'}
# Group together [(t-shirt, shirt), (trouser), (pullover, coat), (dress), (sandal, sneaker,ankle boot),(bag)]
FashionMNIST_coarse_labels = np.array([0,1,2,3,2,4,0,4,5,4])

#KMNIST Coarse labels
# https://github.com/rois-codh/kmnist Grouping decided based on how the prototypes look like from this link
#{0: 'o', 1: 'ki', 2: 'su', 3: 'tsu', 4: 'na', 5: 'ha', 6: 'ma', 7: 'ya', 8: 're', 9: 'wo'}
# Group together [(o,tsu), (ki,ma),(su),(na,ha),(ya,re),(wo)]
KMNIST_coarse_labels = np.array([0, 1, 2, 0, 3, 3, 1, 4, 4, 5])


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x 

class Moco2TrainFashionMNISTTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=9):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            fashionmnist_normalization()
        ])
        self.normalization = fashionmnist_normalization()

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k


class Moco2EvalFashionMNISTTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=9):
        self.test_transform = transforms.Compose([
            transforms.Resize(height + 12),
            transforms.CenterCrop(height),
            transforms.ToTensor(),
            fashionmnist_normalization(),
        ])

        self.normalization = fashionmnist_normalization()

    def __call__(self, inp):
        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k

class Moco2MultiFashionMNISTTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self,num_augmentations,height=9):
        # image augmentation functions
        self.num_augmentations = num_augmentations
        self.multi_transform = transforms.Compose([
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            fashionmnist_normalization(),
        ])

        self.normalization = fashionmnist_normalization()

    def __call__(self, inp):
        multiple_aug_inp = [self.multi_transform(inp) for i in range(self.num_augmentations)]
        return multiple_aug_inp


class Moco2TrainMNISTTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=9):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            mnist_normalization()
        ])
        # Required for ODIN
        self.normalization = mnist_normalization()

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k


class Moco2EvalMNISTTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=9):
        self.test_transform = transforms.Compose([
            transforms.Resize(height + 12),
            transforms.CenterCrop(height),
            transforms.ToTensor(),
            mnist_normalization(),
        ])
        # Required for ODIN
        self.normalization = mnist_normalization()

    def __call__(self, inp):
        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k


class Moco2MultiMNISTTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self,num_augmentations, height=9):
        # image augmentation functions
        self.num_augmentations = num_augmentations
        self.multi_transform = transforms.Compose([
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            mnist_normalization()
        ])
        # Required for ODIN
        self.normalization = mnist_normalization()

    def __call__(self, inp):
        multiple_aug_inp = [self.multi_transform(inp) for i in range(self.num_augmentations)]
        return multiple_aug_inp
        

class Moco2TrainKMNISTTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=9):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            kmnist_normalization()
        ])

        # Required for ODIN
        self.normalization = kmnist_normalization()

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k


class Moco2EvalKMNISTTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=9):
        self.test_transform = transforms.Compose([
            transforms.Resize(height + 12),
            transforms.CenterCrop(height),
            transforms.ToTensor(),
            kmnist_normalization(),
        ])

        self.normalization = kmnist_normalization()

    def __call__(self, inp):
        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k

class Moco2MultiKMNISTTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self,num_augmentations, height=9):
        # image augmentation functions
        self.num_augmentations = num_augmentations
        self.multi_transform = transforms.Compose([
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            kmnist_normalization()
        ])
        self.normalization = kmnist_normalization()

    def __call__(self, inp):
        multiple_aug_inp = [self.multi_transform(inp) for i in range(self.num_augmentations)]
        return multiple_aug_inp
        
class Moco2TrainEMNISTTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=9):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            emnist_normalization()
        ])
        self.normalization = emnist_normalization()

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k



class Moco2EvalEMNISTTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=9):
        self.test_transform = transforms.Compose([
            transforms.Resize(height + 12),
            transforms.CenterCrop(height),
            transforms.ToTensor(),
            emnist_normalization(),
        ])

        self.normalization = emnist_normalization()

    def __call__(self, inp):
        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k
    

class Moco2MultiEMNISTTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self,num_augmentations, height=9):
        # image augmentation functions
        self.num_augmentations = num_augmentations
        self.multi_transform = transforms.Compose([
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            emnist_normalization()
        ])

        self.normalization = emnist_normalization()

    def __call__(self, inp):
        multiple_aug_inp = [self.multi_transform(inp) for i in range(self.num_augmentations)]
        return multiple_aug_inp


    
#https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/18
def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """
    #import ipdb; ipdb.set_trace()
    def __getitem__(self, index):
        
        data, target = cls.__getitem__(self, index)
        return data, target, index
    
    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })

    '''type(name,bases,dict)
    name is the name of the class which corresponds to the __name__ attribute__
    bases: tupe of clases from which corresponds to the __bases__ attribute
    '''

# Subtracts target by 1 to make it so that number of classes go from 0 to 25 rather than 1 to 26
def dataset_with_indices_emnist(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """
    #import ipdb; ipdb.set_trace()
    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        target = target -1 #
        return data, target, index
    
    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    }) 
    '''type(name,bases,dict)
    name is the name of the class which corresponds to the __name__ attribute__
    bases: tupe of clases from which corresponds to the __bases__ attribute
    '''

# Use to apply transforms to the tensordataset  https://stackoverflow.com/questions/55588201/pytorch-transforms-on-tensordataset
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)
        
        # y is from the 1st value to the last value to be able to deal with the coarse values which are present for the task
        

        if len(self.tensors) ==3:
            y = self.tensors[1][index]
            coarse_y = self.tensors[2][index]
            return x, y, coarse_y, index # Added the return of index for the purpose of PCL
            
        else:
            y = self.tensors[1][index]

            return x, y, index # Added the return of index for the purpose of PCL

    def __len__(self):
        return self.tensors[0].size(0)

# Same as tensor dataset but converts into a numpy array beforehand to make it usable
class UpdatedCustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        
        x = Image.fromarray(x.numpy())

        if self.transform:
            x = self.transform(x)
        
        # y is from the 1st value to the last value to be able to deal with the coarse values which are present for the task
        

        if len(self.tensors) ==3:
            y = self.tensors[1][index]
            coarse_y = self.tensors[2][index]
            return x, y, coarse_y, index # Added the return of index for the purpose of PCL
            
        else:
            y = self.tensors[1][index]

            return x, y, index # Added the return of index for the purpose of PCL

    def __len__(self):
        return self.tensors[0].size(0)
