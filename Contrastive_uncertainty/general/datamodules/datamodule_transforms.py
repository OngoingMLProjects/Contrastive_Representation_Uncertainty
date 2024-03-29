#from PIL.Image import Image
import numpy as np
import random
from warnings import warn
from PIL import Image

from numpy.lib.type_check import imag
from Contrastive_uncertainty.general.datamodules.dataset_normalizations import cifar10_normalization,\
    cifar100_normalization, fashionmnist_normalization, mnist_normalization, kmnist_normalization,\
    svhn_normalization, stl10_normalization,\
    caltech101_normalization,caltech256_normalization,\
    imagenet_normalization,tinyimagenet_normalization,\
    cub200_normalization,dogs_normalization, \
    celeba_normalization,widerface_normalization,\
    places365_normalization,voc_normalization,\
    emnist_normalization
    
import torch

from PIL import ImageFilter
from torchvision import transforms
from torch.utils.data import DataLoader, random_split,  Dataset, Subset


# CIFAR100 Coarse labels
CIFAR100_coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                           3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                           6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                           0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                           5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                           16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                           10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                           2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                          16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                          18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    

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

# CIFAR10 Coarse labels
#{0: '0 - airplane', 1: '1 - automobile', 2: '2 - bird', 3: '3 - cat', 4: '4 - deer', 5: '5 - dog', 6: '6 - frog', 7: '7 - horse', 8: '8 - ship', 9: '9 - truck'}
CIFAR10_coarse_labels = np.array([ 0,  2, 3,  5,  6,  5,  4,  6, 1,  2])

# CIFAR100 labels
#{0: '0 - apple', 1: '1 - aquarium_fish', 2: '2 - baby', 3: '3 - bear', 4: '4 - beaver', 5: '5 - bed', 6: '6 - bee', 7: '7 - beetle', 8: '8 - bicycle', 9: '9 - bottle', 10: '10 - bowl', 11: '11 - boy', 12: '12 - bridge', 13: '13 - bus', 14: '14 - butterfly', 15: '15 - camel', 16: '16 - can', 17: '17 - castle', 18: '18 - caterpillar', 19: '19 - cattle', 
# 20: '20 - chair', 21: '21 - chimpanzee', 22: '22 - clock', 23: '23 - cloud', 24: '24 - cockroach', 25: '25 - couch', 26: '26 - crab', 27: '27 - crocodile', 28: '28 - cup', 29: '29 - dinosaur', 30: '30 - dolphin', 31: '31 - elephant', 32: '32 - flatfish', 33: '33 - forest', 34: '34 - fox', 35: '35 - girl', 36: '36 - hamster', 37: '37 - house', 38: '38 - kangaroo', 39: '39 - keyboard', 
# 40: '40 - lamp', 41: '41 - lawn_mower', 42: '42 - leopard', 43: '43 - lion', 44: '44 - lizard', 45: '45 - lobster', 46: '46 - man', 47: '47 - maple_tree', 48: '48 - motorcycle', 49: '49 - mountain', 50: '50 - mouse', 51: '51 - mushroom', 52: '52 - oak_tree', 53: '53 - orange', 54: '54 - orchid', 55: '55 - otter', 56: '56 - palm_tree', 57: '57 - pear', 58: '58 - pickup_truck', 59: '59 - pine_tree', 
# 60: '60 - plain', 61: '61 - plate', 62: '62 - poppy', 63: '63 - porcupine', 64: '64 - possum', 65: '65 - rabbit', 66: '66 - raccoon', 67: '67 - ray', 68: '68 - road', 69: '69 - rocket', 70: '70 - rose', 71: '71 - sea', 72: '72 - seal', 73: '73 - shark', 74: '74 - shrew', 75: '75 - skunk', 76: '76 - skyscraper', 77: '77 - snail', 78: '78 - snake', 79: '79 - spider', 
# 80: '80 - squirrel', 81: '81 - streetcar', 82: '82 - sunflower', 83: '83 - sweet_pepper', 84: '84 - table', 85: '85 - tank', 86: '86 - telephone', 87: '87 - television', 88: '88 - tiger', 89: '89 - tractor', 90: '90 - train', 91: '91 - trout', 92: '92 - tulip', 93: '93 - turtle', 94: '94 - wardrobe', 95: '95 - whale', 96: '96 - willow_tree', 97: '97 - wolf', 98: '98 - woman', 99: '99 - worm'}

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x 

class Moco2TrainCIFAR10Transforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=32):
        # image augmentation functions
        # Used to resize images which are different

        #self.colorize = transforms.Lambda(lambda x: x.repeat(3, 1, 1) ) #  change shape from 2D to 3D
        self.colorize = transforms.Grayscale(num_output_channels=3) #  change shape from 2D to 3D

        self.resize = transforms.Resize(size = (height,height))
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            cifar10_normalization()
        ])
        self.normalization = cifar10_normalization()


    def __call__(self, inp):
        
        if isinstance(inp,Image.Image): 
            if len(inp.getbands())<3:
                #width, height = inp.size
                #inp = inp.view(width, height, 1).expand(-1, -1, 3)
                inp = self.colorize(inp)   
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)

        q = self.train_transform(inp)
        k = self.train_transform(inp)

        return q, k


class Moco2EvalCIFAR10Transforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=32):

        self.colorize = transforms.Grayscale(num_output_channels=3) #  change shape from 2D to 3D

        self.resize = transforms.Resize(size = (height,height))

        self.test_transform = transforms.Compose([
            transforms.Resize(height + 12),
            transforms.CenterCrop(height),
            transforms.ToTensor(),
            cifar10_normalization(),
        ])
        self.normalization = cifar10_normalization()

    def __call__(self, inp):
        if isinstance(inp,Image.Image):
            #https://stackoverflow.com/questions/52962969/number-of-channels-in-pil-pillow-image
            if len(inp.getbands())<3:
                inp = self.colorize(inp)
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)
            
        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k


class Moco2MultiCIFAR10Transforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self,num_augmentations, height=32):
        self.num_augmentations = num_augmentations
        # image augmentation functions

        self.colorize = transforms.Grayscale(num_output_channels=3)

        self.resize = transforms.Resize(height)

        self.multi_transform = transforms.Compose([
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            cifar10_normalization()
        ])
        self.normalization = cifar10_normalization()

    def __call__(self, inp):
        if isinstance(inp,Image.Image):
            if len(inp.getbands())<3:
                inp = self.colorize(inp)
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)

        multiple_aug_inp = [self.multi_transform(inp) for i in range(self.num_augmentations)]
        return multiple_aug_inp


class Moco2TrainCIFAR100Transforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=32):
        # image augmentation functions
        self.colorize = transforms.Grayscale(num_output_channels=3)

        self.resize = transforms.Resize(size = (height,height))
        
        
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            cifar100_normalization()
        ])
        self.normalization = cifar100_normalization()

    def __call__(self, inp):

        if isinstance(inp,Image.Image):
            if len(inp.getbands())<3:
                inp = self.colorize(inp)
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)

        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k


class Moco2EvalCIFAR100Transforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=32):
        self.colorize = transforms.Grayscale(num_output_channels=3)

        self.resize = transforms.Resize(size = (height,height))

        self.test_transform = transforms.Compose([
            transforms.Resize(height + 12),
            transforms.CenterCrop(height),
            transforms.ToTensor(),
            cifar100_normalization(),
        ])
        self.normalization = cifar100_normalization()

    def __call__(self, inp):

        if isinstance(inp,Image.Image):
            if len(inp.getbands())<3:
                inp = self.colorize(inp)
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)

        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k


class Moco2MultiCIFAR100Transforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self,num_augmentations, height=32):
        # image augmentation functions
        self.num_augmentations = num_augmentations
        self.colorize = transforms.Grayscale(num_output_channels=3)

        self.resize = transforms.Resize(size = (height,height))
        self.multi_transform = transforms.Compose([
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            cifar100_normalization()
        ])
        self.normalization = cifar100_normalization()

    def __call__(self, inp):
        if isinstance(inp,Image.Image):
            if len(inp.getbands())<3:
                inp = self.colorize(inp)
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)

        multiple_aug_inp = [self.multi_transform(inp) for i in range(self.num_augmentations)]
        return multiple_aug_inp

class Moco2TrainSVHNTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=32):
        # image augmentation functions
        self.colorize = transforms.Grayscale(num_output_channels=3)

        self.resize = transforms.Resize(size = (height,height))
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            svhn_normalization()
        ])
        self.normalization = svhn_normalization()

    def __call__(self, inp):
        if isinstance(inp,Image.Image):
            if len(inp.getbands())<3:
                inp = self.colorize(inp)
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)

        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k


class Moco2EvalSVHNTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=32):
        self.colorize = transforms.Grayscale(num_output_channels=3)

        self.resize = transforms.Resize(size = (height,height))
        self.test_transform = transforms.Compose([
            transforms.Resize(height + 12),
            transforms.CenterCrop(height),
            transforms.ToTensor(),
            svhn_normalization(),
        ])
        self.normalization = svhn_normalization()

    def __call__(self, inp):
        if isinstance(inp,Image.Image):
            if len(inp.getbands())<3:
                inp = self.colorize(inp)
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)
                
        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k

class Moco2MultiSVHNTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self,num_augmentations, height=32):
        # image augmentation functions
        self.num_augmentations = num_augmentations
        self.colorize = transforms.Grayscale(num_output_channels=3)
        self.resize = transforms.Resize(size = (height,height))
        self.multi_transform = transforms.Compose([
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            svhn_normalization()
        ])
        
        self.normalization = svhn_normalization()

    def __call__(self, inp):
        if isinstance(inp,Image.Image):
            if len(inp.getbands())<3:
                inp = self.colorize(inp)
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)

        multiple_aug_inp = [self.multi_transform(inp) for i in range(self.num_augmentations)]
        return multiple_aug_inp

# Changed STL10 height to the same height as the CIFAR10 datasets
class Moco2TrainSTL10Transforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=32):
        # image augmentation functions
        self.colorize = transforms.Grayscale(num_output_channels=3)
        self.resize = transforms.Resize(size = (height,height))
        self.train_transform = transforms.Compose([
            transforms.Resize(height),
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            stl10_normalization()
        ])

        self.normalization = stl10_normalization()


    def __call__(self, inp):
        if isinstance(inp,Image.Image):
            if len(inp.getbands())<3:
                inp = self.colorize(inp)
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)

        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k

class Moco2EvalSTL10Transforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=32):
        self.colorize = transforms.Grayscale(num_output_channels=3)
        self.resize = transforms.Resize(size = (height,height))
        self.test_transform = transforms.Compose([
            transforms.Resize(height + 12),
            transforms.CenterCrop(height),
            transforms.ToTensor(),
            stl10_normalization(),
        ])

        self.normalization = stl10_normalization()

    def __call__(self, inp):
        if isinstance(inp,Image.Image):
            if len(inp.getbands())<3:
                inp = self.colorize(inp)
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)

        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k


class Moco2TrainCaltech101Transforms:
    
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self, height=32):
        # image augmentation functions
        self.colorize = transforms.Grayscale(num_output_channels=3)

        self.resize = transforms.Resize(size = (height,height))
        self.train_transform = transforms.Compose([
            transforms.Resize(height),
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            caltech101_normalization()
        ])

        self.normalization = caltech101_normalization()

    def __call__(self, inp):
        if isinstance(inp,Image.Image):
            if len(inp.getbands())<3:
                inp = self.colorize(inp)
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)

        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k
    

class Moco2EvalCaltech101Transforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=32):
        self.colorize = transforms.Grayscale(num_output_channels=3) #  change shape from 2D to 3D
        self.resize = transforms.Resize(size = (height,height))
        self.test_transform = transforms.Compose([
            transforms.Resize(height + 12),
            transforms.CenterCrop(height),
            transforms.ToTensor(),
            caltech101_normalization(),
        ])
        self.normalization = caltech101_normalization()


    def __call__(self, inp):
        if isinstance(inp,Image.Image):
            if len(inp.getbands())<3:
                inp = self.colorize(inp)
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)

        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k


class Moco2TrainCaltech256Transforms:
    
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self, height=32):
        # image augmentation functions
        self.colorize = transforms.Grayscale(num_output_channels=3)

        self.resize = transforms.Resize(size = (height,height))
        self.train_transform = transforms.Compose([
            transforms.Resize(height),
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            caltech256_normalization()
        ])
        self.normalization = caltech256_normalization()

    def __call__(self, inp):
        if isinstance(inp,Image.Image):
            if len(inp.getbands())<3:
                inp = self.colorize(inp)
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)

        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k
    

class Moco2EvalCaltech256Transforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=32):
        self.colorize = transforms.Grayscale(num_output_channels=3) #  change shape from 2D to 3D
        self.resize = transforms.Resize(size = (height,height))
        self.test_transform = transforms.Compose([
            transforms.Resize(height + 12),
            transforms.CenterCrop(height),
            transforms.ToTensor(),
            caltech256_normalization(),
        ])
        self.normalization = caltech256_normalization()

    def __call__(self, inp):
        if isinstance(inp,Image.Image):
            if len(inp.getbands())<3:
                inp = self.colorize(inp)
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)

        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k




class Moco2TrainImageNetTransforms:
    
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self, height=32):
        # image augmentation functions
        self.colorize = transforms.Grayscale(num_output_channels=3)

        self.resize = transforms.Resize(size = (height,height))
        self.train_transform = transforms.Compose([
            transforms.Resize(height),
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            imagenet_normalization()
        ])
        self.normalization = imagenet_normalization()

    def __call__(self, inp):
        if isinstance(inp,Image.Image):
            if len(inp.getbands())<3:
                inp = self.colorize(inp)
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)

        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k
    

class Moco2EvalImageNetTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=32):
        self.colorize = transforms.Grayscale(num_output_channels=3) #  change shape from 2D to 3D
        self.resize = transforms.Resize(size = (height,height))
        self.test_transform = transforms.Compose([
            transforms.Resize(height + 12),
            transforms.CenterCrop(height),
            transforms.ToTensor(),
            imagenet_normalization(),
        ])

        self.normalization = imagenet_normalization()

    def __call__(self, inp):
        if isinstance(inp,Image.Image):
            if len(inp.getbands())<3:
                inp = self.colorize(inp)
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)

        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k


class Moco2TrainTinyImageNetTransforms:
    
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self, height=32):
        # image augmentation functions
        self.colorize = transforms.Grayscale(num_output_channels=3)

        self.resize = transforms.Resize(size = (height,height))
        self.train_transform = transforms.Compose([
            transforms.Resize(height),
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            tinyimagenet_normalization()
        ])

        self.normalization = tinyimagenet_normalization()

    def __call__(self, inp):
        if isinstance(inp,Image.Image):
            if len(inp.getbands())<3:
                inp = self.colorize(inp)
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)

        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k
    

class Moco2EvalTinyImageNetTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=32):
        self.colorize = transforms.Grayscale(num_output_channels=3) #  change shape from 2D to 3D
        self.resize = transforms.Resize(size = (height,height))
        self.test_transform = transforms.Compose([
            transforms.Resize(height + 12),
            transforms.CenterCrop(height),
            transforms.ToTensor(),
            tinyimagenet_normalization(),
        ])

        self.normalization = tinyimagenet_normalization()

    def __call__(self, inp):
        if isinstance(inp,Image.Image):
            if len(inp.getbands())<3:
                inp = self.colorize(inp)
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)

        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k


class Moco2TrainCub200Transforms:
    
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self, height=32):
        # image augmentation functions
        self.colorize = transforms.Grayscale(num_output_channels=3)

        self.resize = transforms.Resize(size = (height,height))
        self.train_transform = transforms.Compose([
            transforms.Resize(height),
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            cub200_normalization()
        ])

        self.normalization = cub200_normalization()

    def __call__(self, inp):
        if isinstance(inp,Image.Image):
            if len(inp.getbands())<3:
                inp = self.colorize(inp)
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)

        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k
    

class Moco2EvalCub200Transforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=32):
        self.colorize = transforms.Grayscale(num_output_channels=3) #  change shape from 2D to 3D
        self.resize = transforms.Resize(size = (height,height))
        self.test_transform = transforms.Compose([
            transforms.Resize(height + 12),
            transforms.CenterCrop(height),
            transforms.ToTensor(),
            cub200_normalization(),
        ])

        self.normalization = cub200_normalization()

    def __call__(self, inp):
        if isinstance(inp,Image.Image):
            if len(inp.getbands())<3:
                inp = self.colorize(inp)
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)

        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k


class Moco2TrainDogsTransforms:
    
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self, height=32):
        # image augmentation functions
        self.colorize = transforms.Grayscale(num_output_channels=3)

        self.resize = transforms.Resize(size = (height,height))
        self.train_transform = transforms.Compose([
            transforms.Resize(height),
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            dogs_normalization()
        ])

        self.normalization = dogs_normalization()

    def __call__(self, inp):
        if isinstance(inp,Image.Image):
            if len(inp.getbands())<3:
                inp = self.colorize(inp)
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)

        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k
    

class Moco2EvalDogsTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=32):
        self.colorize = transforms.Grayscale(num_output_channels=3) #  change shape from 2D to 3D
        self.resize = transforms.Resize(size = (height,height))
        self.test_transform = transforms.Compose([
            transforms.Resize(height + 12),
            transforms.CenterCrop(height),
            transforms.ToTensor(),
            dogs_normalization(),
        ])
        self.normalization = dogs_normalization()

    def __call__(self, inp):
        if isinstance(inp,Image.Image):
            if len(inp.getbands())<3:
                inp = self.colorize(inp)
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)

        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k


class Moco2TrainCelebATransforms:
    
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self, height=32):
        # image augmentation functions
        self.colorize = transforms.Grayscale(num_output_channels=3) #  change shape from 2D to 3D
        self.resize = transforms.Resize(size = (height,height))
        self.train_transform = transforms.Compose([
            transforms.Resize(height),
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            celeba_normalization()
        ])

        self.normalization = celeba_normalization()

    def __call__(self, inp):
        if isinstance(inp,Image.Image):
            if len(inp.getbands())<3:
                inp = self.colorize(inp)
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)

        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k
    

class Moco2EvalCelebATransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=32):
        self.colorize = transforms.Grayscale(num_output_channels=3) #  change shape from 2D to 3D
        self.resize = transforms.Resize(size = (height,height))
        self.test_transform = transforms.Compose([
            transforms.Resize(height + 12),
            transforms.CenterCrop(height),
            transforms.ToTensor(),
            celeba_normalization(),
        ])

        self.normalization = celeba_normalization()

    def __call__(self, inp):
        if isinstance(inp,Image.Image):
            if len(inp.getbands())<3:
                inp = self.colorize(inp)
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)

        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k
    

class Moco2TrainWIDERFaceTransforms:
    
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self, height=32):
        # image augmentation functions
        self.colorize = transforms.Grayscale(num_output_channels=3) #  change shape from 2D to 3D
        self.resize = transforms.Resize(size = (height,height))
        self.train_transform = transforms.Compose([
            transforms.Resize(height),
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            widerface_normalization(),
        ])


        self.normalization = widerface_normalization()

    def __call__(self, inp):
        if isinstance(inp,Image.Image):
            if len(inp.getbands())<3:
                inp = self.colorize(inp)
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)

        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k
    

class Moco2EvalWIDERFaceTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=32):
        self.colorize = transforms.Grayscale(num_output_channels=3) #  change shape from 2D to 3D
        self.resize = transforms.Resize(size = (height,height))
        self.test_transform = transforms.Compose([
            transforms.Resize(height + 12),
            transforms.CenterCrop(height),
            transforms.ToTensor(),
            widerface_normalization(),
        ])
        self.normalization = widerface_normalization()

    def __call__(self, inp):
        if isinstance(inp,Image.Image):
            if len(inp.getbands())<3:
                inp = self.colorize(inp)
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)

        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k

class Moco2TrainPlaces365Transforms:
    
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self, height=32):
        # image augmentation functions
        self.colorize = transforms.Grayscale(num_output_channels=3) #  change shape from 2D to 3D
        self.resize = transforms.Resize(size = (height,height))
        self.train_transform = transforms.Compose([
            transforms.Resize(height),
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            places365_normalization(),
        ])

        self.normalization = places365_normalization()

    def __call__(self, inp):
        if isinstance(inp,Image.Image):
            if len(inp.getbands())<3:
                inp = self.colorize(inp)
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)

        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k
    

class Moco2EvalPlaces365Transforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=32):
        self.colorize = transforms.Grayscale(num_output_channels=3) #  change shape from 2D to 3D
        self.resize = transforms.Resize(size = (height,height))
        self.test_transform = transforms.Compose([
            transforms.Resize(height + 12),
            transforms.CenterCrop(height),
            transforms.ToTensor(),
            places365_normalization(),
        ])

        self.normalization = places365_normalization()

    def __call__(self, inp):
        if isinstance(inp,Image.Image):
            if len(inp.getbands())<3:
                inp = self.colorize(inp)
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)

        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k

class Moco2TrainVOCTransforms:
    
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self, height=32):
        # image augmentation functions
        self.colorize = transforms.Grayscale(num_output_channels=3) #  change shape from 2D to 3D
        self.resize = transforms.Resize(size = (height,height))
        self.train_transform = transforms.Compose([
            transforms.Resize(height),
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            voc_normalization(),
        ])
        self.normalization = voc_normalization()

    def __call__(self, inp):
        if isinstance(inp,Image.Image):
            if len(inp.getbands())<3:
                inp = self.colorize(inp)
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)

        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k
    

class Moco2EvalVOCTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=32):
        self.colorize = transforms.Grayscale(num_output_channels=3) #  change shape from 2D to 3D
        self.resize = transforms.Resize(size = (height,height))
        self.test_transform = transforms.Compose([
            transforms.Resize(height + 12),
            transforms.CenterCrop(height),
            transforms.ToTensor(),
            voc_normalization(),
        ])
        self.normalization = voc_normalization()

    def __call__(self, inp):
        if isinstance(inp,Image.Image):
            if len(inp.getbands())<3:
                inp = self.colorize(inp)
            if inp.size[0] !=32:
                inp = self.resize(inp)
        elif isinstance(inp, torch.Tensor):
            if len(inp.shape)<3:
                inp = self.colorize(inp)
            if inp.shape[1] !=32:
                inp = self.resize(inp)

        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k

class Moco2TrainFashionMNISTTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=28):
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
    def __init__(self, height=28):
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
    def __init__(self,num_augmentations,height=28):
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
    def __init__(self, height=28):
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
    def __init__(self, height=28):
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
    def __init__(self,num_augmentations, height=28):
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
    def __init__(self, height=28):
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
    def __init__(self, height=28):
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
    def __init__(self,num_augmentations, height=28):
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
    def __init__(self, height=28):
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
    def __init__(self, height=28):
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
    def __init__(self,num_augmentations, height=28):
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
    #
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

#https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/18
def dataset_with_indices_SVHN(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """
    #
    def __getitem__(self, index):
        
        data, target = cls.__getitem__(self, index)
        return data, torch.tensor(target), index
    
    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })

    '''type(name,bases,dict)
    name is the name of the class which corresponds to the __name__ attribute__
    bases: tupe of clases from which corresponds to the __bases__ attribute
    '''

'''
#https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/18
def dataset_with_indices_hierarchy(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """
    
    def __getitem__(self, index):
        #
        data, target = cls.__getitem__(self, index)
        coarse_target = coarse_labels[target]
        return data, target, coarse_target, index
    
    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    }) 
    
'''

#https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/18
def dataset_with_indices_hierarchy(cls, coarse_labels_map):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """
    
    def __getitem__(self, index):
        #
        data, target = cls.__getitem__(self, index)
        coarse_target = coarse_labels_map[target]
        return data, target, coarse_target, index
    
    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


# Subtracts target by 1 to make it so that number of classes go from 0 to 25 rather than 1 to 26
def dataset_with_indices_emnist(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """
    #
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
