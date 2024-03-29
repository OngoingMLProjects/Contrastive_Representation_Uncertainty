import random
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split,  Dataset, Subset
from Contrastive_uncertainty.general.datamodules.dataset_normalizations import twomoons_normalization, blobs_normalization

class ToyTrainTwoMoonsTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            transforms.RandomApply([GaussianNoise([0, 0.00001])], p=0.5),
        ])
        # Used as placeholder for ODIN
        self.normalization = twomoons_normalization()

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k


class ToyEvalTwoMoonsTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self):
        self.test_transform = transforms.Compose([
        ])
        # Used as placeholder for ODIN
        self.normalization = twomoons_normalization()
    def __call__(self, inp):
        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k


class GaussianNoise(object):
    """Gaussian Noise augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=(0.01, 0.02)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x  + (sigma*torch.randn_like(x))  # adding zero mean gaussian noise
        #x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def Normalize(x, mean, std):
    x = (x - mean) / std
    return x



class ToyTrainBlobsTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            transforms.RandomApply([GaussianNoise([.1, 2.])], p=0.5),
        ])
        # Used as placeholder for ODIN
        self.normalization = blobs_normalization()

    def __call__(self, inp):
        
        q = self.train_transform(inp)
        #print('pre normalised q',q)
        #q = Normalize(q,mean,std)
        #print('post normalised q',q)
        k = self.train_transform(inp)
        #k = Normalize(k,mean,std)
        return q, k


class ToyEvalBlobsTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self):
        self.test_transform = transforms.Compose([
        ])

        # Used as placeholder for ODIN
        self.normalization = blobs_normalization()

    def __call__(self, inp):
        q = self.test_transform(inp)
        #q = Normalize(q, mean, std)
        k = self.test_transform(inp)
        #k = Normalize(k, mean, std)
        return q, k


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
        
        # Takes into account a hierarchy
        if len(self.tensors) ==3:
            y = self.tensors[1][index]
            coarse_y = self.tensors[2][index]
            return x, y, coarse_y, index # Added the return of index for the purpose of PCL
            
        else:
            y = self.tensors[1][index]

            return x, y, index # Added the return of index for the purpose of PCL

    def __len__(self):
        return self.tensors[0].size(0)