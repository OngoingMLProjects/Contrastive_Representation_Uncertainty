from warnings import warn

from torchvision import transforms

def imagenet_normalization():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalize

def cifar10_normalization():
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    return normalize
#https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151 - Normalisation obtained from here
def cifar100_normalization():
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
                                    
    return normalize

def stl10_normalization():
    normalize = transforms.Normalize(mean=(0.43, 0.42, 0.39), std=(0.27, 0.26, 0.27))
    return normalize

def fashionmnist_normalization():
    normalize = transforms.Normalize(mean =(0.2861,),std =(0.3530,))
    return normalize

def mnist_normalization():
    normalize = transforms.Normalize(mean =(0.1307,),std =(0.3081,))
    return normalize

def kmnist_normalization():
    normalize = transforms.Normalize(mean =(0.1918,),std =(0.3385,))
    return normalize

def emnist_normalization():
    normalize = transforms.Normalize(mean =(0.1344,),std =(1.0520,))
    return normalize

def svhn_normalization():
    normalize = transforms.Normalize(mean = [0.4380,0.440,0.4730], std = [0.1751,0.1771,0.1744])
    return normalize

# Obtained from https://pretagteam.com/question/pytorch-lightning-get-models-output-on-full-train-data-during-training
def caltech101_normalization():
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    return normalize


# calculated manually using the procedure described in https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/6
def caltech256_normalization():
    normalize = transforms.Normalize(mean = [0.5522, 0.5334, 0.5043], std = [0.2100, 0.2092, 0.2123])
    return normalize

# calculated manually using the procedure described in https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/6
def celeba_normalization():
    normalize = transforms.Normalize(mean = [0.5060, 0.4254, 0.3828], std = [0.2650, 0.2441, 0.2402])
    return normalize


# calculated manually using the procedure described in https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/6
def widerface_normalization():
    normalize = transforms.Normalize(mean = [0.4696, 0.4345, 0.4078], std = [0.2488, 0.2387, 0.2390])
    return normalize


# Currently have not calculated it
def places365_normalization():
    normalize = transforms.Normalize(mean = [0.4696, 0.4345, 0.4078], std = [0.2488, 0.2387, 0.2390])
    return normalize


# calculated manually using the procedure described in https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/6
def voc_normalization():
    normalize = transforms.Normalize(mean = [0.3748, 0.3472, 0.3164], std = [0.2261, 0.2138, 0.2083])
    return normalize

# https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2#:~:text=Using%20the%20mean%20and%20std,mean%20and%20std%20is%20recommended.
def imagenet_normalization():
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    return normalize