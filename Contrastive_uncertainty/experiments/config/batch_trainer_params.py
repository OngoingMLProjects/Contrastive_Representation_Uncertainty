from pytorch_lightning import callbacks

trainer_hparams_1 = dict(
# Miscellaneous arguments
seed = 26,
epochs = 0,
bsz = 128, #32,#64,
OOD_dataset = ['STL10', 'CelebA','WIDERFace','SVHN', 'Caltech101','Caltech256','CIFAR10','CIFAR100', 'VOC', 'Places365','TinyImageNet','Cub200','Dogs', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
#OOD_dataset = ['MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
#OOD_dataset = ['KMNIST'],
#OOD_dataset =['MNIST','FashionMNIST','KMNIST','EMNIST','Places365','VOC'],
# Trainer configurations
fast_run = False,
quick_callback = False,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 1,  # evaluation, Moco_training

callbacks = ['Mahalanobis Distance','Nearest 10 Neighbours Class Quadratic 1D Typicality'],
)

trainer_hparams_2 = dict(
# Miscellaneous arguments
seed = 26,
epochs = 0,
bsz = 128, #32,#64,
OOD_dataset = ['STL10', 'CelebA','WIDERFace','SVHN', 'Caltech101','Caltech256','CIFAR10','CIFAR100', 'VOC', 'Places365','TinyImageNet','Cub200','Dogs', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
#OOD_dataset = ['MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
#OOD_dataset = ['KMNIST'],
#OOD_dataset =['MNIST','FashionMNIST','KMNIST','EMNIST','Places365','VOC'],
# Trainer configurations
fast_run = False,
quick_callback = False,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 1,  # evaluation, Moco_training

callbacks = ['Mahalanobis Distance','Nearest 10 Neighbours Class Quadratic 1D Typicality'],
)

batch_trainer_hparams = [trainer_hparams_1,trainer_hparams_2]