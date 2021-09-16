from pytorch_lightning import callbacks
from Contrastive_uncertainty.general.datamodules.datamodule_dict import OOD_dict

base_hparams = dict(
# Optimizer parameters in common
#optimizer = 'adam', #'adam',
#learning_rate= 3e-4, #3e-4,

optimizer = 'sgd', #'adam',
learning_rate= 3e-2, #3e-4,

momentum= 0.9,
weight_decay = 1e-4,

# Training parameters in common
emb_dim = 128,
instance_encoder = 'resnet50', # Use resnet 18 for confusion log probability 
dataset = 'KMNIST',
#OOD_dataset = ['CIFAR10'],
#dataset = 'CIFAR100',
#OOD_dataset = ['SVHN'],
pretrained_network = None,

# Wandb parameters in common
project = 'evaluation',

group = 'New Model Testing',
notes = 'Testing whether new models are able to train effectively',  # Add notes to the specific models each time

# VAE specific params
kl_coeff = 0.1,
first_conv = False,
maxpool1 = False,
enc_out_dim = 128,

# Cross entropy Specific parameters
label_smoothing = False,

# Contrastive specific parameters
num_negatives = 4096,
encoder_momentum = 0.999,
softmax_temperature = 0.07,

# centroid vicreg params
invariance_weight = 1.0,
variance_weight = 1.0,
covariance_weight = 1.0,

single_model = 'Baselines'
)  # evaluation

# Updates OOD dataset if not manually specified
if 'OOD_dataset' in base_hparams:
    pass    
else:
    base_hparams['OOD_dataset'] = OOD_dict[base_hparams['dataset']]


trainer_hparams = dict(

# Miscellaneous arguments in common
seed = 42,
epochs = 50, #300,
bsz = 256, #512,

# Trainer configurations in common
fast_run = False,
quick_callback = False,
training_ratio = 0.2, #1.0,
validation_ratio = 0.2, #1.0,
test_ratio = 0.2,
val_check = 20,
model_saving = 200, # Used to control how often the model is saved



callbacks = [],#['Model_saving'],
)