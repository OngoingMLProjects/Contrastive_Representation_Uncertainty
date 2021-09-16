from pytorch_lightning import callbacks
from Contrastive_uncertainty.general.datamodules.datamodule_dict import OOD_dict

practice_base_hparams = dict(
# Optimizer parameters in common
optimizer = 'sgd',
learning_rate= 0.03,#3e-4,
momentum= 0.9,
weight_decay = 1e-4,

# Training parameters in common
emb_dim = 128,
instance_encoder = 'resnet18',

dataset = 'MNIST',
#OOD_dataset = ['SVHN'],
#OOD_dataset = ['SVHN','CIFAR10'],


# Wandb parameters in common
project = 'practice',
group = None,
notes = None, # Add notes to the specific models each time


# Cross entropy Specific parameters
label_smoothing = False,

# Contrastive specific parameters
num_negatives = 512,
encoder_momentum = 0.999,
softmax_temperature = 0.07,


single_model = 'Baselines'
)  # evaluation


if 'OOD_dataset' in practice_base_hparams:
    pass    
else:
    practice_base_hparams['OOD_dataset'] = OOD_dict[practice_base_hparams['dataset']]


practice_trainer_hparams = dict(

# Miscellaneous arguments in common
seed = 26,
epochs = 300,
bsz = 64,

# Trainer configurations in common
fast_run = True,
quick_callback = True,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 20,
model_saving = 200, # Used to control how often the model is saved

callbacks = ['Model_saving'],
)