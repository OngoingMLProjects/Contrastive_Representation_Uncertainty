from pytorch_lightning import callbacks
from Contrastive_uncertainty.basic_replica.basic_general.datamodules.datamodule_dict import OOD_dict


base_hparams = dict(
# Optimizer parameters in common
#optimizer = 'adam', #'adam',
#learning_rate= 3e-4, #3e-4,

optimizer = 'sgd', #'adam',
learning_rate= 3e-2, #3e-4,

momentum= 0.9,
weight_decay = 1e-4,

# Training parameters in common
emb_dim = 64,
dataset = 'MNIST',
#OOD_dataset = ['CIFAR10'],
#dataset = 'CIFAR100',
#OOD_dataset = ['SVHN'],
pretrained_network = None,

# Wandb parameters in common
project = 'basic_evaluation',

group = 'Practice_basic',
notes = 'Testing whether new models are able to train effectively',  # Add notes to the specific models each time


# Cross entropy Specific parameters
label_smoothing = False,

# Contrastive specific parameters
num_negatives = 4096,
encoder_momentum = 0.999,
softmax_temperature = 0.07,

# centroid vicreg params
invariance_weight = 15.0,
variance_weight = 15.0,
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
epochs = 100, #300,
bsz = 512, #512,

# Trainer configurations in common
fast_run = False,
quick_callback = False,
training_ratio = 1.0, #1.0,
validation_ratio = 1.0, #1.0,
test_ratio = 1.0,
val_check = 20,
model_saving = 200, # Used to control how often the model is saved


callbacks = ['Model_saving'], #'Model_saving'
)


batch_base_hparams = [base_hparams]
batch_trainer_hparams = [trainer_hparams]

assert len(batch_base_hparams) == len(batch_trainer_hparams)