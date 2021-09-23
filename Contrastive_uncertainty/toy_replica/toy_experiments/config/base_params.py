from pytorch_lightning import callbacks
from Contrastive_uncertainty.toy_replica.toy_general.datamodules.datamodule_dict import OOD_dict

base_hparams = dict(
# Optimizer parameters in common
optimizer = 'sgd',
learning_rate= 0.03,#3e-4,
momentum= 0.9,
weight_decay = 1e-4,

# Training parameters in common
emb_dim = 128,
instance_encoder = 'resnet50',
dataset = 'Blobs',
OOD_dataset = ['TwoMoons'],
# OOD_dataset = ['TwoMoons'],
pretrained_network = None,


# Wandb parameters in common
project = 'Toy_evaluation',
group = 'Practice040721',
notes = 'Examining how to automate selection of runs',  # Add notes to the specific models each time

# Cross entropy Specific parameters
label_smoothing = False,

# Contrastive specific parameters
num_negatives = 4096,
encoder_momentum = 0.999,
softmax_temperature = 0.07,

# Either goes through all the models or goes through baselines

single_model = 'Baselines'
)  # evaluation
# Updates OOD dataset if not manually specified
if 'OOD_dataset' in base_hparams:
    pass    
else:
    base_hparams['OOD_dataset'] = OOD_dict[base_hparams['dataset']]


trainer_hparams = dict(

# Miscellaneous arguments in common
seed = 26,
epochs = 1,
bsz = 128,

# Trainer configurations in common
fast_run = False,
quick_callback = False,#True,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 1,
model_saving = 1, # Used to control how often the model is saved
callbacks = ['Nearest Neighbours Class 1D Typicality'],
#callbacks = ['Nearest Class Neighbours'], #['Mahalanobis OOD Fractions'], #['Model_saving'],
)


batch_base_hparams = [base_hparams]
batch_trainer_hparams = [trainer_hparams]

assert len(batch_base_hparams) == len(batch_trainer_hparams)