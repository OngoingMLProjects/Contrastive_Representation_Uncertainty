from pytorch_lightning import callbacks
from pytorch_lightning.callbacks import base
from Contrastive_uncertainty.toy_replica.toy_general.datamodules.datamodule_dict import OOD_dict

base1_hparams = dict(
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
if 'OOD_dataset' in base1_hparams:
    pass    
else:
    base1_hparams['OOD_dataset'] = OOD_dict[base1_hparams['dataset']]


trainer1_hparams = dict(

# Miscellaneous arguments in common
seed = 26,
epochs = 1,
bsz = 64,

# Trainer configurations in common
fast_run = False,
quick_callback = False,#True,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 1,
model_saving = 1, # Used to control how often the model is saved
callbacks = ['Model_saving','Metrics'],#['Model_saving'],
)
####################################


base2_hparams = dict(
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
if 'OOD_dataset' in base2_hparams:
    pass    
else:
    base2_hparams['OOD_dataset'] = OOD_dict[base2_hparams['dataset']]


trainer2_hparams = dict(

# Miscellaneous arguments in common
seed = 26,
epochs = 1,
bsz = 64,

# Trainer configurations in common
fast_run = False,
quick_callback = False,#True,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 1,
model_saving = 1, # Used to control how often the model is saved
callbacks = ['Metrics'],#['Model_saving'],
)





base3_hparams = dict(
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
if 'OOD_dataset' in base3_hparams:
    pass    
else:
    base3_hparams['OOD_dataset'] = OOD_dict[base3_hparams['dataset']]


trainer3_hparams = dict(

# Miscellaneous arguments in common
seed = 26,
epochs = 1,
bsz = 64,

# Trainer configurations in common
fast_run = False,
quick_callback = False,#True,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 1,
model_saving = 1, # Used to control how often the model is saved
callbacks = ['Metrics'],#['Model_saving'],
)





batch_base_hparams = [base1_hparams, base2_hparams, base3_hparams]
batch_trainer_hparams = [trainer1_hparams, trainer2_hparams, trainer3_hparams]

assert len(batch_base_hparams) == len(batch_trainer_hparams)