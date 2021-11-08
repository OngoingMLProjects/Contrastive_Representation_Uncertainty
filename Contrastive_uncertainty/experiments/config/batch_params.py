from pytorch_lightning import callbacks
from Contrastive_uncertainty.general.datamodules.datamodule_dict import OOD_dict

base1_hparams = dict(
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
#dataset = 'TinyImageNet',
dataset = 'Cub200',
#OOD_dataset = ['CIFAR10'],
#dataset = 'CIFAR100',
#OOD_dataset = ['SVHN'],
pretrained_network = None,

# Wandb parameters in common
project = 'evaluation',

group = 'Baselines Repeats', #'New Model Testing',
notes = 'Repeating the hierarchical baselines', #'Testing whether new models are able to train effectively',  # Add notes to the specific models each time

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
if 'OOD_dataset' in base1_hparams:
    pass    
else:
    base1_hparams['OOD_dataset'] = OOD_dict[base1_hparams['dataset']]


trainer1_hparams = dict(

# Miscellaneous arguments in common
seed = 150,
epochs = 300, #300,
bsz = 256, #512,

# Trainer configurations in common
fast_run = False,
quick_callback = False,
training_ratio = 1.0, #1.0,
validation_ratio = 1.0, #1.0,
test_ratio = 1.0,
val_check = 20,
model_saving = 200, # Used to control how often the model is saved

callbacks = ['Model_saving'],
#callbacks = ['Model_saving','Mahalanobis Distance','Nearest 10 Neighbours Class 1D Typicality','Nearest 10 Neighbours Class Quadratic 1D Typicality'], #'Model_saving'
)

base2_hparams = dict(
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
#dataset = 'TinyImageNet',
dataset = 'Dogs',
#OOD_dataset = ['CIFAR10'],
#dataset = 'CIFAR100',
#OOD_dataset = ['SVHN'],
pretrained_network = None,

# Wandb parameters in common
project = 'evaluation',


group = 'Baselines Repeats', #'New Model Testing',
notes = 'Repeating the hierarchical baselines', #'Testing whether new models are able to train effectively',  # Add notes to the specific models each time

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
if 'OOD_dataset' in base2_hparams:
    pass    
else:
    base2_hparams['OOD_dataset'] = OOD_dict[base2_hparams['dataset']]



trainer2_hparams = dict(

# Miscellaneous arguments in common
seed = 150,
epochs = 300, #300,
bsz = 256, #512,

# Trainer configurations in common
fast_run = False,
quick_callback = False,
training_ratio = 1.0, #1.0,
validation_ratio = 1.0, #1.0,
test_ratio = 1.0,
val_check = 20,
model_saving = 200, # Used to control how often the model is saved

callbacks = ['Model_saving'],
#callbacks = ['Model_saving','Mahalanobis Distance','Nearest 10 Neighbours Class 1D Typicality','Nearest 10 Neighbours Class Quadratic 1D Typicality'], #'Model_saving'
)




base3_hparams = dict(
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
#dataset = 'TinyImageNet',
dataset = 'Caltech101',
#OOD_dataset = ['CIFAR10'],
#dataset = 'CIFAR100',
#OOD_dataset = ['SVHN'],
pretrained_network = None,

# Wandb parameters in common
project = 'evaluation',

group = 'Baselines Repeats', #'New Model Testing',
notes = 'Repeating the hierarchical baselines', #'Testing whether new models are able to train effectively',  # Add notes to the specific models each time

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
if 'OOD_dataset' in base3_hparams:
    pass    
else:
    base3_hparams['OOD_dataset'] = OOD_dict[base3_hparams['dataset']]



trainer3_hparams = dict(

# Miscellaneous arguments in common
seed = 150,
epochs = 300, #300,
bsz = 256, #512,

# Trainer configurations in common
fast_run = False,
quick_callback = False,
training_ratio = 1.0, #1.0,
validation_ratio = 1.0, #1.0,
test_ratio = 1.0,
val_check = 20,
model_saving = 200, # Used to control how often the model is saved

callbacks = ['Model_saving'],
#callbacks = ['Model_saving','Mahalanobis Distance','Nearest 10 Neighbours Class 1D Typicality','Nearest 10 Neighbours Class Quadratic 1D Typicality'], #'Model_saving'
)


base4_hparams = dict(
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
#dataset = 'TinyImageNet',
dataset = 'Caltech256',
#OOD_dataset = ['CIFAR10'],
#dataset = 'CIFAR100',
#OOD_dataset = ['SVHN'],
pretrained_network = None,

# Wandb parameters in common
project = 'evaluation',

group = 'Baselines Repeats', #'New Model Testing',
notes = 'Repeating the hierarchical baselines', #'Testing whether new models are able to train effectively',  # Add notes to the specific models each time

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
if 'OOD_dataset' in base4_hparams:
    pass    
else:
    base4_hparams['OOD_dataset'] = OOD_dict[base4_hparams['dataset']]



trainer4_hparams = dict(

# Miscellaneous arguments in common
seed = 150,
epochs = 300, #300,
bsz = 256, #512,

# Trainer configurations in common
fast_run = False,
quick_callback = False,
training_ratio = 1.0, #1.0,
validation_ratio = 1.0, #1.0,
test_ratio = 1.0,
val_check = 20,
model_saving = 200, # Used to control how often the model is saved

callbacks = ['Model_saving'],
#callbacks = ['Model_saving','Mahalanobis Distance','Nearest 10 Neighbours Class 1D Typicality','Nearest 10 Neighbours Class Quadratic 1D Typicality'], #'Model_saving'
)

'''
batch_base_hparams = [base1_hparams]
batch_trainer_hparams = [trainer1_hparams]
'''

batch_base_hparams = [base1_hparams, base2_hparams,base3_hparams, base4_hparams]
batch_trainer_hparams = [trainer1_hparams, trainer2_hparams,trainer3_hparams, trainer4_hparams]

'''
batch_base_hparams = [base1_hparams, base2_hparams, base3_hparams]
batch_trainer_hparams = [trainer1_hparams, trainer2_hparams, trainer3_hparams]
'''
assert len(batch_base_hparams) == len(batch_trainer_hparams)



base5_hparams = dict(
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
dataset = 'TinyImageNet',
#dataset = 'Cub200',
#OOD_dataset = ['CIFAR10'],
#dataset = 'CIFAR100',
#OOD_dataset = ['SVHN'],
pretrained_network = None,

# Wandb parameters in common
project = 'evaluation',

group = 'Baselines Repeats', #'New Model Testing',
notes = 'Repeating the hierarchical baselines', #'Testing whether new models are able to train effectively',  # Add notes to the specific models each time

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
if 'OOD_dataset' in base5_hparams:
    pass    
else:
    base5_hparams['OOD_dataset'] = OOD_dict[base5_hparams['dataset']]


trainer5_hparams = dict(

# Miscellaneous arguments in common
seed = 175,
epochs = 300, #300,
bsz = 256, #512,

# Trainer configurations in common
fast_run = False,
quick_callback = False,
training_ratio = 1.0, #1.0,
validation_ratio = 1.0, #1.0,
test_ratio = 1.0,
val_check = 20,
model_saving = 200, # Used to control how often the model is saved

callbacks = ['Model_saving'],
#callbacks = ['Model_saving','Mahalanobis Distance','Nearest 10 Neighbours Class 1D Typicality','Nearest 10 Neighbours Class Quadratic 1D Typicality'], #'Model_saving'

)

batch_base_hparams_1 = [base5_hparams]
batch_trainer_hparams_1 = [trainer5_hparams]

assert len(batch_base_hparams_1) == len(batch_trainer_hparams_1)




base6_hparams = dict(
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
#dataset = 'TinyImageNet',
dataset = 'Cub200',
#OOD_dataset = ['CIFAR10'],
#dataset = 'CIFAR100',
#OOD_dataset = ['SVHN'],
pretrained_network = None,

# Wandb parameters in common
project = 'evaluation',

group = 'Baselines Repeats', #'New Model Testing',
notes = 'Repeating the hierarchical baselines', #'Testing whether new models are able to train effectively',  # Add notes to the specific models each time

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
if 'OOD_dataset' in base6_hparams:
    pass    
else:
    base6_hparams['OOD_dataset'] = OOD_dict[base6_hparams['dataset']]


trainer6_hparams = dict(

# Miscellaneous arguments in common
seed = 175,
epochs = 300, #300,
bsz = 256, #512,

# Trainer configurations in common
fast_run = False,
quick_callback = False,
training_ratio = 1.0, #1.0,
validation_ratio = 1.0, #1.0,
test_ratio = 1.0,
val_check = 20,
model_saving = 200, # Used to control how often the model is saved

callbacks = ['Model_saving'],
#callbacks = ['Model_saving','Mahalanobis Distance','Nearest 10 Neighbours Class 1D Typicality','Nearest 10 Neighbours Class Quadratic 1D Typicality'], #'Model_saving'
)




base7_hparams = dict(
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
#dataset = 'TinyImageNet',
dataset = 'Dogs',
#OOD_dataset = ['CIFAR10'],
#dataset = 'CIFAR100',
#OOD_dataset = ['SVHN'],
pretrained_network = None,

# Wandb parameters in common
project = 'evaluation',


group = 'Baselines Repeats', #'New Model Testing',
notes = 'Repeating the hierarchical baselines', #'Testing whether new models are able to train effectively',  # Add notes to the specific models each time

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
if 'OOD_dataset' in base7_hparams:
    pass    
else:
    base7_hparams['OOD_dataset'] = OOD_dict[base7_hparams['dataset']]



trainer7_hparams = dict(

# Miscellaneous arguments in common
seed = 175,
epochs = 300, #300,
bsz = 256, #512,

# Trainer configurations in common
fast_run = False,
quick_callback = False,
training_ratio = 1.0, #1.0,
validation_ratio = 1.0, #1.0,
test_ratio = 1.0,
val_check = 20,
model_saving = 200, # Used to control how often the model is saved

callbacks = ['Model_saving'],
#callbacks = ['Model_saving','Mahalanobis Distance','Nearest 10 Neighbours Class 1D Typicality','Nearest 10 Neighbours Class Quadratic 1D Typicality'], #'Model_saving'
)




base8_hparams = dict(
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
#dataset = 'TinyImageNet',
dataset = 'Caltech101',
#OOD_dataset = ['CIFAR10'],
#dataset = 'CIFAR100',
#OOD_dataset = ['SVHN'],
pretrained_network = None,

# Wandb parameters in common
project = 'evaluation',

group = 'Baselines Repeats', #'New Model Testing',
notes = 'Repeating the hierarchical baselines', #'Testing whether new models are able to train effectively',  # Add notes to the specific models each time

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
if 'OOD_dataset' in base8_hparams:
    pass    
else:
    base8_hparams['OOD_dataset'] = OOD_dict[base8_hparams['dataset']]



trainer8_hparams = dict(

# Miscellaneous arguments in common
seed = 175,
epochs = 300, #300,
bsz = 256, #512,

# Trainer configurations in common
fast_run = False,
quick_callback = False,
training_ratio = 1.0, #1.0,
validation_ratio = 1.0, #1.0,
test_ratio = 1.0,
val_check = 20,
model_saving = 200, # Used to control how often the model is saved

callbacks = ['Model_saving'],
#callbacks = ['Model_saving','Mahalanobis Distance','Nearest 10 Neighbours Class 1D Typicality','Nearest 10 Neighbours Class Quadratic 1D Typicality'], #'Model_saving'
)


base9_hparams = dict(
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
#dataset = 'TinyImageNet',
dataset = 'Caltech256',
#OOD_dataset = ['CIFAR10'],
#dataset = 'CIFAR100',
#OOD_dataset = ['SVHN'],
pretrained_network = None,

# Wandb parameters in common
project = 'evaluation',

group = 'Baselines Repeats', #'New Model Testing',
notes = 'Repeating the hierarchical baselines', #'Testing whether new models are able to train effectively',  # Add notes to the specific models each time

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
if 'OOD_dataset' in base9_hparams:
    pass    
else:
    base9_hparams['OOD_dataset'] = OOD_dict[base9_hparams['dataset']]



trainer9_hparams = dict(

# Miscellaneous arguments in common
seed = 175,
epochs = 300, #300,
bsz = 256, #512,

# Trainer configurations in common
fast_run = False,
quick_callback = False,
training_ratio = 1.0, #1.0,
validation_ratio = 1.0, #1.0,
test_ratio = 1.0,
val_check = 20,
model_saving = 200, # Used to control how often the model is saved

callbacks = ['Model_saving'],
#callbacks = ['Model_saving','Mahalanobis Distance','Nearest 10 Neighbours Class 1D Typicality','Nearest 10 Neighbours Class Quadratic 1D Typicality'], #'Model_saving'
)

'''
batch_base_hparams = [base1_hparams]
batch_trainer_hparams = [trainer1_hparams]
'''

batch_base_hparams_2 = [base6_hparams, base7_hparams, base8_hparams, base9_hparams]
batch_trainer_hparams_2 = [trainer6_hparams, trainer7_hparams, trainer8_hparams, trainer9_hparams]

'''
batch_base_hparams = [base1_hparams, base2_hparams, base3_hparams]
batch_trainer_hparams = [trainer1_hparams, trainer2_hparams, trainer3_hparams]
'''
assert len(batch_base_hparams_2) == len(batch_trainer_hparams_2)




base10_hparams = dict(
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
dataset = 'TinyImageNet',
#dataset = 'Cub200',
#OOD_dataset = ['CIFAR10'],
#dataset = 'CIFAR100',
#OOD_dataset = ['SVHN'],
pretrained_network = None,

# Wandb parameters in common
project = 'evaluation',

group = 'Baselines Repeats', #'New Model Testing',
notes = 'Repeating the hierarchical baselines', #'Testing whether new models are able to train effectively',  # Add notes to the specific models each time

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
if 'OOD_dataset' in base10_hparams:
    pass    
else:
    base10_hparams['OOD_dataset'] = OOD_dict[base10_hparams['dataset']]


trainer10_hparams = dict(

# Miscellaneous arguments in common
seed = 200,
epochs = 300, #300,
bsz = 256, #512,

# Trainer configurations in common
fast_run = False,
quick_callback = False,
training_ratio = 1.0, #1.0,
validation_ratio = 1.0, #1.0,
test_ratio = 1.0,
val_check = 20,
model_saving = 200, # Used to control how often the model is saved

callbacks = ['Model_saving'],
#callbacks = ['Model_saving','Mahalanobis Distance','Nearest 10 Neighbours Class 1D Typicality','Nearest 10 Neighbours Class Quadratic 1D Typicality'], #'Model_saving'

)

batch_base_hparams_3 = [base10_hparams]
batch_trainer_hparams_3 = [trainer10_hparams]

assert len(batch_base_hparams_3) == len(batch_trainer_hparams_3)