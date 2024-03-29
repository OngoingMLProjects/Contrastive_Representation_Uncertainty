from Contrastive_uncertainty.general.run.model_names import model_names_dict

outlier_exposure_hparams = dict(
emb_dim = 128,
instance_encoder = 'resnet50',

# optimizer args
optimizer = 'sgd',
learning_rate= 0.03,#3e-4,
momentum= 0.9,
weight_decay = 1e-4,

dataset = 'CIFAR100',
auxillary_dataset = 'Caltech256', # OOD dataset in which outlier exposure is training on 
OOD_dataset = ['SVHN', 'CIFAR10'],


model_type = model_names_dict['CE'],
project = 'evaluation',# evaluation, Moco_training
group = None,
notes = None,
)


trainer_hparams =  dict(

# Miscellaneous arguments
seed = 42,
epochs = 300,
bsz = 256,

# Trainer configurations
fast_run = False,
quick_callback = False,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 10,
model_saving = 200, # Used to control how often the model is saved
callbacks = ['Model_saving']
)