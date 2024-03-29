from Contrastive_uncertainty.general.run.model_names import model_names_dict

cross_entropy_ensemble_hparams = dict(
emb_dim = 128,
instance_encoder = 'resnet50',

# optimizer args
optimizer = 'sgd',
learning_rate= 0.03,#3e-4,
momentum= 0.9,
weight_decay = 1e-4,

bsz = 256,
dataset = 'CIFAR100',
OOD_dataset = ['SVHN', 'CIFAR10'],

label_smoothing =False,

# Miscellaneous arguments
seed = 42,
epochs = 300,

# Trainer configurations
fast_run = False,
quick_callback = False,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 10,
model_saving = 200, # Used to control how often the model is saved

num_augmentations = 5, # Used to control the number of data augmentations for multiloader callback
pretrained_network = None,#'Pretrained_models/finetuned_network.pt',
num_models = 3,

#vector_level = ['instance'],
#label_level = ['fine'],
#callbacks = ['Model_saving'],
callbacks = ['Model_saving'],

model_type = model_names_dict['CEEnsemble'],
project = 'evaluation',# evaluation, Moco_training
group = None,
notes = None,
)