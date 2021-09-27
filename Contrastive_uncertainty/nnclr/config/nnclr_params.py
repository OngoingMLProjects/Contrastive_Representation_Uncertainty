from Contrastive_uncertainty.general.run.general_run_setup import model_names_dict

nnclr_hparams  = dict(

emb_dim = 128,
queue_size = 65536,
softmax_temperature = 0.07,
instance_encoder = 'resnet50',

# optimizer args
optimizer = 'sgd',
learning_rate= 0.03,#3e-4,
momentum= 0.9,
weight_decay = 1e-4,

dataset = 'MNIST',
OOD_dataset = ['FashionMNIST'],

model_type = model_names_dict['NNCLR'],
project = 'evaluation',# evaluation, Moco_training
group = None,
notes = None,
)


trainer_hparams = dict(

bsz = 256,
# Miscellaneous arguments
seed = 42,
epochs = 300,

# Trainer configurations
fast_run = False,
quick_callback = False,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 20,
model_saving = 200, # Used to control how often the model is saved

callbacks = ['Model_saving'],
)
