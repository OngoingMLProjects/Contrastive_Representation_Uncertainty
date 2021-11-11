from Contrastive_uncertainty.general.run.model_names import model_names_dict

cross_entropy_hparams = dict(
emb_dim = 64,


# optimizer args
optimizer = 'sgd',
learning_rate= 0.03,#3e-4,
momentum= 0.9,
weight_decay = 1e-4,

dataset = 'MNIST',
OOD_dataset = ['FashionMNIST'],

label_smoothing =False,

model_type = model_names_dict['CE'],
project = 'basic_evaluation',# evaluation, Moco_training
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