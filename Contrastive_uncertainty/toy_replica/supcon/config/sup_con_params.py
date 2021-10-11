from Contrastive_uncertainty.general.run.model_names import model_names_dict

sup_con_hparams = dict(
emb_dim = 128,
softmax_temperature = 0.07,
instance_encoder = 'resnet50',

# optimizer args
optimizer = 'sgd',
learning_rate= 0.03,#3e-4,
momentum= 0.9,
weight_decay = 1e-4,

dataset = 'Blobs',
OOD_dataset = ['TwoMoons','Diagonal'],

contrast_mode = 'one',

callbacks = ['Model_saving','Mahalanobis'],

model_type = model_names_dict['SupCon'],
project = 'toy_replica',# evaluation, Moco_training
group = None,
notes = None,
)