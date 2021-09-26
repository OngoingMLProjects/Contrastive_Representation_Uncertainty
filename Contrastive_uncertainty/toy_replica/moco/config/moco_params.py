from Contrastive_uncertainty.toy_replica.toy_experiments.train.names_dict import model_names_dict

moco_hparams = dict(
emb_dim = 128,
num_negatives = 65536,
encoder_momentum = 0.999,
softmax_temperature = 0.07,
instance_encoder = 'resnet50',

# optimizer args
optimizer = 'sgd',
learning_rate= 0.03,#3e-4,
momentum= 0.9,
weight_decay = 1e-4,

dataset = 'Blobs',
OOD_dataset = ['TwoMoons','Diagonal'],

callbacks = ['Model_saving','MMD_instance','Metrics','Visualisation','Mahalanobis'],

model_type = model_names_dict['Moco'],
project = 'toy_replica',# evaluation, Moco_training
group = None,
notes = None,
)