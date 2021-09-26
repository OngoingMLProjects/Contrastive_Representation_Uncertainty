from Contrastive_uncertainty.toy_replica.toy_experiments.train.names_dict import model_names_dict

centroid_vicreg_hparams = dict(
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

use_mlp = True,

invariance_weight = 1.0,
variance_weight = 1.0,
covariance_weight = 1.0,

callbacks = ['Model_saving'],

model_type = model_names_dict['Centroid_VicReg'],
project = 'toy_replica',# evaluation, Moco_training
group = None,
notes = None,
)