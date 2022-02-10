from Contrastive_uncertainty.general.run.model_names import model_names_dict


cross_entropy_hparams = dict(
emb_dim = 128,
instance_encoder = 'resnet50',

# optimizer args
optimizer = 'sgd',
learning_rate= 0.03,#3e-4,
momentum= 0.9,
weight_decay = 1e-4,

dataset = 'MNIST',
OOD_dataset = ['FashionMNIST'],

label_smoothing =False,
use_mlp = True,
pretrained_network = None,#'Pretrained_models/finetuned_network.pt',

model_type = model_names_dict['CE'],
project = 'toy_replica',  # evaluation, Moco_training
group = None,
notes = None,
)