import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


def weight_init(m): # Nawid - weight normalisation
    """Kaiming_normal is standard for relu networks, sometimes."""
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)): # Nawid - if m is an an instance of torch.nn linear or conv2d, then apply weight normalisation
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_in",
            nonlinearity="relu")
        torch.nn.init.zeros_(m.bias)

# Neural network based on https://towardsdatascience.com/build-a-fashion-mnist-cnn-pytorch-style-efb297e22582
class Classification_Backbone(nn.Module):
    def __init__(self, emb_dim,num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12,kernel_size=3)
        
        # Could look at increasing the conv layers present, 
        self.fc1 = nn.Linear(12*5*5, emb_dim)
        self.fc2 = nn.Linear(emb_dim, emb_dim)
        # classification layer
        self.fc3 = nn.Linear(emb_dim, num_classes)
        
        #self.apply(weight_init)
        
    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # fc1
        x = x.reshape(-1,12*5*5) # 12 channesl and height adnd width of 5 and 5
        x = F.relu(self.fc1(x))

        x = self.fc2(x) # output (batch, emb_dim)

        return x
    def class_forward(self,z):

        class_output = self.fc3(F.relu(z)) # output (batch, num classes)
        return class_output

'''
class Classification_Backbone(nn.Module):
    def __init__(self, hidden_dim, emb_dim):
        super().__init__()

        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, emb_dim)
        
        #self.apply(weight_init)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Nawid -output (batch, features)
        return x
'''



# Neural network
class Backbone(nn.Module):
    def __init__(self, hidden_dim, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12,kernel_size=3)
        
        # Could look at increasing the conv layers present, 
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, emb_dim)
        # classification layer
        
        #self.apply(weight_init)
        
    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # fc1
        x = x.reshape(-1,128)
        x = F.relu(self.fc1(x))
        # fc2
        x = self.fc2(x) # output (batch, embedding dim)
        return x