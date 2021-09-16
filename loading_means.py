from scipy.io import loadmat
import torch

# From TRAIN TF code
#Load centers in MMC
kernel_dict = loadmat('meanvar1_featuredim128_class10.mat') # Nawid - load precomputed centres
mean_logits_np = kernel_dict['mean_logits'] #num_class X num_dense # Nawid - centres
#mean_logits = FLAGS.mean_var * tf.constant(mean_logits_np,dtype=tf.float32) # Nawid -calculate the centres



# FROM MMLDA CODE 
'''
def precompute_means(self):
        mean_var = 1
        mean_logits =torch.zeros((self.hparams,self.hparams.latent_size))
        mean_logits[0,0] = 1 # set the first mean vector equal to the first unit basis vector
        for k in range(1,self.hparams.num_classes):
            for j in range(k-1):
                mean_logits[k,j] = (-1/(self.hparams.num_classes)+torch.dot(mean_logits[k],mean_logits[j])).div(mean_logits[j,j])
            mean_logits[k,k] = torch.sqrt(torch.abs(1 - torch.pow(torch.linalg.norm(mean_logits[k]),2)))

        self.mean_logits = (mean_logits * mean_var)
        return self.mean_logits
'''

def precompute_means():
    mean_var = 1
    mean_logits =torch.zeros((10,128))
    mean_logits[0,0] = 1 # set the first mean vector equal to the first unit basis vector
    for k in range(1,10):
        for j in range(k-1):
            mean_logits[k,j] = (-1/(10)+torch.dot(mean_logits[k],mean_logits[j])).div(mean_logits[j,j])
            mean_logits[k,k] = torch.sqrt(torch.abs(1 - torch.pow(torch.linalg.norm(mean_logits[k]),2)))

        mean_logits = (mean_logits * mean_var)
        return mean_logits

means = precompute_means()


# Check if the way of calculating the means are the same in each case