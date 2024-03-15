import torch.nn as nn
import torch
from src.transformer_utils import TransformerBlock
import numpy as np

def gen_patches(input, patch_dim):
    # helper function where we will take the image and create the patches
    dims = input.shape
    # print (dims)

    # assume 224x224x3 images are coming in 

    return torch.reshape(input, (dims[0], -1, (patch_dim**2)*dims[-3]))

def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result

class ViT(nn.Module):

    def __init__(self, img_size = 224, patch_dim = 16, hidden_d = 8, k_heads = 2, num_classes= 10):
        super(ViT, self).__init__()
        self.patch_dim = patch_dim
        n_patches = int(((img_size**2)*3) / ((patch_dim**2)*3))

        # create the linear projection layer. basically means nn.Linear
        # default is 16x16x3 patches, with 8 output dims adding bias, 
        # we thus have 16x16x3 = 768 + 1 (bias), then 769 x 8 = 6152 dims here.
        self.linear_projection = nn.Linear((patch_dim**2)*3, hidden_d)

        # add in parameter for class token, which is a 1-D 
        self.class_token = nn.Parameter(torch.rand(1, hidden_d))

        # add positional embedding
        # why do we even need nn.Parameter here?
        self.pos_embed = nn.Parameter(torch.tensor(get_positional_embeddings(n_patches + 1, hidden_d)))
        self.pos_embed.requires_grad = False

        # transformer block
        self.transformer_encoder = TransformerBlock(hidden_d,k_heads)

        self.mlp = nn.Sequential(nn.Linear(hidden_d, num_classes), nn.Softmax(dim = -1))
        
    def forward(self, input):

        # generate patches here
        out = gen_patches(input, self.patch_dim)

        # run a nn.Linear layer here
        # linear_projection = nn.Linear(patches.shape[2], self.hidden_d)
        # out = linear_projection(patches)
        out = self.linear_projection(out) # typically will be (batch, n_patches, hidden_d), n_patches = 196, hidden_d = 8

        # stack the class token on top of previous token
        out = torch.stack([torch.vstack((self.class_token, out[i])) for i in range(len(out))]) # will now be (batch, n_patches + 1, hidden_d)

        # add the positional embeddings
        out += self.pos_embed # same as after class token, will be (batch, n_patches + 1, hidden_d)

        out = self.transformer_encoder(out)
        
        out = out[:, 0, :]
        out = self.mlp(out)

        return out

if __name__ == "__main__":
    import numpy as np
    import torch
    from torchsummary import summary

    np.random.seed(42)
    torch.manual_seed(42)    

    X = np.random.rand(5, 3, 224, 224).astype('float32')
    X = torch.tensor(X)

    model = ViT()
    
    summary(model, (1, 3, 224, 224))
    print (model.forward(X).shape)