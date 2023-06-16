import torch.nn as nn
import torch
import copy
import math

class MultiHeadBlockLooper(nn.Module):

    def __init__(self, hidden_d = 8, k_heads = 2):
        super(MultiHeadBlockLooper, self).__init__()
        self.hidden_d = hidden_d
        self.k_heads = k_heads
        head_size = int(hidden_d / k_heads)
        self.head_size = head_size
        self.q_weights = nn.ModuleList([nn.Linear(head_size, head_size) for i in range(k_heads)])
        self.k_weights = nn.ModuleList([nn.Linear(head_size, head_size) for i in range(k_heads)])
        self.v_weights = nn.ModuleList([nn.Linear(head_size, head_size) for i in range(k_heads)])
        
    def forward(self, x):
        # x will be split into N x D/k to make it mutable for U_{qkv} 
        dim_x = x.shape
        res = []
        for head in range(self.k_heads):
            q = self.q_weights[head](x[:, :, head * self.head_size: head * self.head_size + self.head_size])
            k = self.k_weights[head](x[:, :, head * self.head_size: head * self.head_size + self.head_size])
            v = self.v_weights[head](x[:, :, head * self.head_size: head * self.head_size + self.head_size])

            sa_out = torch.matmul(torch.softmax(input = torch.matmul(q, k.transpose(-1,-2))/math.sqrt(self.hidden_d), dim = 2), v)
            res.append(sa_out)

        res = torch.cat(res, dim = 2)

        return res
    
class MLPBlock(nn.Module):
    
    def __init__(self, hidden_d = 8):
        super(MLPBlock, self).__init__()
        
        self.FC_1 = nn.Linear(hidden_d, hidden_d)
        self.FC_2 = nn.Linear(hidden_d, hidden_d)
        self.gelu = nn.GELU()

    def forward(self, x):
        out = self.FC_1(x)
        out = self.gelu(x)
        out = self.FC_1(x)

        return out
   

class TransformerBlockLooper(nn.Module):
    # loop implementation for multihead because I'm a scrub

    def __init__(self, hidden_d = 8, k_heads = 2):
        super(TransformerBlockLooper, self).__init__()
        self.hidden_d = hidden_d
        self.num_heads = k_heads

        self.layer_norm = nn.LayerNorm(hidden_d)
        self.msa = MultiHeadBlockLooper(hidden_d=hidden_d, k_heads=k_heads)
        self.mlp = MLPBlock(hidden_d)
        
    def forward(self, input):
        
        skip_con_1 = torch.clone(input)
        out = self.layer_norm(input)
        out = self.msa(out) + skip_con_1
        
        skip_con_2 = torch.clone(out)
        out = self.layer_norm(out)
        out = self.mlp(out) + skip_con_2

        return out

class MultiHeadBlock(nn.Module):

    def __init__(self, hidden_d = 8, k_heads = 2):
        super(MultiHeadBlock, self).__init__()
        self.hidden_d = hidden_d
        self.k_heads = k_heads
        head_size = int(hidden_d / k_heads)
        self.head_size = head_size

        self.q_weights = nn.Parameter(torch.rand(k_heads,head_size,head_size))
        self.q_bias = nn.Parameter(torch.rand(k_heads,head_size))
        self.k_weights = nn.Parameter(torch.rand(k_heads,head_size,head_size))
        self.k_bias = nn.Parameter(torch.rand(k_heads,head_size))
        self.v_weights = nn.Parameter(torch.rand(k_heads,head_size,head_size))
        self.v_bias = nn.Parameter(torch.rand(k_heads,head_size))        

    def forward(self, x):
        # TODO: work on how to get it working using nn.Parameter or something similar.

        # x will be split into N x D/k to make it mutable for U_{qkv} 
        n, num_patches, dims = x.shape
        x_split = torch.reshape(x, (n, num_patches, self.k_heads, self.head_size))
        print (x_split.shape)
        print (self.q_weights.shape)
        print (self.q_bias.shape)
        q = x_split @ torch.transpose(self.q_weights, 1,2) + self.q_bias
        k = x_split @ self.k_weights.T + self.k_bias
        v = x_split @ self.v_weights.T + self.v_bias

        print (q.shape)
        # sa_out = torch.matmul(torch.softmax(input = torch.matmul(q, k.transpose(-1,-2))/math.sqrt(self.hidden_d), dim = 2), v)
        # res = torch.cat((res, sa_out),2)

        # return res
    
class TransformerBlock(nn.Module):

    def __init__(self, input_dim = (3, 224, 224), n_patches = 196, hidden_d = 8, k_heads = 2):
        super(TransformerBlock, self).__init__()
        self.hidden_d = hidden_d
        self.num_heads = k_heads
        
    def forward(self, input):

        return 
    
if __name__=="__main__":

    import numpy as np
    import torch

    np.random.seed(42)
    torch.manual_seed(42)    

    X = np.random.rand(1, 50, 8).astype('float32')
    X = torch.tensor(X)

    # print (X)

    # t_block = TransformerBlock(hidden_d = 2)
    
    # print (t_block.forward(X).shape)

    msa_block = MultiHeadBlockLooper()
    print (msa_block.forward(X).shape)