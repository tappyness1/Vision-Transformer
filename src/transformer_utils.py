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
        print (res.shape)

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

class MultiHeadBlock(nn.Module):

    def __init__(self, hidden_d = 8, k_heads = 2):
        super(MultiHeadBlock, self).__init__()
        self.hidden_d = hidden_d
        self.k_heads = k_heads
        head_size = int(hidden_d / k_heads)
        self.head_size = head_size

        # don't split into head yet
        # doesn't this just make it the same as if you did not have multi-head?
        # not so, because by right your scaled dot product attention calculations will be different
        # will see that happen later because we split by heads in later code
        self.q_weights = nn.Linear(hidden_d, hidden_d)
        self.k_weights = nn.Linear(hidden_d, hidden_d)
        self.v_weights = nn.Linear(hidden_d, hidden_d)

    def forward(self, x):

        n, num_patches, dims = x.shape

        # after run the calculations by linear combination, we will then 
        # split to the different heads before doing the scaled dot product attention, SDPA

        q = self.q_weights(x).reshape(n, self.k_heads, num_patches, self.head_size)
        k = self.k_weights(x).reshape(n, self.k_heads, num_patches, self.head_size)
        v = self.v_weights(x).reshape(n, self.k_heads, num_patches, self.head_size)

        sa_out = torch.matmul(torch.softmax(input = torch.matmul(q, k.transpose(-1,-2))/math.sqrt(self.hidden_d), dim = 2), v)

        # concatenate by reshaping back to (batch, num_patches, hidden_dim)
        res = sa_out.reshape(n, num_patches, self.k_heads * self.head_size)

        return res

class TransformerBlock(nn.Module):
    # loop implementation for multihead because I'm a scrub

    def __init__(self, hidden_d = 8, k_heads = 2, looper = False):
        super(TransformerBlock, self).__init__()
        self.hidden_d = hidden_d
        self.num_heads = k_heads

        self.layer_norm = nn.LayerNorm(hidden_d)
        if looper:
            self.msa = MultiHeadBlockLooper(hidden_d=hidden_d, k_heads=k_heads)
        else:
            self.msa = MultiHeadBlock(hidden_d=hidden_d, k_heads=k_heads)

        self.mlp = MLPBlock(hidden_d)
        
    def forward(self, input):
        
        skip_con_1 = torch.clone(input)
        out = self.layer_norm(input)
        out = self.msa(out) + skip_con_1
        
        skip_con_2 = torch.clone(out)
        out = self.layer_norm(out)
        out = self.mlp(out) + skip_con_2

        return out
    
if __name__=="__main__":

    import numpy as np
    import torch

    np.random.seed(42)
    torch.manual_seed(42)    

    X = np.random.rand(1, 50, 8).astype('float32')
    X = torch.tensor(X)

    # print (X)

    t_block = TransformerBlock(hidden_d = 8)
    
    print (t_block.forward(X).shape)

    # msa_block = MultiHeadBlockLooper()
    # msa_block = MultiHeadBlock()

    # print (msa_block.forward(X).shape)