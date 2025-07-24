from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_positional_embeddings(sequence_length, d): # not used in ViT
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result

class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim = 8, num_heads = 2):
        super().__init__()
        self.head_size = hidden_dim // num_heads
        self.num_heads = num_heads
        self.query = nn.Linear(hidden_dim, hidden_dim, bias = True)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias = True)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias = True)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.shape
        Q, K, V = self.query(x), self.key(x), self.value(x)
        Q, K, V = Q.view(B, self.num_heads, T, self.head_size), K.view(B, self.num_heads, T, self.head_size), V.view(B, self.num_heads, T, self.head_size)
        wei = Q @ K.transpose(-2, -1) / (self.head_size ** 0.5)
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)
        out = wei @ V
        out = out.view(B, T, self.num_heads * self.head_size) 
        out = self.dropout(self.proj(out))
        return out

class MLPLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(hidden_dim*4, hidden_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        out = self.linear_1(x)
        out = self.gelu(out)
        out = self.linear_2(out)
        out = self.dropout(out)
        return out

class TransformerEncoderLayer(nn.Module):

    def __init__(self, hidden_dim = 8, num_heads = 2):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.attn = AttentionBlock(hidden_dim, num_heads)
        self.mlp = MLPLayer(hidden_dim)

    def forward(self, x):
        out = x + self.attn(self.ln_1(x))
        out = out + self.mlp(self.ln_2(x))
        return out

class ViT(nn.Module):

    def __init__(self, 
                img_dim: Tuple[int,int,int] = (3, 224, 224), 
                patch_size: int = 16, 
                num_classes = 102,
                hidden_dim = 8,
                num_heads = 2,
                num_transformers = 1):
        super().__init__()
        C, H, _ = img_dim
        N = int((H/patch_size) **2)
        flat_patch_dim = patch_size**2 * C
        self.hidden_dim = hidden_dim

        self.unfold = torch.nn.Unfold(kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))

        self.img_enc = nn.Linear(flat_patch_dim, hidden_dim)        
        self.pos_emb = nn.Parameter(torch.randn((1, N + 1, hidden_dim)))
        self.cls_token = nn.Parameter(torch.randn((1, 1,hidden_dim)))
        self.transformer_blocks = nn.Sequential(*[TransformerEncoderLayer(hidden_dim, num_heads) for _ in range(num_transformers)]) 
        self.mlp = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        out = self.patchify(x)
        out = self.img_enc(out)
        out = torch.cat((self.cls_token.expand(B, -1, -1), out), dim =1)
        out += self.pos_emb
        out = self.transformer_blocks(out)
        out = out[:, 0, :]
        out = self.mlp(out)
        return out 

    def patchify(self, x) -> torch.Tensor:

        patches = self.unfold(x)
        patches = patches.transpose(1,2)
        return patches

if __name__ == "__main__":
    from torchsummary import summary

    np.random.seed(42)
    torch.manual_seed(42)    

    X = np.random.rand(5, 3, 224, 224).astype('float32')
    X = torch.tensor(X)

    model = ViT()
    
    # summary(model, (3, 224, 224))
    print (model(X).shape)