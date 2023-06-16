import torch.nn as nn
import torch
import copy
import math

class TransformerBlock(nn.Module):

    def __init__(self, input_dim = (3, 224, 224), patch_dim = 16, n_patches = 196, hidden_d = 8):
        super(TransformerBlock, self).__init__()
        self.patch_dim = patch_dim
        self.n_patches = 196
        self.hidden_d = hidden_d

        # first head
        self.Wq_1 = nn.Linear(hidden_d, hidden_d)
        self.Wk_1 = copy.deepcopy(self.Wq_1)
        self.Wv_1 = copy.deepcopy(self.Wq_1)

        # print (self.q_1)
        # print(list(self.Wq_1.parameters()))
        
    def forward(self, input):

        q1 = self.Wq_1(input)
        # print (f"From Q_1: {q1}")
        k1 = self.Wk_1(input)
        v1 = self.Wv_1(input)

        # print (f"From Q_1: {q1}")
        # print (f"From K_1: {k1}")
        # print (f"From V_1: {v1}")

        # print (W_q1.shape)
        # print (W_k1.shape)
        # print (W_v1.shape)

        # pairwise_similarity = torch.matmul(q1, k1.transpose(-2,-1))/math.sqrt(self.hidden_d)
        # # print (pairwise_similarity.shape) # n x 197 x 197
        # print (f"pairwise: \n {pairwise_similarity}")
        # softmaxed = torch.softmax(input = torch.matmul(q1, k1.transpose(-2,-1))/math.sqrt(self.hidden_d), dim = 2)
        # print (f"softmax: \n {softmaxed}")    
    
        SA_1 = torch.matmul(torch.softmax(input = torch.matmul(q1, k1.transpose(-1,-2))/math.sqrt(self.hidden_d), dim = 2), v1)

        # print (SA_1)

        return SA_1
    
if __name__=="__main__":

    import numpy as np
    import torch

    np.random.seed(42)
    torch.manual_seed(42)    

    X = np.random.rand(1, 2, 2).astype('float32')
    X = torch.tensor(X)

    # print (X)

    t_block = TransformerBlock(hidden_d = 2)
    
    print (t_block.forward(X).shape)