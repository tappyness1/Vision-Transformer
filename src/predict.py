import torch.nn as nn
import torch
import pandas as pd

def predict(model, img):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model.to(device)
    return torch.argmax(model(img))