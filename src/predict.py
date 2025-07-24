import torch.nn as nn
import torch
import pandas as pd
import torch.nn.functional as F


# def predict(model, img):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model.eval()
#     model.to(device)
#     return torch.argmax(model(img))

# --- Prediction ---
def predict(model, image: torch.Tensor):
    image = image.unsqueeze(0)
    with torch.no_grad():
        out = model(image)
        probs = F.softmax(out, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
    return pred_idx, probs[0][pred_idx].item()
