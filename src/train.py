import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from src.model import ViT
from torch.utils.data import DataLoader
from tqdm import tqdm



def train(train_set, cfg, in_channels = 3, num_classes = 10):

    loss_function = nn.CrossEntropyLoss()
    
    network = ViT(**cfg['vit_config'])

    network.train()

    optimizer = optim.AdamW(network.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    network = network.to(device)

    train_dataloader = DataLoader(train_set, batch_size=20)
    
    for epoch in range(cfg['train']['epochs']):
        print (f"Epoch {epoch + 1}:")
        # for i in tqdm(range(X.shape[0])):
        with tqdm(train_dataloader) as tepoch:
            for imgs, labels in tepoch:
                # only need to give [N x num_classes]. Loss function will do the rest for you. Probably an internal argmax
                out = network(imgs.to(device))
                loss = loss_function(out, labels.to(device))

                optimizer.zero_grad() 
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
        
    print("training done")
    torch.save(network, cfg['save_model_path'])

    return network

if __name__ == "__main__":

    torch.manual_seed(42)

    from src.dataset import get_load_data
    cfg = {"save_model_path": "model_weights/model_weights.pt",
           'train': {"epochs": 2, 'lr': 31e-4, 'weight_decay': 0.05},
           'vit_config': {
               'img_dim': (3, 224, 224),
               'patch_size': 16,
               'num_classes': 102,
               'hidden_dim': 768,
               'num_heads': 12,
               'num_transformers': 12
           }
           }
    train_set, test_set = get_load_data(root = "data", dataset = "Flowers102")
    train(train_set = train_set, cfg = cfg, in_channels = 3, num_classes = 102)

    # cannot use FashionMNIST because size needs to be 224x224x3 at the very least
    # train_set, test_set = get_load_data(root = "../data", dataset = "FashionMNIST")
    # train(epochs = 1, train_set = train_set, in_channels = 1, num_classes = 10)
    