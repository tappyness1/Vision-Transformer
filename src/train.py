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

    optimizer = optim.SGD(network.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    network = network.to(device)

    train_dataloader = DataLoader(train_set, batch_size=20)
    
    for epoch in range(cfg['train']['epochs']):
        print (f"Epoch {epoch + 1}:")
        # for i in tqdm(range(X.shape[0])):
        with tqdm(train_dataloader) as tepoch:
            for imgs, labels in tepoch:
                print (imgs.shape)
                optimizer.zero_grad() 
                # only need to give [N x num_classes]. Loss function will do the rest for you. Probably an internal argmax
                out = network(imgs.to(device))
                loss = loss_function(out, labels.to(device))
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
           "epochs": 2, 
           'show_model_summary': True, 
           'train': {'lr': 0.001, 'weight_decay': 5e-5},
           'vit_config': {
               'img_dim': (3, 224, 224),
               'patch_size': 16,
               'num_classes': 102,
               'hidden_dim': 768,
               'num_heads': 14,
               'num_transformers': 12
           }
           }
    train_set, test_set = get_load_data(root = "data", dataset = "Flowers102")
    train(train_set = train_set, cfg = cfg, in_channels = 3, num_classes = 102)

    # cannot use FashionMNIST because size needs to be 224x224x3 at the very least
    # train_set, test_set = get_load_data(root = "../data", dataset = "FashionMNIST")
    # train(epochs = 1, train_set = train_set, in_channels = 1, num_classes = 10)
    