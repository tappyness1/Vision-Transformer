from src.train import train
from src.validation import validation
import hydra
from omegaconf import DictConfig, OmegaConf
from src.dataset import get_load_data
import torch

@hydra.main(version_base = None, config_path="../cfg", config_name="cfg")
def train_model(cfg : DictConfig):
    torch.manual_seed(42)
    train_set, test_set = get_load_data(root = "../data", dataset = "Flowers102")
    train(train_set, cfg, in_channels = 3, num_classes = 102)

@hydra.main(version_base = None, config_path="../cfg", config_name="cfg_hptuning.yaml")
def train_model_hptuning(cfg : DictConfig):
    torch.manual_seed(42)
    train_set, test_set = get_load_data(root = "../data", dataset = "Flowers102")
    network = train(train_set, cfg, in_channels = 3, num_classes = 102)
    f1_score = validation(network, test_set)

    return f1_score

if __name__ == "__main__":
    # train_model()
    train_model_hptuning() # remember to use --multirun flag if it is not turned on in the yaml
