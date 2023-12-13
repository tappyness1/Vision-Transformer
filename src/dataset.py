import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor
import matplotlib.pyplot as plt


def get_load_data(root = "data", dataset = "FashionMNIST", download = False):

    if dataset == "FashionMNIST":
        training_data = datasets.FashionMNIST(
            root=root,
            train=True,
            download=download,
            transform=ToTensor()
        )

        test_data = datasets.FashionMNIST(
            root=root,
            train=False,
            download=download,
            transform=ToTensor()
        )
    
    elif dataset == "Flowers102":
        training_data = datasets.Flowers102(
            root=root,
            split="test",
            download=download,
            transform=Compose([Resize((224,224)), ToTensor()]) 
        )

        test_data = datasets.Flowers102(
            root=root,
            split = "train",
            download=download,
            transform=Compose([Resize((224,224)), ToTensor()])
        )
    return training_data, test_data

if __name__ == "__main__":
    train, test = get_load_data(root = "../data")
    img, label = train[1]
    plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

    # # for gcp or whatever
    # train, test = get_load_data(root = "./data", dataset = "Flowers102", download = True)
