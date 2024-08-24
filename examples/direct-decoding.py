import os, pickle
import cornet
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import alexnet, AlexNet_Weights, resnet18, ResNet18_Weights, vgg19, VGG19_Weights
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from Numbersense.utilities.helpers import getenv

class NumbersenseDataset(Dataset):
    def __init__(self, dir):
        self.dir = dir
        self.files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        self.labels = [int(f.split('-')[1]) for f in self.files]

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.dir, self.files[idx])
        label = self.labels[idx]
        data = Image.open(file_path).convert('RGB')
        data = transforms.ToTensor()(data)
        return data, label

def get_dataloader(directory, bs=4, num_workers=1): return DataLoader(NumbersenseDataset(directory), batch_size=bs, shuffle=True, num_workers=num_workers)

def direct_decoding(model, dataloader, batch_count:int):
    model.eval()
    features, numerosities = [], []
    for i, (images, labels) in enumerate(dataloader):
        if i >= batch_count: break
        features.extend(model(images[:, :3, :, :].to(getenv('DEVICE', 'cuda'))).detach().cpu().view(images.size(0), -1).numpy())
        numerosities.append(labels)
    labels = np.concatenate(numerosities)

    model = LinearRegression()
    return (scores := cross_val_score(model, features, labels, cv=5, scoring='r2')), scores.mean(), scores.std()


def get_backbone(model:str, pretrained:bool=False):
    if model.lower() == "alexnet":
        layers = [3, 6, 8, 10, 12]
        features = list(alexnet(weights=AlexNet_Weights.DEFAULT).features.children())
        return [nn.Sequential(*features[:layer+1]) for layer in layers], layers
    elif model.lower() == "resnet18":
        layers = [4, 5, 6, 7]
        r = resnet18(weights=ResNet18_Weights.DEFAULT)
        features = [r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4]
        return [nn.Sequential(*features[:layer+1]) for layer in layers], layers
    elif model.lower() == "vgg19":
        layers = [0, 5, 10, 19, 28]
        features = list(vgg19(weights=VGG19_Weights.DEFAULT).features.children())
        return [nn.Sequential(*features[:layer+1]) for layer in layers], layers
    elif model.lower() == "cornets":
        layers = [0, 1, 2, 3]
        c = cornet.cornet_s(pretrained=pretrained).module
        features = [c.V1, c.V2, c.V4, c.IT]
        return [nn.Sequential(*features[:layer+1]) for layer in layers], layers
    else:
        raise ValueError("Model not supported")

def plot_scores(results:dict):
    models = ['alexnet', 'resnet18', 'vgg19', 'cornets']
    model_data = {model: {} for model in models}

    for key, value in results.items():
        for model in models:
            if model in key:
                model_data[model][key] = value
                break

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12), sharey=True)
    axes = axes.flatten()
    colors = ['b', 'g', 'r', 'c']

    for idx, (model, ax) in enumerate(zip(models, axes)):
        layers = []
        scores = []
        for layer, metrics in model_data[model].items():
            layers.append(layer)
            scores.append(metrics['scores'])
        
        ax.violinplot(scores, showmeans=True)
        ax.set_title(f'{model.capitalize()} R2 Scores')
        ax.set_xlabel('Layer')
        ax.set_ylabel('R2 Score')
        ax.set_xticks(range(1, len(layers) + 1))
        ax.set_xticklabels(layers)
        ax.grid(True)
        ax.tick_params(axis='x', rotation=0)

    fig.suptitle('R2 Scores vs. Model Layers', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("direct-decoding.png")

if __name__ == "__main__":
    PRETRAINED = getenv('PRETRAINED', 1)
    VAL_DATASET_PATH = getenv('VAL_DATASET_PATH', "/home/elwa/research/datasets/varying_size_and_object_4val/test/real/set_0")
    BS = getenv('BS', 100)
    SAMPLES = getenv('SAMPLES', BS * 8)
    DEVICE = getenv('DEVICE', 'cuda')

    assert SAMPLES % BS == 0, "SAMPLES must be a multiple of BS"
    results = {}
    for model in ["alexnet", "resnet18", "vgg19", "cornets"]:
        sliced_backbones, layers = get_backbone(model, PRETRAINED)
        if model == "vgg19": SAMPLES = SAMPLES // 2
        for sliced_backbone, layer in zip(sliced_backbones, layers):
            print(f"{model} - {layer}")
            dataloader = get_dataloader(VAL_DATASET_PATH, bs=BS, num_workers=getenv("NUM_WORKERS", 4))
            sliced_backbone.to(DEVICE)
            scores, avg, std = direct_decoding(sliced_backbone, dataloader, SAMPLES // BS)
            results[f"{model}-{layer}"] = {
                "scores": scores,
                "avg": avg,
                "std": std
            }
            torch.cuda.empty_cache()
    with open("direct-decoding.pkl", "wb") as f: pickle.dump(results, f)
    plot_scores(results) if getenv("PLOT", 1) else None
