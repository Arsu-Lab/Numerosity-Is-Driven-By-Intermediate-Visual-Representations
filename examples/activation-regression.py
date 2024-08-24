import os, json
from typing import List
from net2brain.feature_extraction import FeatureExtractor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from torchvision.models import alexnet, resnet18, vgg19
import cornet

from torch import nn
import numpy as np
from Numbersense.utilities.helpers import getenv

if getenv("DEVICE", "cpu") == "cuda":
    from cuml.linear_model import LinearRegression as GPULinearRegression
    from cuml.metrics import r2_score

from Numbersense.model.networks import EmbeddingNet
from Numbersense.analysis.analyze_model import AnalyzeModel
from Numbersense.utilities.helpers import getenv

def activation_regression(dataset_path:str, model, layers:List[int], save_path:str, pretrained:bool = False):
    device = getenv('DEVICE', "cpu").lower()
    if device == "cuda": model.to(device)
    fx = FeatureExtractor(model, device=device, pretrained=pretrained)
    fx.extract(dataset_path, save_path, layers_to_extract=layers)
    activations = os.listdir(save_path)
    scores = {}
    for layer in activations:
        layer_path = os.path.join(save_path, layer)
        if layer_path.endswith('.npz'):
            data = np.load(layer_path, allow_pickle=True)
            X = np.vstack([data[arr].flatten() for arr in list(data.files)])
            Y = np.array([int(arr_name.split("-")[1]) for arr_name in list(data.files)])
            if device == "cuda":
                scores_accum = []
                for fold in range(int(getenv('NFOLD', 10))):
                    len_fold = len(X) // 10
                    start, end = fold * len_fold, (fold + 1) * len_fold
                    X_test, Y_test = X[start:end, :], Y[start:end]
                    X_train = np.concatenate([X[:start, :], X[end:, :]], axis=0).astype(np.float64)
                    Y_train = np.concatenate([Y[:start], Y[end:]], axis=0).reshape(-1, 1).astype(np.float64)
                    model = GPULinearRegression(copy_X=False)
                    model.fit(X_train, Y_train)
                    Y_pred = model.predict(X_test)
                    scores_accum.append(r2_score(Y_test.astype(np.float64), Y_pred))
                scores[int(layer.split(".")[0])] = np.mean(scores_accum)
            else:
                model = LinearRegression()
                scores[int(layer.split(".")[0])] = cross_val_score(model, X, Y, scoring='r2', cv=10).mean()
            print(f"Layer {layer.split('.')[0]} has mean R^2 score of {scores[int(layer.split('.')[0])]}")
    return scores

def get_backbone(model:str):
    if model.lower() == "alexnet":
        return nn.Sequential(*list(alexnet().features.children())[:13]), ["3", "6", "8", "10", "12"]
    elif model.lower() == "resnet18":
        r = resnet18()
        features = [r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4]
        return nn.Sequential(*features), ['4', '5', '6', '7']
    elif model.lower() == "vgg19":
        return nn.Sequential(*list(vgg19().features.children())[:36]), ['0', '5', '10', '19', '28']
    elif model.lower() == "cornets":
        c = cornet.cornet_s().module
        features = [c.V1, c.V2, c.V4, c.IT]
        return nn.Sequential(*features), ['0', '1', '2', '3']
    else:
        raise ValueError("Model not supported")

if __name__ == "__main__":
    pretrained = getenv('PRETRAINED', 1)
    dataset_path = "/home/elwa/research/activation-regression/dataset1/test/real/set_0"
    scores_path = "/home/elwa/research/activation-regression/scores"
    os.makedirs(scores_path, exist_ok=True)
    pretrained_models = ["VGG19-5-pretrained-unfrozen", "VGG19-5-pretrained-frozen", "ResNet18-4-pretrained-frozen", "ResNet18-4-pretrained-unfrozen", "CORnetS-4-pretrained-frozen", "CORnetS-4-pretrained-unfrozen"]
    untrained_models = ["AlexNet-5-untrained", "ResNet18-4-untrained", "VGG19-5-untrained", "CORnetS-4-untrained"]
    for model_name in pretrained_models if pretrained else untrained_models:
        for set_idx in range(0, int(getenv('SET_RANGE', 10))):
            model_path = f"/home/elwa/research/activation-regression/models/real/{model_name}/set_{set_idx}/final.pt"
            if os.path.exists(model_path) or pretrained:
                save_path = f"/home/elwa/research/activation-regression/activations/{model_name}/{'pretrained' if pretrained else 'untrained'}/set_{set_idx}"
                backbone, layers = get_backbone(model_name.split("-")[0])
                model = AnalyzeModel.load_model(model_path, embedding_net=EmbeddingNet(2, backbone)).embedding_net.backbone if pretrained else backbone
                scores = activation_regression(dataset_path, model, layers=layers, save_path=save_path, pretrained=pretrained)
                scores_save_path = os.path.join(scores_path, f"{model_name}_set_{set_idx}_scores.json")
                with open(scores_save_path, 'w') as f: json.dump(scores, f)
                print(f"Scores saved to {scores_save_path}")
            else:
                print(f"Model path {model_path} does not exist. Skipping...")