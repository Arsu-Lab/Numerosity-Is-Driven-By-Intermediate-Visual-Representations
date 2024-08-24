import os, json, re
from typing import Optional
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import cross_val_score

def get_accuracy_from_json(path: str, fn = "total_accuracy.json"):
    try:
        with open(os.path.join(path, fn), 'r') as f:
            data = json.load(f)
        return float(data['ratio'])
    except FileNotFoundError:
        print(f"Accessing {Path(path).parent.parent} failed!")
        return None

def get_embedding_metric(path: str, nfold: int = 10, get_explained_variance: bool = False, embd_fn = "embeddings.npy", label_fn = "embeddings_counts.npy"):
    try:
        with open(os.path.join(path, embd_fn), 'rb') as f: embeddings = np.load(f)
        with open(os.path.join(path, label_fn), 'rb') as f: labels = np.load(f)
        pca = PCA(n_components=1)
        pca.fit(embeddings)
        X_pca = pca.transform(embeddings)
        clf = LinearRegression()
        score = cross_val_score(clf, X_pca, labels, cv=nfold, scoring='r2')
        return pca.explained_variance_ratio_[0].tolist() if get_explained_variance else np.mean(score).tolist()
    except FileNotFoundError:
        print(f"Accessing {Path(path).parent.parent} failed!")
        return None

def total_classification_accuracy(experiment_path: str, obj_function:str, background_mode:Optional[str] = None, cross_validation_obj: Optional[str] = None, regex: Optional[str] = None, save_path: Optional[str] = None):
    perf_dict = {}
    for model in os.listdir(model_dir := os.path.join(experiment_path, "trained_models", obj_function, background_mode if background_mode else "real")):
        if (not regex or (regex and re.match(regex, model))) and not model.startswith("."):
            perf_dict[model] = {}  # Initialize perf_dict[model] as an empty dictionary
            perf_dict[model]["list"], perf_dict[model]["average"], perf_dict[model]["std_deviation"] = [], 0, 0
            for validation_folder in [os.path.join(model_dir, model, set, "validation_results") for set in os.listdir(os.path.join(model_dir, model)) if not set.startswith(".")]:
                perf = get_accuracy_from_json(os.path.join(validation_folder, cross_validation_obj) if cross_validation_obj else os.path.join(validation_folder, obj_function))
                if not perf: continue
                perf_dict[model]["list"].append(perf)
            perf_dict[model]["average"] = np.mean(perf_dict[model]["list"])
            perf_dict[model]["std_deviation"] = np.std(perf_dict[model]["list"])
    if save_path:
        with open(save_path, 'w') as f: json.dump(perf_dict, f, indent=4)
    return perf_dict

def total_embedding_score(experiment_path: str, obj_function:str, background_mode: Optional[str] = None, get_explained_variance: bool = False, cross_validation_obj: Optional[str] = None, regex: Optional[str] = None, save_path: Optional[str] = None):
    perf_dict = {}
    for model in os.listdir(model_dir := os.path.join(experiment_path, "trained_models", obj_function, background_mode if background_mode else "real")):
        if (not regex or (regex and re.match(regex, model))) and not model.startswith("."):
            perf_dict[model] = {}  # Initialize perf_dict[model] as an empty dictionary
            perf_dict[model]["list"], perf_dict[model]["average"], perf_dict[model]["std_deviation"] = [], 0, 0
            for validation_folder in [os.path.join(model_dir, model, set, "validation_results") for set in os.listdir(os.path.join(model_dir, model)) if not set.startswith(".")]:
                perf = get_embedding_metric(os.path.join(validation_folder, cross_validation_obj) if cross_validation_obj else os.path.join(validation_folder, obj_function), 10, get_explained_variance)
                if not perf: continue
                perf_dict[model]["list"].append(perf)
                perf_dict[model]["average"] = np.mean(perf_dict[model]["list"]) if len(perf_dict[model]["list"]) > 1 else perf
            perf_dict[model]["std_deviation"] = np.std(perf_dict[model]["list"])
    if save_path:
        with open(save_path, 'w') as f: json.dump(perf_dict, f, indent=4)
    return perf_dict