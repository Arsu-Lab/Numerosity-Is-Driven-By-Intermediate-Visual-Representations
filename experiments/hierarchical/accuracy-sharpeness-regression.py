import json
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from Numbersense.analysis.analyze_results import total_classification_accuracy, total_embedding_score
from Numbersense.utilities.helpers import getenv

def to_numpy(accuracies:dict, embedding_scores:dict, selector:str) -> dict:
    common_keys = set(accuracies.keys()) & set(embedding_scores.keys())
    shape = (len(common_keys), 2, 10) if selector == "list" else (len(common_keys), 2)
    data = np.zeros(shape)
    
    for idx, key in enumerate(common_keys):
        if selector == "average":
            accuracy = accuracies[key][selector] * 0.01 # Acc in %
            embedding = embedding_scores[key][selector]
            data[idx, :] = [embedding, accuracy]
        else:
            accuracy = list(map(lambda x: x * 0.01, accuracies[key]["list"])) # Acc in %
            accuracy += [0] * (10 - len(accuracy)) # Pad to 0s of len 10
            embedding = embedding_scores[key]["list"] + [0]*(10-len(embedding_scores[key]["list"]))
            data[idx, :, :] = [embedding, accuracy]
    return data

def plot_acc_vs_embd_score(model, X, Y, accuracies, dataset):
    p_f = [i for i, k in enumerate(accuracies.keys()) if "pretrained-frozen" in k]
    p_u = [i for i, k in enumerate(accuracies.keys()) if "pretrained-unfrozen" in k]
    u_f = [i for i, k in enumerate(accuracies.keys()) if "untrained-frozen" in k]
    u_u = [i for i, k in enumerate(accuracies.keys()) if "untrained-unfrozen" in k]

    plt.scatter(X[p_f], Y[p_f], marker='o', label='pretrained frozen')
    plt.scatter(X[p_u], Y[p_u], marker='s', label='pretrained unfrozen')
    plt.scatter(X[u_f], Y[u_f], marker='^', label='untrained frozen')
    plt.scatter(X[u_u], Y[u_u], marker='*', label='untrained unfrozen')
    plt.plot(X, model.predict(X), color='red', label='Prediction Line')
    plt.xlabel("Embedding Score")
    plt.ylabel("Action Classification Accuracy")
    plt.legend()
    plt.savefig(f"acc_vs_embd_score_{dataset.lower()}_val{validation_numerosity}.png")

# Internally the datasets are called objective functions and have a different naming convention
dataset_mapping = {
    "Dataset1": "varying_size_fixed_image_fixed_between",
    "Dataset2": "varying_size",
    "Dataset3": "varying_size_and_object",
    "Dataset4": "varying_size_and_object_and_background",
}

if __name__ == "__main__":
    experiment_path = getenv("EXPERIMENT_PATH", "hierarchical")

    dataset = getenv("DATASET", "Dataset1")
    assert dataset in dataset_mapping.keys(), f"Dataset {dataset} not found. Available datasets: {dataset_mapping.keys()}"

    val_on_4 = getenv("VAL_ON_4", 0) # Default is val on 8
    validation_numerosity = 4 if val_on_4 else 8
    selector = "average" if getenv("AVERAGE", 0) else "list" # Whether to calculate the scores for each model or the set averages

    total_experiment_accuracy_path = getenv("SAVE_ACC_PATH", f"accuracies_{dataset.lower()}_val{validation_numerosity}.json")
    total_experiment_embedding_score_path = getenv("SAVE_EMB_PATH", f"embbeding_scores_{dataset.lower()}_val{validation_numerosity}.json")
    merged_acc_embd_path = getenv("MERGED_ACC_EMBD_PATH", f"combined_{dataset.lower()}_val{validation_numerosity}.json")

    # Gather the experiment accuracies and embedding score
    print("Gathering accuracies and embedding scores...")
    accuracies = total_classification_accuracy(experiment_path, dataset_mapping[dataset], cross_validation_obj=dataset_mapping[dataset]+"_4val" if val_on_4 else None, save_path=total_experiment_accuracy_path)
    embedding_scores = total_embedding_score(experiment_path, dataset_mapping[dataset], cross_validation_obj=dataset_mapping[dataset]+"_4val" if val_on_4 else None, save_path=total_experiment_embedding_score_path)

    combined = {}
    for k in set(accuracies.keys()) & set(embedding_scores.keys()):
        combined[k] = {"accuracies": accuracies[k][selector], "embedding_scores": embedding_scores[k][selector]}
    with open(merged_acc_embd_path, "w") as f: json.dump(combined, f, ensure_ascii=False, indent=4)

    # Construct numpy array of shape (n_models, 2, n_sets) or (n_models, 2)
    print("Constructing numpy array...")
    data = to_numpy(accuracies, embedding_scores, selector)

    # X := Embedding scores, Y := Accuracies
    X, Y = data[:, 0].reshape(-1, 1), data[:, 1].reshape(-1, 1)

    # Filter out missing data that was set to 0
    non_zero_indices = (X != 0.0) & (Y != 0.0)
    X, Y = X[non_zero_indices].reshape(-1, 1), Y[non_zero_indices].reshape(-1, 1)
    
    # Calculate the correlation coefficient
    correlation_coefficient = np.corrcoef(X.flatten(), Y.flatten())[0, 1]
    print("Correlation Coefficient:", correlation_coefficient)

    # Plot accuracy against embedding score
    print("Fitting model and plotting...")
    model = LinearRegression().fit(X, Y)
    plot_acc_vs_embd_score(model, X, Y, accuracies, dataset)
