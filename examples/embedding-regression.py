import numpy as np
import os
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

from Numbersense.utilities.helpers import getenv

def embedding_regression(embds:np.ndarray, labels:np.ndarray, n_splits:int = 10):
    model = LinearRegression()
    scores = cross_val_score(model, embds, labels.ravel(), cv=n_splits, scoring='r2')
    return scores.mean(), scores.std()


if __name__ == "__main__":
    save_path = getenv("EXPERIMENT_PATH", "/home/elwa/repos/experiment")
    embds = np.load(os.path.join(save_path, "trained_models", "numerosity", "plain", "vanilla-perona", "set_0", "validation_results", "numerosity", "embeddings.npy"))
    labels = np.load(os.path.join(save_path, "trained_models", "numerosity", "plain", "vanilla-perona", "set_0", "validation_results", "numerosity", "embeddings_counts.npy"))

    score = embedding_regression(embds, labels)
    print(f"Mean R^2: {score[0]}, Std R^2: {score[1]}")
