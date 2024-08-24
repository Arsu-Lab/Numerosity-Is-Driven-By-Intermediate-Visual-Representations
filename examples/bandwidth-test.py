import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

from Numbersense.utilities.helpers import getenv

def bandwidth_test(embeddings:np.ndarray, labels:np.ndarray):
    distances = {}
    for numerosity in np.unique(labels):
        numerosity_indices = np.where(labels == numerosity)
        numerosity_embeddings = embeddings[numerosity_indices]
        model = LinearRegression()
        X = numerosity_embeddings[:, 0].reshape(-1, 1)
        Y = numerosity_embeddings[:, 1]
        model.fit(X, Y)
        predictions = model.predict(X)
        distances[numerosity] = np.abs(Y - predictions)

    numerosities = list(distances.keys())
    all_distances = [distances[num] for num in numerosities]

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=all_distances)
    plt.xticks(ticks=range(len(numerosities)), labels=numerosities)
    plt.xlabel('Numerosity')
    plt.ylabel('Distances')
    plt.title('Distances - Real scene images (Dataset 1) - AlexNet-4-pretrained-frozen')
    plt.savefig(getenv("FIG_SAVE_PATH", "/home/elwa/repos/bandwidth_test/distances"))

if __name__ == "__main__":
    embeddings_path = "/home/elwa/repos/bandwidth_test/trained_models/numerosity/real/fixed_image_fixed_between-alexnet-4-pretrained-frozen/set_0/validation_results/varying_size_fixed_image_fixed_between/embeddings.npy"
    labels_path = "/home/elwa/repos/bandwidth_test/trained_models/numerosity/real/fixed_image_fixed_between-alexnet-4-pretrained-frozen/set_0/validation_results/varying_size_fixed_image_fixed_between/embeddings_counts.npy"
    with open(embeddings_path, 'rb') as f: embeddings = np.load(f)
    with open(labels_path, 'rb') as f: labels = np.load(f)
    bandwidth_test(embeddings, labels)

