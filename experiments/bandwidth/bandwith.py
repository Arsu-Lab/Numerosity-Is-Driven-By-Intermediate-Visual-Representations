import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from Numbersense.utilities.helpers import getenv

def create_confusion_matrix(embeddings, embedding_labels):
    num_val_samples = int(0.2 * embeddings.shape[0]) # 20/80 split
    embeddings_train, embeddings_val = embeddings[:-num_val_samples], embeddings[-num_val_samples:]
    labels_train, labels_val = embedding_labels[:-num_val_samples], embedding_labels[-num_val_samples:]

    model = LogisticRegression(max_iter=1000)
    model.fit(embeddings_train, labels_train)

    labels_pred = model.predict(embeddings_val)
    return confusion_matrix(labels_val, labels_pred), accuracy_score(labels_val, labels_pred)

if __name__ == "__main__":
    # Default: Trained on Dataset1
    EXPERIMENT_DIR = getenv("EXPERIMENT_DIR", "hierarchical")
    SAVE_DIR = getenv("SAVE_DIR", ".")
    DATASET = getenv("DATASET", "varying_size_fixed_image_fixed_between")
    MODEL = getenv("MODEL", "VGG19-3-pretrained-frozen")
    SET = getenv("SET", 0)
    SET_COUNT = getenv("SET_COUNT", -1) # e.g. Set to 10 to do the first 10 sets

    confusion_matrices, accuracies = [], []
    for set_idx in range(SET, SET+1) if SET_COUNT == -1 else range(SET_COUNT):
        print(EXPERIMENT_DIR)
        embeddings = np.load(f"{EXPERIMENT_DIR}/trained_models/{DATASET}/real/{MODEL}/set_{set_idx}/validation_results/{DATASET}/embeddings.npy")
        embedding_labels = np.load(f"{EXPERIMENT_DIR}/trained_models/{DATASET}/real/{MODEL}/set_{set_idx}/validation_results/{DATASET}/embeddings_counts.npy")
        cm, accuracy = create_confusion_matrix(embeddings, embedding_labels)
        confusion_matrices.append(cm)
        accuracies.append(accuracy)
        if set_idx == 0:
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix for Set {set_idx}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.savefig(f"plot.png")
    print(f"Accuracy on validation dataset: {sum(accuracies)/len(accuracies):.2f}")

    confusion_matrices = np.array(confusion_matrices)
    np.save(f"{SAVE_DIR}/{MODEL}-confusion-matrices.npy", confusion_matrices)
