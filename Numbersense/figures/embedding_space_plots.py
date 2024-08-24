"""Plot all the embedding diagrams per given model."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from numpy import ndarray
from sklearn.decomposition import PCA


def visualize_embedding_space_with_labels(
    embeddings: ndarray, labels: ndarray, ax: Any, plot_params: dict = None
):
    """Plots all the embedding datapoints with corresponding labels.

    :param ndarray embeddings: The datapoints to be plotted.
    :param ndarray labels: The corresponding labels.
    :param Any ax: The sub-plot to create the visualization.
    :param dict plot_params: Parameters to take into account during creation of the plots.
    """
    # embeddings.shape: (16200, 2)
    # labels.shape: (16200, 2)

    if plot_params is None:
        plot_params = {}

    # PCA:
    if embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        pca.fit(embeddings)
        pca.transform(embeddings)

    col_pal = plot_params.get("color_palette", "crest")
    my_cmap = None
    if col_pal == "crest":
        my_cmap = sns.color_palette("crest", as_cmap=True)
    elif col_pal == "bright":
        cp = sns.color_palette("bright").as_hex()
        my_cmap = ListedColormap(cp)

    unique_labels = np.unique(labels)  # unique_labels.shape: (9,)

    inset = plot_params.get("inset", False)

    axins = None
    inset_limit = None
    if inset:
        inset_axes = plot_params.get("inset_axes", [0.55, 0.55, 0.4, 0.4])
        inset_limit = plot_params.get("inset_limit", 5)
        axins = ax.inset_axes(inset_axes)

    early_stop = plot_params.get("early_stop", None)
    inset_count = 0
    for i, ul in enumerate(unique_labels):
        if early_stop is not None and i >= early_stop:
            break
        mask = labels == ul  # len(mask): 16200

        if col_pal == "crest":
            cmap_val = i / max(unique_labels)
        else:
            cmap_val = i % my_cmap.N

        if type(ul) is str:
            label = ul
        else:
            label = int(ul)

        # Only relevant for Clustering run
        if ul == -1 or ul == "-1":
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                color="black",
                marker="x",
                alpha=1,
                s=5,
                zorder=5,
                label="X",
            )
            continue

        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            color=my_cmap(cmap_val),
            alpha=1,
            s=5,
            zorder=10,
            label=label,
        )
        if inset and inset_count < inset_limit:
            if plot_params.get("inset_labels", None) is None or ul in plot_params.get(
                "inset_labels"
            ):
                inset_count += 1
                axins.scatter(
                    embeddings[mask, 0],
                    embeddings[mask, 1],
                    color=my_cmap(cmap_val),
                    alpha=1,
                    s=5,
                    zorder=15,
                )

    if inset:
        if labels.dtype == np.int8:
            mask = labels == -1
        else:
            mask = labels == "-1"

        sub_mask = (
            (axins.axis()[0] <= embeddings[mask, 0])
            & (embeddings[mask, 0] <= axins.axis()[2])
            & (axins.axis()[1] <= embeddings[mask, 1])
            & (embeddings[mask, 1] <= axins.axis()[3])
        )

        mask[mask] = sub_mask
        axins.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            color="black",
            marker="x",
            alpha=1,
            s=5,
            zorder=5,
            label="X",
        )

        axins.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.indicate_inset_zoom(axins, zorder=5, edgecolor="black", alpha=0.8)

    if plot_params.get("annotate", None) is not None:
        ax.annotate(
            text=plot_params["annotate"],
            fontsize=17,
            weight="bold",
            xy=(0.025, 0.89),
            xycoords="figure fraction",
        )

    plt.xlabel("dim 1", fontsize=18)  # embeddings.shape[1][0]
    plt.ylabel("dim 2", fontsize=18)  # embeddings.shape[1][1]
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.locator_params(axis="x", nbins=4)
    handles, labels = ax.get_legend_handles_labels()
    # del handles[0], labels[0]

    # if "inset" in labels[0]:
    #    del handles[0], labels[0]

    if plot_params.get("legend_visible", True):
        ax.legend(
            handles=handles,
            labels=labels,
            loc="best",
            fontsize=10,
            ncol=2 if len(handles) > 10 else 1,
            markerscale=2,
        )

    if plot_params.get("apply_tight_layout", True):
        plt.tight_layout()

    return ax


def visualize_confusion_matrix(
    confusion_matrix: ndarray, ax: Any, plot_params: dict = None
):
    """Plots the confusion matrix of a set of given datapoints.

    :param confusion_matrix: The array of transformed model embeddings.
    :param Any ax: The sub-plot to create the visualization.
    :param dict plot_params: Parameters to take into account during creation of the plots.
    """
    if plot_params is None:
        plot_params = {}

    my_cmap = plt.get_cmap(plot_params.get("cmap", "viridis"))
    my_cmap.set_bad(my_cmap(0.1))

    # Normalize heatmap, replace 0 with 0.0001 for log scale
    confusion_matrix /= np.sum(confusion_matrix, axis=0)
    confusion_matrix[confusion_matrix == 0] = np.nan

    ylabel = "Number of objects"
    xlabel = "Cluster labels"

    column_names = plot_params.get("column_names", None)
    row_names = plot_params.get("row_names", None)

    if column_names is None:
        column_names = range(confusion_matrix.shape[1])
    if row_names is None:
        row_names = range(confusion_matrix.shape[0])

    if plot_params.get("transpose", False):
        im = ax.imshow(np.log10(confusion_matrix), aspect=0.05, cmap=my_cmap, vmin=-4)
        temp = column_names
        column_names = row_names
        row_names = temp
        temp = xlabel
        xlabel = ylabel
        ylabel = temp
    else:
        print("test")
        im = ax.imshow(np.log10(confusion_matrix.T), aspect=0.2, cmap=my_cmap, vmin=-4)

    if row_names is not None:
        if len(row_names) > 10:
            ticks = list(map(int, list(range(0, len(row_names), 2))))
            ax.set_yticks(ticks=ticks)
            ax.set_yticklabels(row_names[0::2])
        else:
            ax.set_yticks(ticks=list(range(len(row_names))))
            ax.set_yticklabels(row_names)

    if column_names is not None:
        if len(column_names) > 10:
            ticks = list(map(int, list(range(0, len(column_names), 2))))
            ax.set_xticks(ticks=ticks)
            ax.set_xticklabels(column_names[0::2])
        else:
            ax.set_xticks(ticks=list(range(0, len(column_names), 1)))
            ax.set_xticklabels(column_names[0::1])

    if plot_params.get("add_labels", True):
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_xlabel(xlabel, fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

    if plot_params.get("add_colorbar", True):
        cbar = plt.colorbar(im, shrink=plot_params.get("colbar_shrink", 0.8))
        cbar.ax.tick_params(labelsize=14)
        cbar.set_ticks([0, -1, -2, -3, -4])

    if plot_params.get("annotate", None) is not None:
        ax.annotate(
            text=plot_params["annotate"],
            fontsize=17,
            weight="bold",
            xy=plot_params.get("annotate_pos", (0.028, 0.92)),
            xycoords="figure fraction",
        )

    return ax
