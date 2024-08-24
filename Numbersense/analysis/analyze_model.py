import json, re, os
from typing import Any, Optional

import hdbscan
import numpy as np
import torch
import torchvision.transforms as transforms
from numpy import ndarray
from scipy.optimize import curve_fit
from sklearn.metrics.pairwise import pairwise_distances
from torch import nn
from torch.utils.data import DataLoader

from Numbersense.analysis.utilities import ModelHelper, _load_transform_dict
from Numbersense.config import Dataloader_Parameters
from Numbersense.data_loaders.validation_dataloader import ValidationDataloader
from Numbersense.utilities.helpers import get_compute, getenv


class AnalyzeModel:
    def __init__(
        self,
        dataloader_parameters: Dataloader_Parameters,
        model_path: str,
        dataset_path: str,
        model_objective_function: str,
        embedding_net: nn.Module,
        concat: bool,
        data_objective_function: Optional[str],
        state: str = "final.pt",
    ):
        self.dataloader_parameters = dataloader_parameters
        self.model_path = model_path
        self.state_path = os.path.join(model_path, state)
        self.dataset_path = dataset_path
        self.model_objective_function = model_objective_function
        self.data_objective_function = data_objective_function
        self.embedding_net = embedding_net
        self.concat = concat
        self.device = get_compute()

        # Load the model with respect to given parameters.
        self.model = AnalyzeModel.load_model(
            model_path=self.state_path,
            embedding_net=self.embedding_net,
        )

        print("\nLoaded model " + self.model.name + "!")

        self.model.to(self.device)
        self.state_name = state.split(".")[0]

    def save_predictions(
        self,
    ):
        save_folder = os.path.join(
            self.model_path,
            "validation_results",
            f"{self.data_objective_function if self.data_objective_function else self.model_objective_function}",
        )
        os.makedirs(save_folder, exist_ok=True)

        if not os.path.exists(os.path.join(save_folder, "correct_per_action.json")):
            dataloader = AnalyzeModel.load_dataloader(
                self,
                dataloader_type="as_pairs",
                dataset_folder=self.dataset_path,
                concat=self.concat,
            )

            print("\nPredictions will be saved in " + save_folder)

            predicted_actions, correct_per_action = self.generate_predictions(
                dataloader=dataloader, save_folder=save_folder, concat=self.concat
            )
            np.save(
                os.path.join(save_folder, "predicted_actions.npy"), predicted_actions
            )
            json.dump(
                correct_per_action,
                fp=open(os.path.join(save_folder, "correct_per_action.json"), "w"),
                indent=1,
            )
        else:
            print(
                "\nThis dataset has already been used for testing! The corresponging files exist!"
            )
            print("Existing `correct_per_action.json`:")
            f = open(os.path.join(save_folder, "correct_per_action.json"), "r")
            correct_per_action = json.load(f)
            f.close()
        return correct_per_action

    def save_embeddings(self, concat: bool = False):
        """Generates model embeddings which are required for creating parts of the plots."""
        # Predicions are already in place.
        save_folder = os.path.join(
            self.model_path,
            "validation_results",
            f"{self.data_objective_function if self.data_objective_function else self.model_objective_function}",
        )
        os.makedirs(save_folder, exist_ok=True)

        if not os.path.exists(os.path.join(save_folder, "embeddings.npy")):
            dataloader = AnalyzeModel.load_dataloader(
                self,
                dataset_folder=self.dataset_path,
                dataloader_type="as_pairs",
                concat=concat,
            )

            print("\nEmbeddings will be saved to " + save_folder)

            g_embeddings, g_counts, g_actions = self.generate_embeddings(dataloader)
            np.save(os.path.join(save_folder, "embeddings.npy"), g_embeddings)
            np.save(
                os.path.join(save_folder, "embeddings_counts.npy"),
                np.int8(g_counts),
            )
            np.save(
                os.path.join(save_folder, "embeddings_actions.npy"),
                np.int8(g_actions),
            )
        else:
            print("\nEmbedding files already exist!")

    @staticmethod
    def load_model(
        model_path: str,
        embedding_net: nn.Module = None,
    ):
        # Loads the actual model.
        helper = ModelHelper(
            model_path,
            embedding_net=embedding_net,
        )

        return helper.model

    def load_analysis(self, analysis_path: str, target_file: str) -> dict:
        """Loads model or model embeddings respecitvely.

        :param dict dataset_params: The parameters of the dataset to be used.
        :param str target_file: The stored model results to be loaded.
        :return: The loaded analysis as an object.
        """
        assert target_file is not None

        analysis_obj = dict()
        if os.path.exists(os.path.join(analysis_path, target_file)):
            if target_file.split(".")[-1] == "json":
                with open(os.path.join(analysis_path, target_file), "r") as f:
                    analysis_obj = json.load(f)
            elif target_file.split(".")[-1] == "npy":
                analysis_obj = np.load(os.path.join(analysis_path, target_file))
        else:
            if target_file.split(".")[-1] == "json":
                self.save_predictions()
                with open(os.path.join(analysis_path, target_file), "r") as f:
                    analysis_obj = json.load(f)
            elif target_file.split(".")[-1] == "npy":
                self.save_embeddings()
                analysis_obj = np.load(os.path.join(analysis_path, target_file))
            else:
                raise Exception("\nUnknown file extension: " + target_file)

        return analysis_obj

    def load_dataloader(self, dataloader_type: str, dataset_folder: str, concat: bool):
        trans = transforms.Compose([transforms.ToTensor()])

        if dataloader_type == "as_pairs":
            loader_obj = ValidationDataloader(
                data_dir=dataset_folder,
                test_set_len=self.dataloader_parameters.test_pair_num,
                transform=trans,
                concat=concat,
                objective_function=self.model_objective_function,
            )
            dataloader = torch.utils.data.DataLoader(
                loader_obj,
                batch_size=16,
                shuffle=False,
                num_workers=getenv("NUM_WORKERS", os.cpu_count()),
            )
        else:
            raise Exception("dataloader_type is not recognized!")
        return dataloader

    def generate_embeddings(self, dataloader: DataLoader):
        """Generates the model embeddings used for creating parts of the plots.

        :param DataLoader dataloader: The dataloader object used to load data.
        :return: The generated embeddings, the actions that took place and the number of objects
        that was investigated at each step.
        """

        print("\nGenerating Embeddings...")

        torch.manual_seed(0)
        self.model.eval()
        self.model.embedding_net.eval()

        with torch.no_grad():
            actions = dataloader.dataset.total_actions
            counts = dataloader.dataset.total_counts
            embeddings = []
            emb_count = 0
            for imgs, _, _ in dataloader:
                if emb_count % 50 == 0 and getenv('VERBOSE', 1):
                    print("\nLooking at img-batch " + str(emb_count) + " ...")
                gpu_embedding = self.model.get_embedding(
                    imgs[0][:, :3, :, :].to(self.device)
                )  # Result after ff-run
                embeddings.append(gpu_embedding.to(torch.float).cpu().numpy())
                emb_count += 1
            embeddings = np.concatenate(embeddings)
        counts = np.array([i[0] for i in counts])

        return embeddings, counts, actions

    def generate_predictions(
        self, dataloader: DataLoader, save_folder: str, concat: bool
    ):
        """Generates the model predictions used for creating parts of the plots.

        :param Dataloader dataloader: The dataloader object used to load data.
        :param str save_folder: The path to the folder where the results will be stored.
        :param bool concat: Specifies whether or not to use concatenated images as model input.
        :return: The predicted actions as well as the ratio of correctness per object cardinality.
        """

        print("\nGenerating Predictions...")

        self.model.to(self.device)
        self.model.eval()
        self.model.embedding_net.eval()

        total = 0
        num_correct = 0
        predicted_actions = []
        correct_per_cardinality_action = {}

        target_transform = _load_transform_dict(self.model_objective_function)

        ratio = 0
        pred_errors = 0
        with torch.no_grad():
            # 'batch_size' parameter determined in AnalyzeModel.load_dataloader()
            for batch_idx, (data, targets, counts) in enumerate(dataloader):
                # data[0].size(): torch.Size([16, 3, 244, 244])
                # targets.size(): torch.Size([16, 3])
                if batch_idx % 50 == 0 and getenv('VERBOSE', 1):
                    print(
                        "\nEntering batch "
                        + str(batch_idx + 1)
                        + "/"
                        + str(len(dataloader))
                        + "..."
                    )
                keys = []

                # Send all targets to the used device.
                targets = targets.type(torch.LongTensor).to(self.device)
                batch_target_actions = torch.argmax(targets, dim=1)

                # Send all images to the used device and calculate model output.
                if concat:
                    data = data.to(self.device)
                    output = self.model(data)
                else:
                    for k in range(len(data)):
                        data[k] = data[k].to(self.device)
                    output = self.model(data[0], data[1])
                batch_predicted_actions = torch.argmax(output, dim=1)

                # output[0].size(): torch.Size([3])
                # batch_predicted_actions.size(): torch.Size([16])
                # batch_target_actions.size(): torch.Size([16])

                # Compare
                correct_indices = batch_predicted_actions == batch_target_actions
                predicted_actions.append(batch_predicted_actions)

                # Track the number of correct predictions
                num_correct += torch.sum(correct_indices).item()
                total += len(batch_predicted_actions)

                # counts: tensor([[0, 1], [1, 0], [0, 0], [0, 1], [1, 0], [0, 0], [0, 1], [1, 1], [1, 0], [0, 1],
                #                 [1, 1], [1, 2], [2, 1], [1, 1], [1, 1], [1, 0]], dtype=torch.int32)

                # Number correct per action
                for k, correct_indice in enumerate(correct_indices):
                    # Generate a key that describes the cardinality of the first image and the action that took place
                    # ie. 2>, 3<, 3-
                    key = (
                        str(counts[0][k].item())  # TODO have a look at this again
                        + target_transform[str(targets[k].cpu().tolist())]
                    )
                    keys.append(key)

                    # Add to dict if key exists, create dict entry otherwise
                    vals = correct_per_cardinality_action.get(key, [])
                    if len(vals) == 0:
                        # vals := [correct_or_not, count_of_predictions]
                        vals: list = [int(correct_indice), 1]
                        correct_per_cardinality_action[key] = vals
                    else:
                        # Adds one if k is in correct_indices
                        vals[0] += int(correct_indice)
                        vals[1] += 1

                # Sort dict by numerosity:
                correct_per_cardinality_action = dict(
                    sorted(correct_per_cardinality_action.items())
                )

                ratio = num_correct / total * 100
                pred_errors = total - num_correct
                if batch_idx % 50 == 0 and getenv('VERBOSE', 1):
                    print(
                        "\nRatio of correct predictions: " + str(round(ratio, 2)) + "%"
                    )
                    print("Num. of wrong predictions so far:", pred_errors, "\n")

        correct_per_cardinality_action = dict(
            sorted(
                correct_per_cardinality_action.items(),
                key=lambda x: int(re.search(r"\d+", x[0]).group()),
            )
        )
        prediction_stats = {
            "total_predictions": total,
            "correct_predictions": num_correct,
            "wrong_predictions": pred_errors,
            "ratio": round(ratio, 2),
        }
        json.dump(
            prediction_stats,
            fp=open(os.path.join(save_folder, "total_accuracy.json"), "w"),
            indent=1,
        )

        # predicted_actions[0]: tensor([2, 0, 1, 2, 0, 1, 2, 1, 0, 2, 1, 2, 0, 1, 1, 0])
        # correct_per_cardinality_action: {'0>': [95, 95], '1<': [87, 87], '0-': [74, 74], '1-': [83, 83], ...}

        # Add ratio for each key-action
        for k in correct_per_cardinality_action:
            num_correct = correct_per_cardinality_action[k][0]
            total = correct_per_cardinality_action[k][1]
            ratio = round(num_correct / total * 100, 2)
            correct_per_cardinality_action[k].append(ratio)

        predicted_actions = torch.cat(predicted_actions).cpu().numpy()

        return predicted_actions, correct_per_cardinality_action

    @staticmethod
    def hdbscan_clustering(embeddings: dict, min_cluster_size: int = 20):
        """Clusters the model embeddings to groups corresponding to a certain amount of objects.

        :param dict embeddings: The model's embeddings to be clustered.
        :param int min_cluster_size: The min. amount of clusters.
        :return: Datapoint indices and labels of the clusters.
        """
        clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        clustering.fit(embeddings)

        base = [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "Y",
            "Z",
        ]
        cluster_labels_map = {}

        k = 0
        for cl in np.unique(clustering.labels_):
            if cl == -1:
                cid = "-1"
            else:
                cid = base[k % len(base)] + str(int(k / len(base)))
                k += 1
            cluster_labels_map[cl] = cid

        cluster_embedding_indices_dict = {}
        for ind, cluster_id in enumerate(clustering.labels_):
            if cluster_labels_map[cluster_id] not in cluster_embedding_indices_dict:
                cluster_embedding_indices_dict[cluster_labels_map[cluster_id]] = {ind}
            else:
                cluster_embedding_indices_dict[cluster_labels_map[cluster_id]].add(ind)

        labels = np.empty(shape=embeddings.shape[0], dtype=object)
        for key in cluster_embedding_indices_dict.keys():
            indices = np.array(list(cluster_embedding_indices_dict[key]))
            labels[indices] = key

        return cluster_embedding_indices_dict, labels

    @staticmethod
    def generate_confusion_matrix(
        predicted_labels: list,
        true_labels: list,
        remove_uncertain: bool = False,
        sort: bool = True,
    ):
        """Plots a matrix showing absolute and relative numerosities for a given range of object numbers.

        :param list predicted_labels: List of the predicted labels of the clusters.
        :param list true_labels: List of the true labels of the clusters.
        :param bool remove_uncertain: Specifies whether to remove uncertain predictions (marked as x in the clustering
        plot) or not.
        :param bool sort: Specifies whether to sort the prediced labels or not.
        :return: Datapoints in the confusion matrix, the unique predicted labels and the unique true labels.
        """
        unique_true_labels = np.unique(true_labels)
        unique_predicted_labels = np.unique(predicted_labels)

        ind_map = dict(
            zip(unique_predicted_labels, range(len(unique_predicted_labels)))
        )
        confusion_matrix = np.zeros(
            shape=(len(unique_predicted_labels), len(unique_true_labels))
        )  # (rows, cols)
        for i, un in enumerate(unique_true_labels):
            mask = true_labels == un
            unique_elements, counts_elements = np.unique(
                predicted_labels[mask], return_counts=True
            )
            inds = [ind_map[u] for u in unique_elements]
            confusion_matrix[inds, i] = counts_elements

        if sort:
            order = [
                np.max(np.where(confusion_matrix[i] > 0)[0])
                for i in range(len(unique_predicted_labels))
            ]
            sort_inds = sorted(
                range(len(unique_predicted_labels)), key=lambda x: order[x]
            )
            confusion_matrix = confusion_matrix[sort_inds, :]
            unique_predicted_labels = unique_predicted_labels[sort_inds]
            ind_map = dict(
                zip(unique_predicted_labels, range(len(unique_predicted_labels)))
            )

        if remove_uncertain and ind_map.get("-1") is not None:
            rem_ind = ind_map["-1"]
            confusion_matrix = np.delete(confusion_matrix, rem_ind, 0)
            unique_predicted_labels = unique_predicted_labels[
                unique_predicted_labels != "-1"
            ]

        return confusion_matrix, unique_predicted_labels, unique_true_labels

    def rescale_embeddings(
        self,
        embeddings: list,
        counts: list,
        rescaling_limit: int,
        get_indices_dict: bool = False,
    ):
        """Rescales the embeddings so that all have a uniformly calculated distance to the origin point. This way, the
        resulting representations can be mapped to a scale of the true numerosity.

        :param list embeddings: The list of the embeddings which should be transformed.
        :param list counts: The list the number of objects corresponding to a cetain embedding.
        :param int rescaling_limit: The max. number of the true numberosity to map against.
        :param bool get_indices_dict: Specifies whether to return a dictionary of the representation indices or not.
        :return: The rescaled embeddings and opt. a dictionary of the representation indices.
        """
        un_counts = np.unique(counts)

        estimate_dist_dict = {}
        for j in range(rescaling_limit):
            uc = un_counts[j]
            mask = counts == uc
            mask_to = counts == (uc + 1)
            dists = pairwise_distances(embeddings[mask], embeddings[mask_to]).flatten()
            estimate_dist_dict[str(j)] = dists

        # np.mean(embeddings[:10][counts[:10] == 0], axis=0): [ 22.6409   -24.313528]

        emb_zeros = [0 for _ in range(self.model.embedding_dimension)]
        estimated_fit_scalar = np.mean(list(estimate_dist_dict.values()))
        offset = emb_zeros - np.mean(embeddings[counts == 0, :], axis=0)

        rescaled_distances_from_origin = {}
        mean_dists = []
        indices = np.arange(len(embeddings))
        indices_dict = {}
        for k in range(len(un_counts)):
            uc = un_counts[k]
            mask = counts == uc
            dists = (
                pairwise_distances(
                    (embeddings[mask] + offset), np.array(emb_zeros).reshape(1, -1)
                )
                / estimated_fit_scalar
            )
            rescaled_distances_from_origin[k] = dists.flatten()
            indices_dict[k] = indices[mask]
            mean_dists.append(np.mean(dists.flatten()))

        if get_indices_dict:
            return rescaled_distances_from_origin, indices_dict
        else:
            return rescaled_distances_from_origin

    @staticmethod
    def fit_power_function(rescaled_embeddings_dict: dict):
        """Plots a power function to given a given set of embedding values.

        :param dict rescaled_embeddings_dict: The dictionary of the given embeddings to be plotted as power funciton.
        :return: The embedding datapoints on the power function.
        """

        def power_function(n: float, beta: float, alpha: float):
            """The formula of the power function to use.

            :param float n: The embedding value to be transformed.
            :param float beta: The exponent.
            :param float alpha: The coefficient.
            :return: The embedding value after applying the power function.
            """
            return alpha * n**beta

        def get_r2(func: Any, xvals_1: ndarray, yvals_1: ndarray, popt: ndarray):
            """To be done."""

            y_mean = np.mean(yvals_1)  # Mean of y values
            ss_tot = np.sum((yvals_1 - y_mean) ** 2)  # Sum of squares
            ss_res = np.sum((yvals_1.flatten() - func(xvals_1.flatten(), *popt)) ** 2)
            r2 = 1 - ss_res / ss_tot
            return r2

        xvals = np.array(
            [
                [key] * len(rescaled_embeddings_dict[key])
                for key in rescaled_embeddings_dict.keys()
            ],
            dtype=list,
        )
        yvals = np.array(list(rescaled_embeddings_dict.values()), dtype=list)
        opt_values, _ = curve_fit(
            f=power_function, xdata=xvals.flatten(), ydata=yvals.flatten()
        )
        calc_r2 = get_r2(
            func=power_function, xvals_1=xvals, yvals_1=yvals, popt=opt_values
        )

        return opt_values, calc_r2
