"""Visualizes the internal embeddings of each given model after testing."""

import os
from typing import Optional, Union

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from torch import nn
from torchvision.models import alexnet, AlexNet_Weights

from Numbersense.analysis.analyze_model import AnalyzeModel
from Numbersense.config import (
    Background,
    Dataloader_Parameters,
    Experiment_Parameters,
    ObjectiveFunction,
)
from Numbersense.figures.action_prediction_task_plots import plot_action_error
from Numbersense.figures.embedding_space_plots import (
    visualize_confusion_matrix,
    visualize_embedding_space_with_labels,
)
from Numbersense.model.networks import EmbeddingNet
from Numbersense.utilities.navigator import Navigator


class Plot:
    def __init__(
        self,
        dataloader_parameters: Dataloader_Parameters,
        experiment_parameters: Experiment_Parameters,
    ):
        self.dataloader_parameters = dataloader_parameters
        self.experiment_parameters = experiment_parameters

    def plot_embeddings(
        self,
        model_objective_function: ObjectiveFunction,
        background: Background,
        set_num: int,
        model_num: int,
        model_id: str,
        embedding_net: Optional[nn.Module] = None,
        concat: bool = False,
        data_objective_function: Optional[ObjectiveFunction] = None,
        only_embeddings: bool = False
    ):
        assert ObjectiveFunction.is_valid(
            model_objective_function
        ), "Unknown model objective function"
        assert Background.is_valid(background), "Unknown background"
        assert type(set_num) == int
        assert type(model_num) == int

        self.model_objective_function = model_objective_function
        self.data_objective_function = data_objective_function
        self.background = background
        self.set_num = set_num

        print("\nLoaded everyting!")

        model_path = Navigator.get_model_path(
            self.experiment_parameters.save_directory,
            self.model_objective_function.value,
            background.value,
            model_id,
            with_set=True,
            set=model_num,
        )
        self.model_path = model_path

        data_dir = Navigator.get_dataset_path(
            self.experiment_parameters.save_directory,
            self.data_objective_function.value
            if self.data_objective_function
            else self.model_objective_function.value,
            "test",
            background.value,
            with_set=True,
            set=set_num,
        )
        self.dataset_path = data_dir

        self.figure_output_folder = os.path.join(
            self.model_path,
            "validation_results",
            f"{self.data_objective_function.value if self.data_objective_function else self.model_objective_function.value}",
        )
        os.makedirs(self.figure_output_folder, exist_ok=True)

        print("\nAnalyzing...")

        a = AnalyzeModel(
            model_path=self.model_path,
            dataset_path=self.dataset_path,
            dataloader_parameters=self.dataloader_parameters,
            model_objective_function=self.model_objective_function.value,
            data_objective_function=self.data_objective_function.value
            if self.data_objective_function
            else None,
            embedding_net=embedding_net if embedding_net else EmbeddingNet(2, alexnet(weights=AlexNet_Weights.DEFAULT).features[0:5], concatenated=False),
            concat=concat,
            state="final.pt",
        )

        a.save_embeddings(concat=concat)
        if not only_embeddings:
            self.generate_main_figure(a_model=a)

    # TODO: Refactor
    def plot_action_accuracy(
        self,
        objective_function: str,
        background: str,
        set_num: int,
        model_num: int,
        pretrained: bool,
        concat: bool,
        set_num2: int = 0,
        model_num2: int = 0,
        background2: str = "",
        compute_err_ratio: bool = True,
        compare: bool = False,
        action_name_mapping: dict = None,
        state: str = "final.pt",
        model_ids: Union[list[str], None] = None,
    ):
        assert type(set_num) == int
        assert type(set_num2) == int
        assert type(model_num) == int
        assert type(model_num2) == int
        assert type(pretrained) == bool
        assert type(concat) == bool
        assert type(state) == str

        self.objective_function = objective_function
        self.background = background
        self.pretrained = pretrained
        self.concat = concat
        self.state_name = state.split(".")[0]

        print("\nLoaded everyting!")

        correct_per_action_list = []
        if compare:
            for model_num, set_num, background in zip(
                [set_num, set_num2], [model_num, model_num2], [background, background2]
            ):
                model_path = Navigator.get_model_path(
                    self.experiment_parameters.save_directory,
                    objective_function,
                    background,
                    with_set=True,
                    set=model_num,
                )
                models = Navigator.get_model_names(model_path, pretrained)
                if not models:
                    raise FileNotFoundError("No model available.")

                model = None
                try:
                    model = models.pop()
                except IndexError:
                    print(f"\nNo model with name: {model} was found.")
                    return

                # +1 because one model was removed previously with `.pop()`
                if len(models) + 1 > 1:
                    print(f"Multiple models detected. Using model: {model}")
                else:
                    print(f"Using model: {model}")

                self.model_name = model

                print("\nAnalyzing...")

                a = AnalyzeModel(
                    pretrained=pretrained,
                    experiment_parameters=self.experiment_parameters,
                    dataloader_parameters=self.dataloader_parameters,
                    objective_function=objective_function.value,
                    background=background.value,
                    concat=concat,
                    model_name=self.model_name,
                    model_num=model_num,
                    set_num=set_num,
                    state="final.pt",
                )
                correct_per_action_dict = a.save_predictions()
                correct_per_action_list.append(correct_per_action_dict)
        else:
            model_path = Navigator.get_model_path(
                self.experiment_parameters.save_directory,
                objective_function,
                background,
                with_set=True,
                set=model_num,
            )
            models = Navigator.get_model_names(model_path, pretrained)
            if not models:
                raise FileNotFoundError("No model available.")

            model = None
            try:
                model = models.pop()
            except IndexError:
                print(f"\nNo model with name: {model} was found.")
                return

            # +1 because one model was removed previously with `.pop()`
            if len(models) + 1 > 1:
                print(f"Multiple models detected. Using model: {model}")
            else:
                print(f"Using model: {model}")

            self.model_name = model

            print("\nAnalyzing...")

            a = AnalyzeModel(
                pretrained=pretrained,
                experiment_parameters=self.experiment_parameters,
                dataloader_parameters=self.dataloader_parameters,
                objective_function=objective_function,
                background=background,
                concat=concat,
                model_name=self.model_name,
                model_num=model_num,
                set_num=set_num,
                state="final.pt",
            )
            correct_per_action_list.append(a.save_predictions())

        colors = ["cornflowerblue", "orange"]
        fig, axes = plt.subplots(ncols=3, constrained_layout=True)
        fig.set_size_inches(12, 4)

        # Read dictionary
        # TODO: Get split point from params.txt file
        plot_action_error(
            correct_per_action_list,
            axes,
            split_point=3,
            colors=colors,
            action_mapping=action_name_mapping,
            compare=compare,
            compute_err_ratio=compute_err_ratio,
        )
        model_lines = []
        labels = model_ids if compare else ["1"]
        for i, label in enumerate(labels):
            model_lines.append(
                mlines.Line2D(
                    [],
                    [],
                    color=colors[i],
                    marker="o",
                    markersize=5,
                    label="Model " + label,
                )
            )
        axes[0].legend(handles=model_lines)
        axes[0].set_ylabel("Error" if compute_err_ratio else "Accuracy", fontsize=20)
        axes[0].set_xlabel("Number of Objects", fontsize=20)
        axes[1].set_yticks([])
        axes[2].set_yticks([])

        plt.show()

        print("\nSafed everything successfully!")

    def generate_main_figure(self, a_model: any):
        """Generates all embedding plots per dataset, i.e., the ground truth, the clustering and the confusion matrix.

        :param Any a_model: The model to generate the plots for.
        :param list datasets: A list of all the testsets that the model will process.
        :param str out_folder: The path to the output folder.
        """

        print("\nUsing test dataset:")
        print(self.dataset_path)

        os.makedirs(self.figure_output_folder, exist_ok=True)
        embeddings = a_model.load_analysis(
            self.figure_output_folder, target_file="embeddings.npy"
        )
        embedding_counts = a_model.load_analysis(
            self.figure_output_folder, target_file="embeddings_counts.npy"
        )
        _, labels = a_model.hdbscan_clustering(
            embeddings, min_cluster_size=70
        )  # cluster_indices_dict2 missing

        # embeddings.shape: (16200, 2)
        # embeddings_counts.shape: (16200, 2)

        # Ground Truth
        fig, ax = plt.subplots()  # nrows = 1, ncols = 1
        visualize_embedding_space_with_labels(
            embeddings,
            embedding_counts,
            ax,
            {
                "color_palette": "crest",
                "inset": False,
                # 'inset_axes': [0.05, 0.52, 0.4, 0.4],
                "annotate": "A",
            },
        )
        fig.savefig(os.path.join(self.figure_output_folder, "emb_ground_truth.png"))
        plt.close(fig)

        # Clustering
        fig, ax = plt.subplots()  # nrows = 1, ncols = 1
        visualize_embedding_space_with_labels(
            embeddings,
            labels,
            ax,
            {
                "color_palette": "bright",
                "inset": False,
                "label_every": 1,
                # 'inset_axes': [0.05, 0.52, 0.4, 0.4],
                # 'inset_limit': 6,
                "annotate": "B",
                "inset_labels": ["A0", "B0", "C0", "D0", "F0"],
            },
        )
        fig.savefig(os.path.join(self.figure_output_folder, "emb_unsupervised.png"))
        plt.close(fig)

        # Confusion Matrix
        fig, ax = plt.subplots(constrained_layout=True)  # nrows = 1, ncols = 1
        fig.set_size_inches(13, 4.2)
        conf_matrix, column_names, row_names = a_model.generate_confusion_matrix(
            labels, embedding_counts, remove_uncertain=True, sort=True
        )
        visualize_confusion_matrix(
            conf_matrix,
            ax,
            {
                "column_names": column_names,
                "row_names": row_names,
                "colbar_shrink": 0.8,
                "annotate": "C",
                "annotate_pos": (0.028, 0.9),
                "transpose": True,
                "cmap": "Reds",
            },
        )
        fig.savefig(os.path.join(self.figure_output_folder, "emb_conf_matrix.png"))
        plt.close(fig)
