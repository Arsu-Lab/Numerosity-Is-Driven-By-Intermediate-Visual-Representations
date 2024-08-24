import os
import re
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from torch import nn

from Numbersense.analysis.analyze_model import AnalyzeModel
from Numbersense.config import (
    Background,
    Dataloader_Parameters,
    Experiment_Parameters,
    ObjectiveFunction,
)
from Numbersense.utilities.navigator import Navigator


class Analysis:
    def __init__(
        self,
        dataloader_parameters: Dataloader_Parameters,
        experiment_parameters: Experiment_Parameters,
    ) -> None:
        self.dataloader_parameters = dataloader_parameters
        self.experiment_parameters = experiment_parameters

    def validate_accuracy(
        self,
        set_num: int,
        model_num: int,
        background: Background,
        model_objective_function: ObjectiveFunction,
        model_id: str,
        embedding_net: nn.Module,
        concat: bool = False,
        data_objective_function: Optional[ObjectiveFunction] = None,
    ):
        assert type(set_num) == int
        assert type(model_id) == str
        assert Background.is_valid(background), "Unknown background mode"
        assert ObjectiveFunction.is_valid(
            model_objective_function
        ), "Unknown model objective function"

        assert type(concat) == bool

        model_path = Navigator.get_model_path(
            self.experiment_parameters.save_directory,
            model_objective_function.value,
            background.value,
            model_id,
            with_set=True,
            set=model_num,
        )

        data_dir = Navigator.get_dataset_path(
            self.experiment_parameters.save_directory,
            data_objective_function.value
            if data_objective_function
            else model_objective_function.value,
            "test",
            background.value,
            with_set=True,
            set=set_num,
        )

        # Start with the testing procedure!
        a = AnalyzeModel(
            dataloader_parameters=self.dataloader_parameters,
            model_path=model_path,
            dataset_path=data_dir,
            model_objective_function=model_objective_function.value,
            data_objective_function=data_objective_function.value
            if data_objective_function
            else None,
            concat=concat,
            embedding_net=embedding_net,
            state="final.pt",
        )
        return a.save_predictions()

    def compile_experiment_metrics(
        self,
        metric: str,
        objective_functions: list[ObjectiveFunction],
        background: Background,
        model_ids: list[str],
        plot_name: str,
        action_specific: bool,
        fix_lims: bool,
        lims: Tuple[int, int],
        cross_val_objective_function: Optional[ObjectiveFunction] = None,
        classifier: str = "linear",
    ):
        assert all(
            [ObjectiveFunction.is_valid(of) for of in objective_functions]
        ), "Unknown objective function"
        assert Background.is_valid(background), "Unknown background"
        assert len(model_ids) > 0, "Empty list of model names"
        assert metric in ["accuracy", "sharpeness", "explained_variance"], ""
        assert not (
            metric != "accuracy" and action_specific
        ), "Invalid combination of metric and action_specific"
        assert classifier in [
            "linear",
            "logistic",
        ], "Classifier must be either 'linear' or 'logistic'"

        # Creating the plot
        plt.figure(figsize=(10, 6))
        y_min = lims[0]
        y_max = lims[1]
        for objective_function in objective_functions:
            experiment_path = os.path.join(
                self.experiment_parameters.save_directory,
                "trained_models",
                objective_function.value,
                background.value,
            )
            accuracy_metrics = []
            models = os.listdir(experiment_path)
            model_metrics = {}
            for model in models:
                if model in model_ids:
                    if cross_val_objective_function:
                        model_path = os.path.join(
                            self.experiment_parameters.save_directory,
                            "trained_models",
                            cross_val_objective_function.value,
                            background.value,
                            model,
                        )
                    else:
                        model_path = os.path.join(experiment_path, model)
                    if action_specific:
                        if metric == "accuracy":
                            model_metrics[model] = AnalyzeModel.get_accuracy(
                                model_path,
                                per_action=True,
                                cross_val_label=objective_function.value,
                            )
                    else:
                        if metric == "accuracy":
                            model_metrics[model] = AnalyzeModel.get_accuracy(
                                model_path, cross_val_label=objective_function.value
                            )
                        else:
                            model_metrics[model] = AnalyzeModel.get_embedding_score(
                                model_path,
                                classifier=classifier,
                                return_pca_variance=(metric == "explained_variance"),
                                cross_val_label=objective_function.value,
                            )

            if len(model_ids) != len(model_metrics):
                missing_models = set(model_ids) - set(models)
                print(
                    f"Warning: Missing models {missing_models}. 'x' and 'y' may not have the same size."
                )

            # Extracting the numbers from the model names and sorting according to x
            x = sorted(
                [
                    int(re.search(r"-(\d+)-", model).group(1))
                    for model in model_ids
                    if re.search(r"-(\d+)-", model)
                ]
            )
            accuracy_metrics = [
                model_metrics[model]
                for model in sorted(
                    model_ids,
                    key=lambda model: int("".join(filter(str.isdigit, model))),
                )
            ]
            if action_specific:
                y_mean_minus = [metric[0][0] for metric in accuracy_metrics]
                y_std_dev_minus = [metric[0][1] for metric in accuracy_metrics]
                y_mean_greater = [metric[1][0] for metric in accuracy_metrics]
                y_std_dev_greater = [metric[1][1] for metric in accuracy_metrics]
                y_mean_less = [metric[2][0] for metric in accuracy_metrics]
                y_std_dev_less = [metric[2][1] for metric in accuracy_metrics]

                plt.plot(x, y_mean_minus, "-o", label=f"{objective_function.value} -")
                plt.fill_between(
                    x,
                    [mean - dev for mean, dev in zip(y_mean_minus, y_std_dev_minus)],
                    [mean + dev for mean, dev in zip(y_mean_minus, y_std_dev_minus)],
                    alpha=0.5,
                )
                plt.plot(x, y_mean_greater, "-o", label=f"{objective_function.value} >")
                plt.fill_between(
                    x,
                    [
                        mean - dev
                        for mean, dev in zip(y_mean_greater, y_std_dev_greater)
                    ],
                    [
                        mean + dev
                        for mean, dev in zip(y_mean_greater, y_std_dev_greater)
                    ],
                    alpha=0.5,
                )
                plt.plot(x, y_mean_less, "-o", label=f"{objective_function.value} <")
                plt.fill_between(
                    x,
                    [mean - dev for mean, dev in zip(y_mean_less, y_std_dev_less)],
                    [mean + dev for mean, dev in zip(y_mean_less, y_std_dev_less)],
                    alpha=0.5,
                )
            else:
                y_mean = [metric[0] for metric in accuracy_metrics]
                y_std_dev = [metric[1] for metric in accuracy_metrics]

                plt.plot(x, y_mean, "-o", label=objective_function.value)
                plt.fill_between(
                    x,
                    [mean - dev for mean, dev in zip(y_mean, y_std_dev)],
                    [mean + dev for mean, dev in zip(y_mean, y_std_dev)],
                    alpha=0.5,
                )

            if fix_lims:
                if action_specific:
                    y_min = min(
                        [
                            min(
                                mean - dev
                                for mean, dev in zip(y_mean_minus, y_std_dev_minus)
                            ),
                            min(
                                mean - dev
                                for mean, dev in zip(y_mean_greater, y_std_dev_greater)
                            ),
                            min(
                                mean - dev
                                for mean, dev in zip(y_mean_less, y_std_dev_less)
                            ),
                            y_min,
                        ]
                    )
                    y_max = max(
                        [
                            max(
                                mean + dev
                                for mean, dev in zip(y_mean_minus, y_std_dev_minus)
                            ),
                            max(
                                mean + dev
                                for mean, dev in zip(y_mean_greater, y_std_dev_greater)
                            ),
                            max(
                                mean + dev
                                for mean, dev in zip(y_mean_less, y_std_dev_less)
                            ),
                            y_max,
                        ]
                    )
                else:
                    y_min = min(
                        y_min, min(mean - dev for mean, dev in zip(y_mean, y_std_dev))
                    )
                    y_max = max(
                        y_max, max(mean + dev for mean, dev in zip(y_mean, y_std_dev))
                    )
            lims = (min(lims[0], y_min), max(lims[1], y_max))
            plt.ylim(lims)
        plt.title(plot_name, fontsize=20)
        plt.xlabel("Number of Layers / Blocks", fontsize=16)
        plt.ylabel("Accuracy (%)", fontsize=16)
        plt.xticks(x, fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend()

        plot_name = plot_name.replace(" ", "_")
        if metric == "accuracy":
            path = (
                f"{self.experiment_parameters.save_directory}/accuracy/{plot_name}.png"
            )
        elif metric == "explained_variance":
            path = f"{self.experiment_parameters.save_directory}/explained_variance/{plot_name}.png"
        else:
            path = f"{self.experiment_parameters.save_directory}/sharpness/{plot_name}_{classifier}.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close("all")
        return path, lims
