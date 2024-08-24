from typing import Optional

import torch
from torch import optim
from torchvision.models import AlexNet_Weights, alexnet

from Numbersense.config import (
    Background,
    Dataloader_Parameters,
    Experiment_Parameters,
    ObjectiveFunction,
)
from Numbersense.model.networks import EmbeddingNet, SiameseActionClassificationNet
from Numbersense.training.runner import Runner
from Numbersense.utilities.navigator import Navigator


class Training:
    def __init__(
        self,
        dataloader_parameters: Dataloader_Parameters,
        experiment_parameters: Experiment_Parameters,
    ):
        self.dataloader_parameters = dataloader_parameters
        self.experiment_parameters = experiment_parameters

    def train(
        self,
        objective_function: ObjectiveFunction,
        background: Background,
        set_num: int = 0,
        embedding_net: EmbeddingNet = None,
        model_id: str = "",
        model_num: Optional[int] = None,
    ):
        assert ObjectiveFunction.is_valid(
            objective_function
        ), "Unknown objective function."
        assert Background.is_valid(background), "Unknown background."

        dataloader_parameters = self.dataloader_parameters
        experiment_parameters = self.experiment_parameters
        epochs = dataloader_parameters.train_epochs

        # General prerequisits
        dataset_path = Navigator.get_dataset_path(
            experiment_parameters.save_directory,
            objective_function.value,
            "train",
            background.value,
            with_set=True,
            set=set_num,
        )

        # Configuring the model & training parameters
        lr = 0.0001
        if not embedding_net:
            # Default backbone is first 5 layers of pretrained AlexNet
            embedding_net = EmbeddingNet(
                2,
                alexnet(weights=AlexNet_Weights.DEFAULT).features[0:5],
                concatenated=False,
            )
        model = SiameseActionClassificationNet(embedding_net, unique_id=model_id)

        loss = torch.nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = None

        r = Runner(
            dataset_path=dataset_path,
            dataloader_parameters=dataloader_parameters,
            objective_function=objective_function.value,
            background_mode=background.value,
            experiment_parameters=experiment_parameters,
            model=model,
            model_num=model_num,
        )
        r.train_embedding_model_unsupervised(
            optimizer=optimizer,
            loss_fn=loss,
            scheduler=scheduler,
            num_epochs=epochs,
        )
