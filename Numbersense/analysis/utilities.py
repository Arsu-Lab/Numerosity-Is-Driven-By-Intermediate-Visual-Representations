import importlib
import torch
from torch import nn
from torchvision.models import AlexNet_Weights, alexnet

from Numbersense.model.networks import EmbeddingNet, SiameseActionClassificationNet


class ModelHelper:
    """Helper class which loads a model based on specified parameters.

    :param dict dataset_params: The specified training parameters.
    :param Any pretrained: Specifies if the model was pretrained or not.
    :param bool concat: Specifies whether or not to use concatenated images as model input.
    :param str state: The state of the model which is about to be loaded.
    :param str model_name: The name of the model to be loaded.
    """

    def __init__(
        self,
        model_path: str,
        embedding_net: nn.Module = None,
    ):
        self.model_path = model_path
        self.embedding_net = embedding_net

        self.load_trained_model()

    def load_trained_model(self):
        """Loads a model based on the specified parameters."""
        # By default the embedding has AlexNet pretrained first 5 layers as backbone
        if not self.embedding_net:
            backbone = alexnet(weights=AlexNet_Weights.DEFAULT).features[0:5]
            self.embedding_net = EmbeddingNet(2, backbone, concatenated=False)

        model = SiameseActionClassificationNet(
            self.embedding_net, unique_id=self.model_path  # Maybe change the unique id
        )

        model.load_state_dict(torch.load(self.model_path, map_location="cpu")["model"])
        self.model = model

def _load_transform_dict(objective_function: str) -> str:
    """
    Loads the objective function specific target transform
    --------------------------------

    :param objective_function str: The objective function of the target transform

    :returns `None`
    """
    try:
        module_path = ".".join(["Numbersense", "analysis", "target_transforms"])
        module = importlib.import_module(module_path)
        target_dict = getattr(module, f"target_transform_{objective_function}")
        return target_dict
    except ImportError as e:
        print(f"Failed to import module: {module_path}")
        print("ERROR:\n")
        print(e)
        return
