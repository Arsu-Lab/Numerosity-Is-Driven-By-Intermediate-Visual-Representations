import importlib
import os
from typing import Optional

from Numbersense.config import (
    Background,
    Dataloader_Parameters,
    Experiment_Parameters,
    ObjectiveFunction,
)
from Numbersense.datagen.runtime import Runtime
from Numbersense.utilities.navigator import Navigator


class Generator:
    """Generator
    ===
    Generates synthetic images for model training.

    Usage:
    ---------------
    Create an instance of a Generator class::

    >>> generator = Generator(experiment_params, dataloader_params)

    Create a dataset::

    >>> generator.generate_dataset(objective_function, background, "train", seed, PID)
    """

    def __init__(
        self,
        experiment_parameters: Experiment_Parameters,
        dataloader_parameters: Dataloader_Parameters,
    ):
        """
        Initializes an instance of the ``Generator`` class.
        ----------------------------

        :param `experiment_parameters`: Parameters to set the experiment save and render image gallery location
        :param `dataloader_parameters`: Parameters for the dataloaders of the model to train in the future

        :returns: `Generator`: A instance of the `Generator` class

        """
        self.experiment_parameters = experiment_parameters
        self.dataloader_parameters = dataloader_parameters

    def generate_dataset(
        self,
        objective_function: ObjectiveFunction,
        background: Background,
        validation: bool,
        seed: int,
        device: str = "cpu",
        cores: Optional[int] = None,
    ) -> None:
        """Generates synthetic images for model training.
        ------------------------------
        Only works on linux and macOS

        Parameters
        ----------
        objective_function : str
            Objective function to use for training
        background : str
            Background mode to use for training
        seed : int
            Seed for the random number generator
        device : str, optional
            Which accelerator to use. Either `cpu` or `gpu`. `gpu` only supports metal devices. Default: cpu
        cores : int, optional
            Number of cores to use for multiprocessing. Default: `os.cpu_cores()`
        batch_size : int, optional
            Batch size for gpu. Default: 400

        Returns
        -------
        None
        """
        assert ObjectiveFunction.is_valid(
            objective_function
        ), "Unknown objective function"
        assert Background.is_valid(background), "Unknown background mode"
        assert (
            self.dataloader_parameters.test_pair_num % cores == 0
            if validation
            else self.dataloader_parameters.train_pair_num % cores == 0
        ), "Cores do not divide evenly across workload"
        assert type(seed) == int, "Seed must be an integer"
        assert device in ["cpu", "gpu"], "Unknown accelerator"

        save_path = Navigator.get_dataset_path(
            self.experiment_parameters.save_directory,
            objective_function.value,
            "test" if validation else "train",
            background.value,
        )
        os.makedirs(save_path, exist_ok=True)
        set_dir = Navigator.get_next_set(save_path)
        self.set_dir = set_dir
        os.makedirs(set_dir)

        file_path = "Numbersense/datagen/descriptors.py"
        name = f"describe_pair_{objective_function.value}"
        try:
            module = importlib.import_module(file_path[:-3].replace("/", "."))
            function = getattr(module, name)
        except AttributeError:
            print(f"Function {name} not found in module {file_path}")
            function = None

        Runtime(
            function,
            set_dir,
            self.dataloader_parameters,
            self.experiment_parameters,
            test=validation,
        )(seed, cores)
