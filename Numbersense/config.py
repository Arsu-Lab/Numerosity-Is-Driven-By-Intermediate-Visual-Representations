import os
from enum import Enum
from typing import Optional


# Add new background mode here:
class Background(Enum):
    REAL = "real"
    PLAIN = "plain"

    @classmethod
    def is_valid(cls, value):
        """Check if the value is a valid enum value."""
        if not isinstance(value, Enum):
            return False
        return any(value.value == item.value for item in cls)


# Add new objective functions here:
class ObjectiveFunction(Enum):
    ###### 2. Hierarchial experiment with backbone configs
    VARYING_SIZE = "varying_size"
    VARYING_SIZE_4VAL = "varying_size_4val"
    VARYING_SIZE_FIXED_IMAGE_FIXED_BETWEEN = "varying_size_fixed_image_fixed_between"
    VARYING_SIZE_FIXED_IMAGE_FIXED_BETWEEN_4VAL = "varying_size_fixed_image_fixed_between_4val"
    VARYING_SIZE_AND_OBJECT = "varying_size_and_object"
    VARYING_SIZE_AND_OBJECT_4VAL = "varying_size_and_object_4val"
    VARYING_SIZE_AND_OBJECT_AND_BACKGROUND = "varying_size_and_object_and_background"
    VARYING_SIZE_AND_OBJECT_AND_BACKGROUND_4VAL = "varying_size_and_object_and_background_4val"
    ###### 1. Experiment with backbone configs
    NUMEROSITY = "numerosity"
    NUMEROSITY_3VAL = "numerosity_3val"
    NUMEROSITY_BACKGROUND_CHANGE = "numerosity_background_change"
    NUMEROSITY_BACKGROUND_CHANGE_3VAl = "numerosity_background_change_3val"
    NUMEROSITY_OBJECT_CHANGE = "numerosity_object_change"
    NUMEROSITY_OBJECT_CHANGE_3VAL = "numerosity_object_change_3val"
    NUMEROSITY_SIZE_DIFFERENCE = "numerosity_size_difference"
    NUMEROSITY_SIZE_DIFFERENCE_3VAL = "numerosity_size_difference_3val"
    FILL_OBJECTS_NUMEROSITY = "fill_objects_numerosity"
    FILL_OBJECTS_NUMEROSITY_3VAL = "fill_objects_numerosity_3val"
    ###### Experiment with loss functions
    OBJECT_DIFFERENTIATION = "object_differentiation"
    CATEGORY_DIFFERENTIATION = "category_differentiation"
    EVEN_ODD = "even_odd"
    FIXED_OBJECT = "fixed_object"
    TARGET_OBJECT = "target_object"
    TARGET_NUMBER = "target_number"

    @classmethod
    def is_valid(cls, value):
        """Check if the value is a valid enum value."""
        if not isinstance(value, Enum):
            return False
        return any(value.value == item.value for item in cls)


class Dataloader_Parameters:
    """
    A class used to represent the parameters of a dataloader.

    Attributes
    ----------
    train_num_batches : int
        Number of training batches
    test_num_batches : int
        Number of testing batches
    train_batch_sizes : int
        Size of training batches
    test_batch_sizes : int
        Size of testing batches
    train_max_objs : int
        Maximum number of objects in training
    test_max_objs : int
        Maximum number of objects in testing
    train_epochs : int
        Number of training epochs
    test_epochs : int
        Number of testing epochs
    train_to_test_ratio : Optional[float]
        Ratio of training to testing
    action_sizes : int
        Size of actions
    """

    def __init__(
        self,
        train_num_batches: int = 30,
        test_num_batches: int = 45,
        train_batch_sizes: int = 180,
        test_batch_sizes: int = 8,
        train_max_objs: int = 3,
        test_max_objs: int = 8,
        train_epochs: int = 30,
        test_epochs: int = 45,
        train_to_test_ratio: Optional[float] = None,
        action_sizes: int = 1,
    ):
        """
        Initialize the Dataloader_Parameters class.

        Parameters
        ----------
        train_num_batches : int, optional
            Number of training batches, by default 30
        test_num_batches : int, optional
            Number of testing batches, by default 45
        train_batch_sizes : int, optional
            Size of training batches, by default 180
        test_batch_sizes : int, optional
            Size of testing batches, by default 16
        train_max_objs : int, optional
            Maximum number of objects in training, by default 3
        test_max_objs : int, optional
            Maximum number of objects in testing, by default 8
        train_epochs : int, optional
            Number of training epochs, by default 30
        test_epochs : int, optional
            Number of testing epochs, by default 45
        train_to_test_ratio : Optional[float], optional
            Ratio of training to testing, by default None
        action_sizes : int, optional
            Size of actions, by default 1
        """
        self.train_num_batches = train_num_batches
        self.test_num_batches = test_num_batches

        self.train_batch_sizes = train_batch_sizes
        self.test_batch_sizes = test_batch_sizes

        self.train_max_objs = train_max_objs
        self.test_max_objs = test_max_objs

        self.train_epochs = train_epochs
        self.train_pair_num = (
            train_epochs * self.train_num_batches * self.train_batch_sizes
        )

        if train_to_test_ratio:
            assert (
                train_to_test_ratio <= 1
            ), "Test dataset should be smaller than train dataset"
        self.train_to_test_ratio = train_to_test_ratio

        self.test_pair_num = (
            int(train_to_test_ratio * self.train_pair_num)
            if train_to_test_ratio
            else test_epochs * self.test_num_batches * self.test_batch_sizes
        )

        self.action_sizes = action_sizes

    def save_parameters(self, save_path: str):
        """
        Save the parameters to a specified path.

        Parameters
        ----------
        save_path : str
            The path where the parameters will be saved
        """
        assert os.path.isdir(save_path), "Must be a valid path"
        os.makedirs(save_path, exist_ok=True)
        # TODO: Convert to json and save dir
        NotImplementedError("Save functionality not implemented yet")

    def read_parameters(self, save_path: str):
        NotImplementedError("Not implemented yet")


class Experiment_Parameters:
    """
    A class used to represent the parameters of an experiment

    ...

    Attributes
    ----------
    save_directory : str
        The directory where the experiment results will be saved
    render_directory : str
        The directory where the pre-rendered images are located.
        Requires specific structure. See README for more details.

    """

    save_directory = os.getcwd()

    def __init__(self, save_dir: str, render_dir: Optional[str] = None):
        """
        Initializes the experiment parameters with the given save and render directories.

        Parameters
        ----------
            save_dir : str
                The path to save experiment results.
            render_dir : Optional[str], default=None
                The path to the directory containing prerendered images, if any.
        """
        self.save_directory = save_dir
        self.render_directory = render_dir
        os.makedirs(self.save_directory, exist_ok=True)
        if self.render_directory: os.makedirs(self.render_directory, exist_ok=True)
