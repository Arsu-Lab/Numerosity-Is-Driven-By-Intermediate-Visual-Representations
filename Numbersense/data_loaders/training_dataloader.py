"""Loads all the training data."""

import json
import os
from typing import Union

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image

from Numbersense.config import Dataloader_Parameters
from Numbersense.data_loaders.transform_loader import _load_target_transforms
from Numbersense.utilities.helpers import getenv


class TrainDataloader(torch.utils.data.Dataset):
    def __init__(
        self,
        dataloader_parameters: Dataloader_Parameters,
        data_dir: Union[str, None] = None,
        save_dir: Union[str, None] = None,
        classf: bool = False,
        save: bool = False,
        objective_function: str = "numerosity",
    ):
        assert data_dir is not None
        assert save_dir is not None

        self.save = save
        self.classf = classf
        self.batch_size = dataloader_parameters.train_batch_sizes
        self.max_objs = dataloader_parameters.train_max_objs
        self.num_batches = dataloader_parameters.train_num_batches
        self.action_size = dataloader_parameters.action_sizes

        # Convert a PIL Image or numpy.ndarray to tensor.
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.target_transform = {
            1: torch.tensor([[0, 0, 1]]),
            0: torch.tensor([[0, 1, 0]]),
            -1: torch.tensor([[1, 0, 0]]),
        }

        self.data_dir = data_dir
        self.save_dir = save_dir

        # Load all data:
        self.total_data = []
        for f_name in os.listdir(self.data_dir):
            if f_name[0] != "." and os.path.isfile(os.path.join(self.data_dir, f_name)):
                self.total_data.append(f_name)
        # self.total_data = sorted(self.total_data, key=lambda d: int(d.split("-")[0]))
        self.total_data = sorted(
            self.total_data, key=lambda d: (int(d.split("-")[0]), int(d.split("-")[1]))
        )

        # Load into pairs:
        self.total_data = [
            (img_0, img_1)
            for img_0, img_1 in zip(self.total_data[::2], self.total_data[1::2])
        ]

        self.total_labels = []
        self.total_actions, self.total_action_sizes = [], []

        # Load functions on demand from /targets
        _load_image_data, _calculate_image_data = _load_target_transforms(
            objective_function
        )
        if _load_image_data and _calculate_image_data:
            self.total_labels = _load_image_data(self.total_data)
            self.total_actions, self.total_action_sizes = _calculate_image_data(
                self.total_labels
            )

        self.total_data = np.array(
            self.total_data
        )  # List of all the paths to each image
        self.total_labels = np.array(
            self.total_labels
        )  # List of the number of objects in one image
        self.total_counts = (
            self.total_labels
        )  # List of the number of objects in one image
        self.total_actions = np.array(
            self.total_actions
        )  # List of which action took place
        self.total_action_sizes = np.array(
            self.total_action_sizes
        )  # List of how many objects have been added /
        # removed in one step
        # self.total_data.shape: (162000, 2)
        # self.total_labels.shape: (162000, 2)
        # self.total_counts.shape: (162000, 2)
        # self.total_actions.shape: (162000,)
        # self.total_action_sizes.shape: (162000,)

        # Used to save data per batch
        self.tmp_data = []
        self.tmp_labels = np.array([])
        self.tmp_counts = np.array([])
        self.tmp_actions = np.array([])
        self.tmp_action_sizes = np.array([])

        with open(os.path.join(self.save_dir, "params.txt"), "w") as f:
            json.dump(
                {
                    "num_batches": self.num_batches,
                    "batch_size": self.batch_size,
                    "max_objs": self.max_objs,
                    "action_size": self.action_size,
                },
                f,
                indent=1,
            )

    def load_data(self, epoch: Union[int, None] = None):
        """Loads data of size self.num_batches * self.batch_size per epoch.

        :param int epoch: The number of the current epoch.
        """
        assert epoch is not None

        runs = self.num_batches * self.batch_size
        start_idx = epoch * runs
        end_idx = (epoch + 1) * runs

        self.tmp_data = self.total_data[start_idx:end_idx]
        self.tmp_labels = self.total_labels[start_idx:end_idx]
        self.tmp_counts = self.total_counts[start_idx:end_idx]
        self.tmp_actions = self.total_actions[start_idx:end_idx]
        self.tmp_action_sizes = self.total_action_sizes[start_idx:end_idx]

    def __getitem__(self, index: int):
        if self.save:
            train_data_path = os.path.join(self.save_dir, "train_properties")
            os.makedirs(train_data_path, exist_ok=True)

            print("\nLoading pixel data...")
            pixel_intensities = []
            for d in self.tmp_data:
                pixel_intensities.append(
                    np.mean(Image.open(os.path.join(self.data_dir, d[0])))
                )
                pixel_intensities.append(
                    np.mean(Image.open(os.path.join(self.data_dir, d[1])))
                )

            np.save(
                os.path.join(train_data_path, "actions.npy"),
                np.int8(self.total_actions),
            )
            np.save(
                os.path.join(train_data_path, "action_sizes.npy"),
                np.int8(self.total_action_sizes),
            )
            np.save(
                os.path.join(train_data_path, "counts.npy"), np.int8(self.total_counts)
            )
            np.save(
                os.path.join(train_data_path, "intensities.npy"),
                np.array(pixel_intensities, dtype=np.float32)
                / 255,  # Changed from np.float
            )
            self.save = False

        # Return an image at the specified index of the dataset.
        try:
            if self.classf:
                target = int(self.tmp_labels[index][0])
            else:
                # Convert the action code in the dataset to a torch one-hot-encoded tensor
                target = self.target_transform[self.tmp_actions[index].item()]
            imgs = [
                Image.open(
                    os.path.join(self.data_dir, self.tmp_data[index][0])
                ).convert("L" if getenv("GRAYSCALE", 0) else "RGB"),
                Image.open(
                    os.path.join(self.data_dir, self.tmp_data[index][1])
                ).convert("L" if getenv("GRAYSCALE", 0) else "RGB"),
            ]

            # target: tensor([[0, 0, 1]])
            # imgs: ['0-0_img_0.png', '1-1_img_0.png']

        except IndexError as e:
            raise e

        # Convert PIL images
        img_ar = []
        for i in range(len(imgs)):
            img = imgs[i]

            # Transform image if transforms are passed
            if self.transforms is not None:
                img = self.transforms(img)

            if self.classf:
                # TODO: Update for new dataset structure!
                img_ar = img.transpose(1, 2)
                break
            else:
                img_ar.append(img.transpose(1, 2))

        return img_ar, target

    def __len__(self):
        return len(self.tmp_actions)
