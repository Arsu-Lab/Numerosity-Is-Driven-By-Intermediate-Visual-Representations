"""Loads all the testing data."""

import os
from typing import Any, Union

import numpy as np
import torch
import torch.utils.data
from PIL import Image

from Numbersense.data_loaders.transform_loader import _load_target_transforms
from Numbersense.utilities.helpers import getenv


class ValidationDataloader(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: Union[str, None] = None,
        test_set_len: Union[int, None] = None,
        transform: Any = None,
        classf: bool = False,
        concat: bool = False,
        objective_function: str = "numerosity",
    ):
        assert data_dir is not None
        assert test_set_len is not None

        self.classf = classf
        self.concat = concat
        self.transform = transform
        self.targets_dict = {
            1: torch.tensor([0, 0, 1]),
            0: torch.tensor([0, 1, 0]),
            -1: torch.tensor([1, 0, 0]),
        }

        # Load all data:
        self.data_dir = data_dir
        self.total_data = []
        for f_name in os.listdir(self.data_dir):
            if f_name[0] != "." and os.path.isfile(os.path.join(self.data_dir, f_name)):
                self.total_data.append(f_name)
        self.total_data = sorted(
            self.total_data, key=lambda d: (int(d.split("-")[0]), int(d.split("-")[1]))
        )

        # Load data into pairs
        self.total_data = [
            (img_0, img_1)
            for img_0, img_1 in zip(self.total_data[::2], self.total_data[1::2])
        ]

        self.total_counts = []  # List of the number of objects in one image
        self.total_actions, self.total_action_sizes = [], []

        # Load functions on demand from /targets
        _load_image_data, _calculate_image_data = _load_target_transforms(
            objective_function
        )
        if _load_image_data and _calculate_image_data:
            self.total_counts = _load_image_data(self.total_data)
            self.total_actions, self.total_action_sizes = _calculate_image_data(
                self.total_counts
            )

        self.total_labels = []  # List of one-hot encoded target-actions
        for action in self.total_actions:
            self.total_labels.append(self.targets_dict[action])

        self.total_data = np.array(
            self.total_data
        )  # List of all the paths to each image
        self.total_actions = np.array(
            self.total_actions
        )  # List of which action took place
        self.total_action_sizes = np.array(
            self.total_action_sizes
        )  # List of how many objects have been added /
        # removed in one step
        print("\nLoaded all the testing data!")

    def __getitem__(self, index):
        # Get targets, counts, data at index
        if self.classf:
            target = int(self.total_counts[index][0])
        else:
            target = self.total_labels[index]
        counts = self.total_counts[index]

        imgs = [
            Image.open(os.path.join(self.data_dir, self.total_data[index][0])).convert(
                "L" if getenv("GRAYSCALE", 0) else "RGB"
            ),
            Image.open(os.path.join(self.data_dir, self.total_data[index][1])).convert(
                "L" if getenv("GRAYSCALE", 0) else "RGB"
            ),
        ]

        # target: tensor([0, 0, 1])
        # counts: [0 1]
        # imgs: ['0-0_img_0.png', '1-1_img_0.png']

        # Convert images to the desired format
        img_ar = []
        for i in range(len(imgs)):
            img = imgs[i]

            if self.transform is not None:
                img = self.transform(img)

            if self.classf:
                img_ar = img
                break
            elif self.concat:
                img_ar = self.transform(img_ar)
                img_ar = img_ar.transpose(1, 2)
                break
            else:
                img_ar.append(img.transpose(1, 2))

        return img_ar, target, counts

    def __len__(self):
        return len(self.total_data)
