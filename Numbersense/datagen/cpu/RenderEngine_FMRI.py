import os
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from numpy import random as rd
from PIL import Image, ImageDraw
from scipy.ndimage import convolve as conv2d


@dataclass
class Object:
    path: str
    x: int
    y: int
    scale_factor: float


class RenderEngine:
    # Hyperparameters of image generation:
    Theta = [-45, -15, 15, 45]
    Phi = [-165, -135, -105, -75, -45, -15, 15, 45, 75, 105, 135, 165]

    max_iters = 10000

    my_screen_dpi = 39

    width, height = 244, 244
    percent_max = 0.95

    NOT_RENDERED, WELL_RENDERED = 0, 1

    field_diameter = 216

    beta = 0
    gamma = 0
    kernel_size = 30

    def __init__(self, render_dir, width: int = 244, height: int = 244):
        """
        Initialize a FMRI_data_generator instance.

        Parameters
        ----------
        render_dir : str
            Path to the directory containing all renders
        width : int, optional
            Width of the image to be generated, by default 244
        height : int, optional
            Height of the image to be generated, by default 244
        """
        self.width = width
        self.height = height
        self.render_dir = render_dir
        # Cache backgrounds

        background_cache = {}
        for background in os.listdir(os.path.join(self.render_dir, "Backgrounds")):
            background = os.path.basename(background).split(".png")[0]
            background_cache[background] = self._load_background(background)
        self.background_cache = background_cache


    def _load_background(self, background_name):
        """
        Loads the background image

        Parameters
        ----------
        background_name : str
            Name of the background image

        Returns
        -------
        Image
            The loaded background image
        """
        width = self.width
        height = self.height
        background_dir = os.path.join(self.render_dir, "Backgrounds") + "/"
        field_view = Image.open(background_dir + background_name + ".png").resize(
            size=(width, height), resample=3
        )
        W_view, H_view = field_view.size
        assert (
            W_view >= width and H_view >= height
        ), "Given background input is too small, it should be at least 900 x 900 pixels"
        crp_field_view = field_view.crop(
            box=(
                int((W_view - width) / 2),
                int((H_view - height) / 2),
                int((W_view + width) / 2),
                int((H_view + width) / 2),
            )
        )
        return crp_field_view

    def _render_stimulus(
        self, info_centers, crp_field_view, save_path, only_display=False
    ):
        """
        Render the stimulus

        Parameters
        ----------
        info_centers : list
            List of dictionaries containing information about the objects to be rendered
        crp_field_view : Image
            The cropped background image
        save_path : str
            The path to save the rendered image
        only_display : bool, optional
            Whether to only display the image and not save it, by default False
        """
        field_mask = Image.new("L", (self.width, self.height), 0)
        draw = ImageDraw.Draw(field_mask)
        draw.ellipse(
            (
                (1 - self.percent_max) * self.width / 2,
                (1 - self.percent_max) * self.height / 2,
                (1 + self.percent_max) * self.width / 2,
                (1 + self.percent_max) * self.height / 2,
            ),
            fill=255,
        )
        field_mask = Image.fromarray(
            conv2d(
                field_mask,
                np.ones([self.kernel_size, self.kernel_size]) / self.kernel_size**2,
            )
        )

        seg_mask = Image.new("L", (self.width, self.height), 0)
        gray_background = Image.new("RGB", size=(self.width, self.height), color="gray")
        gray_background.paste(crp_field_view, mask=field_mask)
        obj_info_array, obj_idx = np.zeros([len(info_centers), 4]), 0
        for info_obj in info_centers:
            x, y = info_obj["position"]
            xi, yi, w, h = info_obj["mask"]
            obj_info_array[obj_idx] = (x + xi, y + yi, w, h)
            obj_idx += 1
            img = info_obj["model"]
            gray_background.paste(img, box=(x, y), mask=img)
            white_mask = Image.new("L", img.size, 1)
            seg_mask.paste(white_mask, box=(x, y), mask=img)

        gray_stimuli = np.mean(np.array(gray_background), axis=2).astype(int)

        if only_display:
            plt.figure(
                figsize=(
                    self.width / self.my_screen_dpi,
                    self.height / self.my_screen_dpi,
                ),
                tight_layout={"pad": 0},
            )
            plt.imshow(gray_stimuli, cmap="gray", vmin=0, vmax=255)
            plt.axis("off")
        else:
            gray_background.save(save_path)
            plt.imsave(
                fname=save_path,
                arr=gray_stimuli,
                dpi=self.my_screen_dpi,
                vmin=0,
                vmax=255,
                cmap="gray",
            )
            plt.close("all")
        return None

    def render_image(
        self,
        background_name: str,
        object_ids: List[Object],
        output_folder: str,
        output_name: str,
    ):
        """
        Render an image

        Parameters
        ----------
        background_name : str
            The name of the background image to use
        object_ids : list
            List of Object instances to be rendered
        output_folder : str
            The folder to save the rendered image
        output_name : str
            The name of the rendered image
        """
        crp_field_view = self.background_cache[background_name]

        object_views = []
        for object in object_ids:
            object_view = Image.open(object.path)
            width, height = object_view.size
            scaled_height = height * object.scale_factor
            scaled_width = width * object.scale_factor
            object_view = object_view.resize(
                size=(int(scaled_width), int(scaled_height)), resample=3
            )
            object_views.append(
                {
                    "position": (object.x, object.y),
                    "mask": (0, 0, scaled_width, scaled_height),
                    "model": object_view,
                }
            )

        self._render_stimulus(
            object_views,
            crp_field_view=crp_field_view,
            save_path=os.path.join(output_folder, output_name),
        )
