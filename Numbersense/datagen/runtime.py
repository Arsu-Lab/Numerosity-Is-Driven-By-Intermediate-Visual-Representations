import json, os, multiprocessing
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from Numbersense.config import Dataloader_Parameters, Experiment_Parameters
from Numbersense.datagen.cpu.RenderEngine_FMRI import RenderEngine
from Numbersense.datagen.descriptors import (
    _list_animals,
    _list_backgrounds,
    _list_tools,
)
from Numbersense.utilities.helpers import getenv

@dataclass
class Object:
    path: str
    x: int
    y: int
    scale_factor: float


class Runtime:
    Theta = [-45, -15, 15, 45]
    Phi = [-165, -135, -105, -75, -45, -15, 15, 45, 75, 105, 135, 165]

    def __init__(
        self,
        describe_pair: callable,
        save_path: str,
        dataloader_parameters: Dataloader_Parameters,
        experiment_parameters: Experiment_Parameters,
        width=244,
        height=244,
        field_diameter=219,
        test=False,
    ):
        self.describe_pair = describe_pair
        self.width = width
        self.height = height
        self.field_diameter = field_diameter
        self.pairs = (
            dataloader_parameters.test_pair_num
            if test
            else dataloader_parameters.train_pair_num
        )
        self.render_dir = experiment_parameters.render_directory
        self.save_path = save_path
        self.max_num = (
            dataloader_parameters.test_max_objs
            if test
            else dataloader_parameters.train_max_objs
        )
        self.backgrounds = _list_backgrounds(self.render_dir)
        self.animals = _list_animals(self.render_dir)
        self.tools = _list_tools(self.render_dir)

    def _generate_metadata(self, seed: int, cores: int) -> List:
        cores = cores if cores else os.cpu_count()
        print(f"{cores} cores -- Generating metadata... ")

        np.random.seed(seed)  # set the seed for the array generation
        seeds = np.random.randint(0, 1000000000, self.pairs)
        metadata = []

        seed_chunks = np.array_split(seeds, cores)

        pool = multiprocessing.Pool()
        metadata_list = list(
            pool.map(
                self._cpu_core_generate_metadata,
                [(PID, seed_chunks[PID]) for PID in range(cores)],
            )
        )
        pool.close()
        pool.join()

        metadata = [item for sublist in metadata_list for item in sublist]
        return metadata

    def _cpu_core_generate_metadata(self, args: any):
        PID, seeds = args
        metadata = []
        chunck_size = len(seeds) * 2
        enumerator = enumerate(tqdm(seeds)) if PID == 0 else enumerate(seeds)
        for iteration, seed in enumerator:
            first, second = self.describe_pair(
                seed,
                self.max_num,
                self.backgrounds,
                self.animals,
                self.tools,
            )

            fails = 0
            # Insert objects into virtual perceptive field
            while True:
                if fails > 10000:
                    print("Failed realizing image")
                    break
                try:
                    id1 = PID * chunck_size + iteration * 2
                    obj1 = {
                        "id": id1,
                        "objects": self._realize(first),
                        "background": first["background"],
                        "name": str(id1) + first["name_suffix"],
                    }
                    id2 = PID * chunck_size + iteration * 2 + 1
                    obj2 = {
                        "id": id2,
                        "objects": self._realize(second),
                        "background": second["background"],
                        "name": str(id2) + second["name_suffix"],
                    }
                    break
                except Exception as e:
                    fails += 1
                    first, second = self.describe_pair(
                        seeds[iteration] + fails,
                        self.max_num,
                        self.backgrounds,
                        self.animals,
                        self.tools,
                    )
                    continue

            metadata.extend([obj1, obj2])
        return metadata

    def _render_metadata_cpu(self, metadata: List, cores: int):
        cores = cores if cores else os.cpu_count()
        print(f"{cores} cores -- Rendering metadata... ")

        if len(metadata) % cores == 0:
            per_core_size = int(len(metadata) / cores)
        else:
            print("Workload does not divide evenly across cores")
            return

        # Spawn multiple processes to execute the generation in parallel
        with multiprocessing.Pool(processes=cores) as pool:
            pool.starmap(
                self._cpu_core_generate_images,
                [
                    (
                        pid,
                        metadata[pid * per_core_size : (pid + 1) * per_core_size],
                    )
                    for pid in range(cores)
                ],
            )
        print("Generation finished")

    def _cpu_core_generate_images(self, PID: int, metadata: List):
        engine = RenderEngine(render_dir=self.render_dir)
        if PID == 0:
            for image in tqdm(metadata):
                engine.render_image(
                    image["background"].replace(".png", ""),
                    image["objects"],
                    self.save_path,
                    image["name"],
                )
        else:
            for image in metadata:
                engine.render_image(
                    image["background"].replace(".png", ""),
                    image["objects"],
                    self.save_path,
                    image["name"],
                )
        print(f"\nCore {PID} finished!")

    def __call__(
        self,
        seed: int,
        cores: Optional[int] = None,
    ):
        metadata = self._generate_metadata(seed, cores)
        self.create_statistics(metadata) if getenv("STATISTICS", 1) else None
        self._render_metadata_cpu(metadata, cores)

    def create_statistics(self, metadata: dict):
        obj_counts, action_counts, avg_sizes, avg_sizes_deviation = {}, {}, {}, {}
        print("Creating statistics...")
        for i, image in enumerate(metadata):
            objs = image["objects"]
            obj_counts[len(objs)] = (
                obj_counts.get(len(objs), 0) + 1
            )  # per image object count

            if i % 2 != 0:
                action = len(prev["objects"]) - len(objs)
                action_counts[action] = (
                    action_counts.get(action, 0) + 1
                )  # per pair action count

            if len(objs) > 0:
                curr_avg = sum([obj.scale_factor * 800 for obj in objs]) / len(objs)
                if len(objs) not in avg_sizes:
                    avg_sizes[len(objs)] = curr_avg
                else:
                    avg_sizes[len(objs)] = (
                        avg_sizes[len(objs)] + curr_avg  # per image avg object size
                    ) / 2
                if len(objs) not in avg_sizes_deviation:
                    avg_sizes_deviation[len(objs)] = self.OnlineStats()
                avg_sizes_deviation[len(objs)].update(curr_avg)
            prev = image

        with open(os.path.join(self.save_path, ".obj_counts.json"), "w") as f:
            json.dump(obj_counts, f)
        with open(os.path.join(self.save_path, ".action_counts.json"), "w") as f:
            json.dump(action_counts, f)
        with open(os.path.join(self.save_path, ".avg_sizes.json"), "w") as f:
            for key in avg_sizes:
                avg_sizes[key] = (avg_sizes[key], avg_sizes_deviation[key].std_dev())
            json.dump(avg_sizes, f)

    # ********** Helpers ***********

    class OnlineStats:  # implements welfords method
        def __init__(self):
            self.n = 0
            self.mean = 0
            self.M2 = 0

        def update(self, x):
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.M2 += delta * delta2

        def variance(self):
            if self.n < 2:
                return float("nan")
            return self.M2 / (self.n - 1)

        def std_dev(self):
            return self.variance() ** 0.5

    def _realize(
        self, image_desc: dict, max_insertion_attempts: int = 1000
    ) -> List[Object]:
        sampling_radius = self.field_diameter / 2
        num = len(image_desc["objs"])
        Theta_sel = np.random.choice(self.Theta, len(image_desc["objs"]))
        Phi_sel = np.random.choice(self.Phi, len(image_desc["objs"]))

        # Get object shapes
        paths = []
        obj_areas: Tuple[int, int, int, int] = []
        for id, object in enumerate(image_desc["objs"]):
            kind = object["object"]
            path = os.path.join(
                self.render_dir,
                "Objects",
                "Animals" if kind in self.animals else "Tools",
                kind,
                f"{kind}_theta{Theta_sel[id]}_phi{Phi_sel[id]}_{image_desc['background']}",
            )
            paths.append(path)
            img = Image.open(path)
            obj_areas.append(self.object_area(img))
            del img

        # Determine position for objects
        info_centers = []
        for id, object in enumerate(image_desc["objs"]):
            insertion_attempts = 0
            factor = object["size"] / 800  # Scaling factor
            obj = obj_areas[id]
            scaled_width = factor * obj[2]
            scaled_height = factor * obj[3]
            xi, yi, w_obj, h_obj = (
                obj[0] * factor,
                obj[1] * factor,
                scaled_width,
                scaled_height,
            )

            while insertion_attempts < max_insertion_attempts:
                x, y = self._random_position_within_perceptive_field(
                    id, num, sampling_radius
                )
                if self._is_inside_field_area(
                    x, y, xi, yi, w_obj, h_obj, self.width, self.height, sampling_radius
                ):
                    is_overlapping = False
                    for prev_obj in info_centers:
                        x0, y0, xt, yt, w, h = prev_obj
                        if (x + xi >= x0 + xt + w or x + xi + w_obj <= x0 + xt) or (
                            y + yi >= y0 + yt + h or y + yi + h_obj <= y0 + yt
                        ):
                            pass
                        else:
                            is_overlapping = True
                            break
                    if not is_overlapping:
                        new_info = (x, y, xi, yi, scaled_width, scaled_height)
                        info_centers.append(new_info)
                        break
                insertion_attempts += 1
            if insertion_attempts == max_insertion_attempts:
                raise Exception("Insertion Failed")

        return [
            Object(
                x=position[0],
                y=position[1],
                path=path,
                scale_factor=object["size"] / 800,
            )
            for object, path, position in zip(image_desc["objs"], paths, info_centers)
        ]

    def object_area(self, img: Image):
        xi, xf, yf, yi = 0, 0, 0, 0
        mask = np.array(img)[:, :, -1]  # alpha channel of RGBA of a PIL image
        height, width = mask.shape
        for i in range(width):
            if np.any(mask[:, i] != 0):
                xi = i
                break
        for i in range(1, width + 1):
            if np.any(mask[:, -i] != 0):
                xf = width - i
                break
        for j in range(height):
            if np.any(mask[j, :] != 0):
                yi = j
                break
        for j in range(1, height + 1):
            if np.any(mask[j, :] != 0):
                yf = height - j
                break
        Dx, Dy = xf - xi, yf - yi
        return (xi, yi, Dx, Dy)

    def _random_position_within_perceptive_field(
        self, inserted_objects: int, n_objects: int, sampling_radius: float
    ):
        beta = 0
        gamma = 0
        radius = sampling_radius * np.sqrt(np.random.uniform(beta, 1))
        angle = np.pi * np.random.uniform(
            low=(inserted_objects + gamma / 2) * 2 / n_objects,
            high=(inserted_objects + 1 - gamma / 2) * 2 / n_objects,
        )
        x, y = radius * np.cos(angle), radius * np.sin(angle)
        return int(x + self.width / 2), int(y + self.height / 2)

    def _is_inside_field_area(self, x, y, xi, yi, w_obj, h_obj, W, H, R):
        # Test in which quarter of the field (circle) the object was inserted and if it respect the field constraint
        is_inside = False
        if x + xi <= W / 2 and y + yi <= H / 2:
            top_left_is_inside = (
                True
                if (x + xi - W / 2) ** 2 + (y + yi - H / 2) ** 2 <= R**2
                else False
            )
            top_right_is_inside = (
                True
                if (x + xi + w_obj - W / 2) ** 2 + (y + yi - H / 2) ** 2 <= R**2
                else False
            )
            bot_left_is_inside = (
                True
                if (x + xi - W / 2) ** 2 + (y + yi + h_obj - H / 2) ** 2 <= R**2
                else False
            )
            bot_right_is_inside = (
                True
                if (x + xi + w_obj - W / 2) ** 2 + (y + yi + h_obj - H / 2) ** 2
                <= R**2
                else False
            )
            is_inside = (
                top_left_is_inside
                and top_right_is_inside
                and bot_right_is_inside
                and bot_left_is_inside
            )
        elif x + xi >= W / 2 and y + yi <= H / 2:
            top_right_is_inside = (
                True
                if (x + xi + w_obj - W / 2) ** 2 + (y + yi - H / 2) ** 2 <= R**2
                else False
            )
            bot_right_is_inside = (
                True
                if (x + xi + w_obj - W / 2) ** 2 + (y + yi + h_obj - H / 2) ** 2
                <= R**2
                else False
            )
            is_inside = top_right_is_inside and bot_right_is_inside
        elif x + xi <= W / 2 and y + yi >= H / 2:
            bot_left_is_inside = (
                True
                if (x + xi - W / 2) ** 2 + (y + yi + h_obj - H / 2) ** 2 <= R**2
                else False
            )
            bot_right_is_inside = (
                True
                if (x + xi + w_obj - W / 2) ** 2 + (y + yi + h_obj - H / 2) ** 2
                <= R**2
                else False
            )
            is_inside = bot_left_is_inside and bot_right_is_inside
        elif x + xi >= W / 2 and y + yi >= H / 2:
            is_inside = (
                True
                if (x + xi + w_obj - W / 2) ** 2 + (y + yi + h_obj - H / 2) ** 2
                <= R**2
                else False
            )
        return is_inside
