import json
import math
import os
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon


class RenderEngine:
    def __init__(
        self,
        image_dimensions: Tuple[int] = (244, 244),
        diameter_range: Tuple[int] = (15, 16),
        color_range: Tuple[int] = (255, 256),
        shape_options: List[str] = ["square"],
        seed: int = 0,
    ) -> None:
        self.image_dimensions = image_dimensions
        self.diameter_range = diameter_range
        self.color_range = color_range
        self.shape_options = shape_options
        self.seed = seed

        # The actual rendering engine:
        self._generateDataByCount = GenerateDataByCount(
            self.image_dimensions,
            diameter_range=self.diameter_range,
            color_range=self.color_range,
            shape_options=self.shape_options,
            seed=self.seed,
        )

    def create_stimulus_image(
        self,
        numerosity: int,
        output_name: str,
        save_path: str,
    ) -> bool:
        """
        Generate a white box stimulus image
        --------------------------------

        :param numerosity str: Number of black boxes in image
        :param output_name str: Name of the output image
        :param save_path str: Name of save directory

        :return bool: If not all objects could be inserted this function returns false
        """
        gd = self._generateDataByCount

        gd.init_image()
        added_shapes = set()
        for _ in range(numerosity):
            shape = gd.generate_object(self.diameter_range, self.color_range)
            succeeded = gd.add_shape_to_img(added_shapes=added_shapes, shape=shape)
            added_shapes.add(shape)
            if not succeeded:
                return True

        save_dir = os.path.join(save_path, output_name)
        gd.img.convert("RGB").save(save_dir + ".png")

        return False


class GenerateDataByCount:
    """
    This class handles generating a training dataset.

    Parameters:
    size - the size of the image
    diameter_range - the range of diameters a shape can have
    color_range - the range of pixel intensities to fill the shape with
    shape_types - the types of shapes to generate for the dataset
    """

    # internal shape class
    class Shape:
        def __init__(self, size=None, vertices=None, holes=None, fill=255):
            self.size = size
            self.vertices = vertices
            if holes is None:
                holes = []
            self.holes = holes

            self.polygon_vertices = []
            self.polygon_holes = []
            self.bounds = list(map(lambda v: [min(v), max(v)], zip(*vertices)))

            self.polygon = None
            self.fill = fill

    def __init__(
        self, size, diameter_range=None, color_range=None, shape_options=None, seed=None
    ):
        self.size = size
        self.img = None
        self.img_draw = None
        self.seed = seed
        np.random.seed(seed)

        self.data_list = []
        self.label_list = []
        self.action_list = []
        self.count_list = []
        self.data = None
        self.labels = None
        self.actions = None
        self.counts = None

        self.diameter_range = diameter_range
        self.color_range = color_range

        self.background_color = 0
        self.shape_options = shape_options

        self.action_dict = {"shake": 0, "add+shake": 1, "remove+shake": -1}

    # creates a Shape object according to specified parameters
    def generate_object(self, diameter_range, color_range):
        # randomly sample shapes from a list of options
        shape_type = np.random.choice(self.shape_options)
        fill, vertices, holes, diameter = None, None, None, None

        if shape_type == "square":
            p = None
            radius = int(np.random.choice(range(*diameter_range)) / 2)
            vertices = []

            # check if vertices define a valid polygon
            while p is None or not p.is_valid:
                vertices, _ = generateSpecificPolygon(0, 0, radius, "square")
                p = Polygon(vertices).difference(Polygon(holes))

        else:
            raise Exception("Unknown shape")

        fill = np.random.choice(range(*color_range))

        shape = self.Shape(size=diameter, vertices=vertices, holes=holes, fill=fill)
        return shape

    # create the initial image
    def init_image(self):
        img_arr = np.zeros(shape=self.size)
        self.img = Image.fromarray(img_arr)
        self.img_draw = ImageDraw.Draw(self.img)

        return img_arr

    """
        add shape to image
        added_shapes - list of Polygon objects that are already in the images
        shape - the new Polygon object to add
    """

    def add_shape_to_img(self, added_shapes, shape):
        # shape of image
        img_arr = np.asarray(self.img)
        arr_shape = img_arr.shape

        count = 0
        # grab vertices from shape object
        vertices = shape.vertices
        holes = shape.holes
        mod_vert = []
        mod_holes = []

        patch_found = None
        while patch_found is None:
            count += 1  # count number of attempts to find an empty patch

            # pick a random spot on the image
            x1 = np.random.randint(0, arr_shape[0])
            x2 = np.random.randint(0, arr_shape[1])
            mod_vert = []
            mod_holes = []
            fail = False
            for v in vertices:
                # check that each vertex, translated by x1 and x2 is within the bounds of the arr
                if not 0 < v[0] + x1 < arr_shape[0] or not 0 < v[1] + x2 < arr_shape[1]:
                    fail = True
                    break

                # track the list of modified vertices
                mod_vert.append((v[0] + x1, v[1] + x2))

            # if it worked for the outer points, it will work for holes vertices as well
            if not fail:
                for h in holes:
                    mod_holes.append((h[0] + x1, h[1] + x2))

            # skip if we failed vertex translation
            if fail:
                continue

            # if we succeed with translation, check that the location doesn't cause overlaps with
            # the already added shapes. Returns a Polygon (shapely) if it is valid.
            patch_found = self.check_polygon_intersection(
                added_shapes, mod_vert, mod_holes
            )

            # repeat for a maximum of 3000 times, if still failed try again with a different shape
            # if this happens too much, object statistics will be messed up...
            if count > 3000:
                print("Tried 3000 positions, image too small?")
                return False
        else:
            # if the patch is found succesfully, update the polygon vertex positions and the
            # shapely Polygon object
            shape.polygon = patch_found
            shape.polygon_vertices = mod_vert
            shape.polygon_holes = mod_holes

            # draw the polygon with PIL img draw
            self.img_draw.polygon(shape.polygon_vertices, fill=shape.fill)

            # add holes, if polygon has holes
            if len(shape.polygon_holes) != 0:
                self.img_draw.polygon(shape.polygon_holes, fill=self.background_color)

        return True

    # use the shapely Polygon class to enforce a margin between polygons
    def check_polygon_intersection(self, added_shapes, vertices, holes):
        polygon = Polygon(vertices)
        if len(holes) != 0:
            polygon = Polygon(vertices).difference(Polygon(holes))

        for s in added_shapes:
            if not s.polygon.buffer(3).disjoint(polygon):
                return None

        # return the polygon if it is a valid polygon with a valid margin
        return polygon

    def remove_shape(self, shape):
        self.img_draw.polygon(shape.polygon_vertices, fill=self.background_color)

    """
        generate a balanced dataset according to the initial specified parameters
        n_shapes - maximum number of shapes
        n_operations - number of actions to perform to create the dataset
    """

    def generate_dataset(self, n_shapes, n_repeats, images_dir=None):
        assert images_dir is not None
        os.makedirs(images_dir, exist_ok=True)

        img_num = 0
        for i in range(n_shapes + 1):
            for _ in range(n_repeats):
                img_num += 1
                count = i
                self.init_image()
                added_shapes = set()
                for _ in range(count):
                    shape = self.generate_object(self.diameter_range, self.color_range)
                    self.add_shape_to_img(added_shapes=added_shapes, shape=shape)
                    added_shapes.add(shape)

                # img_arr = np.asarray(self.img)
                # img_arr = img_arr / 255
                path = images_dir + f"img{img_num}.png"
                self.img.convert("RGB").save(path, "png")

                self.data_list.append(path)
                self.label_list.append(count)
                self.count_list.append(count)
                self.action_list.append(10)

        self.data = np.stack(self.data_list, axis=0)
        self.labels = np.stack(self.label_list, axis=0)
        self.actions = np.stack(self.action_list, axis=0)
        self.counts = np.stack(self.count_list, axis=0)

    def to_3channel(self):
        self.data = np.tile(self.data[:, :, None, :], reps=(1, 1, 3, 1))

    def save(self, tag="train", folder=""):
        path = "../data/raw/" + folder
        os.makedirs(path, exist_ok=True)
        data_path = os.path.join(path, ("data_" + tag + ".npy"))
        labels_path = os.path.join(path, ("labels_" + tag + ".npy"))
        actions_path = os.path.join(path, ("actions_" + tag + ".npy"))
        counts_path = os.path.join(path, ("counts_" + tag + ".npy"))
        np.save(data_path, self.data)
        np.save(labels_path, self.labels)
        np.save(actions_path, self.actions)
        np.save(counts_path, self.counts)

        keys = ["diameter_range", "color_range", "shape_options", "seed"]
        values = (self.diameter_range, self.color_range, self.shape_options, self.seed)
        pdict = dict(dict(zip(keys, values)))
        with open(path + "/params.txt", "w") as f:
            json.dump(pdict, f, indent=1)

    def visualize(self, folder=""):
        path = "../data/raw/" + folder
        path1 = path + "/dataset_visualization.png"

        jump_to = np.where(self.counts == max(self.counts))[0][0]
        seq_len = 9
        jump_to = min(jump_to, len(self.counts) - seq_len - 1)

        c = 1
        axs = None
        fig = None
        if seq_len is not None and c <= seq_len:
            fig, axs = plt.subplots(int(seq_len / 3), int(seq_len / 3))

        for i in range(jump_to, jump_to + seq_len):
            ax = axs.flatten()[c - 1]
            im = ax.imshow(Image.open(self.data[i]), cmap="gray")

            if c == 1:
                fig.colorbar(im, ax=ax)
            ax.set_title(str(self.counts[i]), size=18)
            c += 1

        plt.tight_layout(h_pad=0.5)
        plt.savefig(path1)
        plt.close()

        path2 = path + "/pixel_intensities_vs_count.png"
        pixel_intensities = []
        data_list = []
        for d in self.data:
            data_list.append(Image.open(d))
            pixel_intensities.append(np.mean(data_list[-1]))

        pixel_intensities = np.array(pixel_intensities)
        intensity_list = []
        counts_list = []
        for un in sorted(np.unique(self.counts)):
            mask = self.counts == un
            intensity_list.append(pixel_intensities[mask])
            counts_list.append(un)

        plt.figure()
        ax = plt.gca()
        ax.violinplot(intensity_list, counts_list, widths=0.3)
        plt.ylabel("total intensity")
        plt.xlabel("count")
        plt.savefig(path2)
        plt.close()


# ********* Helper functions **********


# generates a polygon with a random shape, not used
def generatePolygon(ctrX, ctrY, aveRadius, irregularity, spikeyness, numVerts):
    """
    https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon

    Start with the centre of the polygon at ctrX, ctrY,
    then creates the polygon by sampling points on a circle around the centre.
    Random noise is added by varying the angular spacing between sequential points,
    and by varying the radial distance of each point from the centre.

    Params:
    ctrX, ctrY - coordinates of the "centre" of the polygon
    aveRadius - in px, the average radius of this polygon, this roughly controls how large the polygon is, really only useful for order of magnitude.
    irregularity - [0,1] indicating how much variance there is in the angular spacing of vertices. [0,1] will map to [0, 2pi/numberOfVerts]
    spikeyness - [0,1] indicating how much variance there is in each vertex from the circle of radius aveRadius. [0,1] will map to [0, aveRadius]
    numVerts - self-explanatory

    Returns a list of vertices, in CCW order.
    """

    irregularity = clip(irregularity, 0, 1) * 2 * math.pi / numVerts
    spikeyness = clip(spikeyness, 0, 1) * aveRadius

    # generate n angle steps
    angleSteps = []
    lower = (2 * math.pi / numVerts) - irregularity
    upper = (2 * math.pi / numVerts) + irregularity
    sum = 0
    for i in range(numVerts):
        tmp = random.uniform(lower, upper)
        angleSteps.append(tmp)
        sum = sum + tmp

    # normalize the steps so that point 0 and point n+1 are the same
    k = sum / (2 * math.pi)
    for i in range(numVerts):
        angleSteps[i] = angleSteps[i] / k

    # now generate the points
    points = []
    angle = random.uniform(0, 2 * math.pi)
    while len(points) != numVerts:
        r_i = clip(random.gauss(aveRadius, spikeyness), 0, 2 * aveRadius)
        x = ctrX + r_i * math.cos(angle)
        y = ctrY + r_i * math.sin(angle)
        if (int(x), int(y)) in points:
            continue
        points.append((int(x), int(y)))

        angle = angle + angleSteps[len(points) - 1]

    return points


# generates a polygon with specific shape
def generateSpecificPolygon(ctrX, ctrY, aveRadius, shape):
    """

    Adapting from:
    https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon

    Start with the centre of the polygon at ctrX, ctrY,
    then creates the polygon by sampling points on a circle around the centre.

    Params:
    ctrX, ctrY - coordinates of the "centre" of the polygon
    aveRadius - in px, the average radius of this polygon, this roughly controls how large the polygon is, really only useful for order of magnitude.
    shape - string code for shape (circle, disk, square, etc...)

    Returns two lists of vertices, in CCW order. The first describes the vertices on the outer boundary of the shape, the second the internal boundary
    (like a cutout, for example a circle (not a disk which has a filled center)).
    """

    def generate_angle_steps():
        # generate n angle steps
        angleSteps = []
        lower = 2 * math.pi / numVerts
        upper = 2 * math.pi / numVerts
        sum = 0
        for i in range(numVerts):
            tmp = random.uniform(lower, upper)
            angleSteps.append(tmp)
            sum = sum + tmp

        # normalize the steps so that point 0 and point n+1 are the same
        k = sum / (2 * math.pi)
        for i in range(numVerts):
            angleSteps[i] = angleSteps[i] / k
        return angleSteps

    # all shapes
    points = None
    holes = None

    if shape == "circle":
        numVerts = 30
        angleSteps = generate_angle_steps()

        # now generate the points vertices
        points = []
        angle = random.uniform(0, 2 * math.pi)
        while len(points) != numVerts:
            r_i = aveRadius
            x = ctrX + r_i * math.cos(angle)
            y = ctrY + r_i * math.sin(angle)
            if (int(x), int(y)) in points:
                continue
            points.append((int(x), int(y)))

            angle = angle + angleSteps[len(points) - 1]

        # now generate the holes vertices
        holes = []
        angle = random.uniform(0, 2 * math.pi)
        while len(holes) != numVerts:
            r_i = aveRadius - 4  # arbitrary value for thickness
            x = ctrX + r_i * math.cos(angle)
            y = ctrY + r_i * math.sin(angle)
            if (int(x), int(y)) in holes:
                print(int(x), int(y))
                continue
            holes.append((int(x), int(y)))

            angle = angle + angleSteps[len(holes) - 1]

    if shape == "disk":
        numVerts = int(2 * aveRadius)
        angleSteps = generate_angle_steps()

        # now generate the points
        points = []
        angle = random.uniform(0, 2 * math.pi)
        c = 0
        while len(points) != numVerts:
            r_i = aveRadius
            x = ctrX + r_i * math.cos(angle)
            y = ctrY + r_i * math.sin(angle)
            if (int(x), int(y)) in points:
                c += 1
                if c > 300:
                    print(c)
                continue
            else:
                c = 0
            points.append((int(x), int(y)))

            angle = angle + angleSteps[len(points) - 1]

    # not used
    if shape == "right_crescent":
        numVerts = 30
        angleSteps = generate_angle_steps()

        # now generate the points vertices
        points = []
        angle = random.uniform(0, 2 * math.pi)
        while len(points) != numVerts:
            r_i = aveRadius
            x = ctrX + r_i * math.cos(angle)
            y = ctrY + r_i * math.sin(angle)
            if (int(x), int(y)) in points:
                continue
            points.append((int(x), int(y)))

            angle = angle + angleSteps[len(points) - 1]

        # now generate the holes vertices
        holes = []
        angle = random.uniform(0, 2 * math.pi)
        while len(holes) != numVerts:
            r_i = aveRadius - 4
            x = ctrX + 8 + r_i * math.cos(angle)
            y = ctrY + r_i * math.sin(angle)
            if (int(x), int(y)) in holes:
                continue
            holes.append((int(x), int(y)))

            angle = angle + angleSteps[len(holes) - 1]

    # not used
    if shape == "left_crescent":
        numVerts = 30
        angleSteps = generate_angle_steps()

        # now generate the points vertices
        points = []
        angle = random.uniform(0, 2 * math.pi)
        while len(points) != numVerts:
            r_i = aveRadius
            x = ctrX + r_i * math.cos(angle)
            y = ctrY + r_i * math.sin(angle)
            if (int(x), int(y)) in points:
                continue
            points.append((int(x), int(y)))

            angle = angle + angleSteps[len(points) - 1]

        # now generate the holes vertices
        holes = []
        angle = random.uniform(0, 2 * math.pi)
        while len(holes) != numVerts:
            r_i = aveRadius - 4
            x = ctrX - 8 + r_i * math.cos(angle)
            y = ctrY + r_i * math.sin(angle)
            if (int(x), int(y)) in holes:
                continue
            holes.append((int(x), int(y)))

            angle = angle + angleSteps[len(holes) - 1]

    if shape == "+":
        points = []
        if aveRadius % 2 == 0:
            aveRadius += 1

        SKF = 3
        points.append((int(ctrX - aveRadius), int(ctrY + aveRadius / SKF)))
        points.append((int(ctrX - aveRadius), int(ctrY - aveRadius / SKF)))

        points.append((int(ctrX - aveRadius / SKF), int(ctrY - aveRadius / SKF)))
        points.append((int(ctrX - aveRadius / SKF), int(ctrY - aveRadius)))

        points.append((int(ctrX + aveRadius / SKF), int(ctrY - aveRadius)))
        points.append((int(ctrX + aveRadius / SKF), int(ctrY - aveRadius / SKF)))

        points.append((int(ctrX + aveRadius), int(ctrY - aveRadius / SKF)))
        points.append((int(ctrX + aveRadius), int(ctrY + aveRadius / SKF)))

        points.append((int(ctrX + aveRadius / SKF), int(ctrY + aveRadius / SKF)))
        points.append((int(ctrX + aveRadius / SKF), int(ctrY + aveRadius)))

        points.append((int(ctrX - aveRadius / SKF), int(ctrY + aveRadius)))
        points.append((int(ctrX - aveRadius / SKF), int(ctrY + aveRadius / SKF)))

    if shape == "x":
        points = []
        if aveRadius % 2 == 0:
            aveRadius += 1

        SKF = 3
        points.append((int(ctrX - aveRadius), int(ctrY + aveRadius / SKF)))
        points.append((int(ctrX - aveRadius), int(ctrY - aveRadius / SKF)))

        points.append((int(ctrX - aveRadius / SKF), int(ctrY - aveRadius / SKF)))
        points.append((int(ctrX - aveRadius / SKF), int(ctrY - aveRadius)))

        points.append((int(ctrX + aveRadius / SKF), int(ctrY - aveRadius)))
        points.append((int(ctrX + aveRadius / SKF), int(ctrY - aveRadius / SKF)))

        points.append((int(ctrX + aveRadius), int(ctrY - aveRadius / SKF)))
        points.append((int(ctrX + aveRadius), int(ctrY + aveRadius / SKF)))

        points.append((int(ctrX + aveRadius / SKF), int(ctrY + aveRadius / SKF)))
        points.append((int(ctrX + aveRadius / SKF), int(ctrY + aveRadius)))

        points.append((int(ctrX - aveRadius / SKF), int(ctrY + aveRadius)))
        points.append((int(ctrX - aveRadius / SKF), int(ctrY + aveRadius / SKF)))

        p = np.array(points)
        theta = 45 / 360 * (2 * np.pi)
        rotation_mat = np.array(
            [[np.cos(theta), -1 * np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        rp = np.round(p @ rotation_mat)
        points = rp.astype(np.int).tolist()

    if shape == "square":
        points = []
        holes = []

        points.append((int(ctrX - aveRadius), int(ctrY - aveRadius)))
        points.append((int(ctrX - aveRadius), int(ctrY + aveRadius)))
        points.append((int(ctrX + aveRadius), int(ctrY + aveRadius)))
        points.append((int(ctrX + aveRadius), int(ctrY - aveRadius)))

        aveRadius -= 5
        holes.append((int(ctrX - aveRadius), int(ctrY - aveRadius)))
        holes.append((int(ctrX - aveRadius), int(ctrY + aveRadius)))
        holes.append((int(ctrX + aveRadius), int(ctrY + aveRadius)))
        holes.append((int(ctrX + aveRadius), int(ctrY - aveRadius)))

    # not used
    if shape == "diamond":
        points = []
        holes = []

        points.append((int(ctrX - aveRadius), int(ctrY - aveRadius)))
        points.append((int(ctrX - aveRadius), int(ctrY + aveRadius)))
        points.append((int(ctrX + aveRadius), int(ctrY + aveRadius)))
        points.append((int(ctrX + aveRadius), int(ctrY - aveRadius)))

        aveRadius -= 5
        holes.append((int(ctrX - aveRadius), int(ctrY - aveRadius)))
        holes.append((int(ctrX - aveRadius), int(ctrY + aveRadius)))
        holes.append((int(ctrX + aveRadius), int(ctrY + aveRadius)))
        holes.append((int(ctrX + aveRadius), int(ctrY - aveRadius)))

        p = np.array(points)
        theta = 45 / 360 * (2 * np.pi)
        rotation_mat = np.array(
            [[np.cos(theta), -1 * np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        rp = np.round(p @ rotation_mat)
        points = rp.astype(np.int).tolist()

        h = np.array(holes)
        theta = 45 / 360 * (2 * np.pi)
        rotation_mat = np.array(
            [[np.cos(theta), -1 * np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        rh = np.round(h @ rotation_mat)
        holes = rh.astype(np.int).tolist()

    # not used
    if shape == "hexagon":
        points = []
        angle = 0
        while len(points) != 6:
            r_i = aveRadius
            x = ctrX + r_i * math.cos(angle)
            y = ctrY + r_i * math.sin(angle)
            if (int(x), int(y)) in points:
                print("here")
                continue
            points.append((int(x), int(y)))

            angle = angle + 60 / 360 * math.pi * 2

    # not used
    if shape == "square-disk":
        # square with a disk cutout
        points = []

        points.append((int(ctrX - aveRadius), int(ctrY - aveRadius)))
        points.append((int(ctrX - aveRadius), int(ctrY + aveRadius)))
        points.append((int(ctrX + aveRadius), int(ctrY + aveRadius)))
        points.append((int(ctrX + aveRadius), int(ctrY - aveRadius)))

        r_i = aveRadius - 5
        numVerts = 30
        angleSteps = generate_angle_steps()

        # now generate the holes vertices
        holes = []
        angle = random.uniform(0, 2 * math.pi)
        while len(holes) != numVerts:
            x = ctrX + r_i * math.cos(angle)
            y = ctrY + r_i * math.sin(angle)
            if (int(x), int(y)) in holes:
                print(int(x), int(y))
                continue
            holes.append((int(x), int(y)))

            angle = angle + angleSteps[len(holes) - 1]

    return points, holes


# force x to be within min and max bounds
def clip(x, min, max):
    if min > max:
        return x
    elif x < min:
        return min
    elif x > max:
        return max
    else:
        return x
