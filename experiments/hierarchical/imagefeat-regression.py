import os, re
from typing import List
from multiprocessing import Pool
from enum import Enum, auto
import argparse
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from Numbersense.analysis.analyze_data import image_weibull_parameters, image_aggregate_fourier_magnitude, image_spatial_frequency

class FeatureType(Enum):
    WEIBULL = auto()
    SPATIAL_FREQ_MAGNITUDE = auto()
    SPATIAL_FREQ_POWER = auto()
    ALL = auto()

def extract_features(image_path: str, type: FeatureType):
    image = np.array(Image.open(image_path))
    if type == FeatureType.WEIBULL:
        features = image_weibull_parameters(image)
    elif type == FeatureType.SPATIAL_FREQ_MAGNITUDE:
        features = image_aggregate_fourier_magnitude(image)
    elif type == FeatureType.SPATIAL_FREQ_POWER:
        features = image_spatial_frequency(image)
    elif type == FeatureType.ALL:
        features = np.concatenate([image_weibull_parameters(image), image_aggregate_fourier_magnitude(image), image_spatial_frequency(image)])
    return features, int(os.path.basename(image_path).split("-")[1])

def low_level_image_property_regression(images: List[str], type: FeatureType):
    X, Y = [], []
    with Pool() as pool:
       features = pool.starmap(extract_features, [(image, type) for image in images])
    X, Y = list(zip(*features))[0], list(zip(*features))[1]
    model = LinearRegression()
    print("Running cross-validation...")
    return cross_val_score(model, X, Y, cv=10, scoring='r2').mean()

if __name__ == "__main__":
    # e.g: python3 experiments/hierarchical/imagefeat-regression.py weibull ~/research/hierarchical/datasets/varying_size_fixed_image_fixed_between/test/real/set_0/
    parser = argparse.ArgumentParser(description='Run low-level image property regression.')
    parser.add_argument('feature_type', type=str, help='Feature type for regression (WEIBULL, SPATIAL_FREQ_MAGNITUDE, SPATIAL_FREQ_POWER)')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset')
    args = parser.parse_args()
    feature_type = FeatureType[args.feature_type.upper()]
    dataset_path = args.dataset_path

    images = sorted([os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.png')], key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))
    print(f"Got {len(images)} images")
    score = low_level_image_property_regression(images, feature_type)
    print(f"R2 score for {feature_type} is {score:.2f}")
