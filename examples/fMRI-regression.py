#pip install pandas
import os
from pathlib import Path
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from Numbersense.utilities.helpers import getenv

class ROI(Enum):
    STREAMS = "streams"
    PRF = "prf-visualrois"
    FLOC_BODIES = "floc-bodies"
    FLOC_FACES = "floc-faces"
    FLOC_PLACES = "floc-places"
    FLOC_WORDS = "floc-words"

def get_fMRI_row_mapping(path:str, subject_id:str, row_lookup_table:pd.DataFrame):
    dataset = {}
    for subset in os.listdir(path):
        if subset.isdigit() and os.path.isdir(os.path.join(path, subset)):
            dataset[subset] = [lookup_fmri_row(image.split(".")[0], subject_id, row_lookup_table) for image in os.listdir(os.path.join(path, subset))]
    return dataset

def lookup_fmri_row(image_nsd_id:str, subject_id:str, df:pd.DataFrame):
    col_name = f"subj{subject_id.zfill(2)}_image_id"
    return int(df.loc[df['nsd_image_id'] == int(image_nsd_id), col_name].values[0])

def get_fmri(path:str, subject_id:str, hemisphere:str):
    return np.load(os.path.join(path, f"sub{subject_id.zfill(2)}", f"{'r' if hemisphere == 'right' else 'l'}h_training_fmri.npy"))

def get_roi_mask(path:str, subject_id:str, hemisphere:str, roi_type: ROI):
    return np.load(os.path.join(path, f"sub{subject_id.zfill(2)}", "roi_masks", f"{'r' if hemisphere == 'right' else 'l'}h.{roi_type.value}_challenge_space.npy"))

def get_roi_mapping(path:str, subject_id:str, roi_type: ROI) -> dict:
    return np.load(os.path.join(path, f"sub{subject_id.zfill(2)}", "roi_masks", f"mapping_{roi_type.value}.npy"), allow_pickle=True).item()

def subject_regression(subject_id:int, hemisphere: str = "right"):
    fMRI_row_mapping = get_fMRI_row_mapping(str(Path(__file__).parent.parents[0] / "fmri" / "images"), subject_id, nsd_to_fmri_row_lookup)
    fmri_scan = get_fmri(str(Path(__file__).parent.parents[0] / "fmri"), subject_id, hemisphere)
    roi_mask = get_roi_mask(str(Path(__file__).parent.parents[0] / "fmri"), subject_id, hemisphere, ROI.PRF)
    roi_mapping = get_roi_mapping(str(Path(__file__).parent.parents[0] / "fmri"), subject_id, ROI.PRF) 

    scores = {roi: 0 for roi in roi_mapping.values() if roi != "Unknown"}
    for i, roi in enumerate([k for k in roi_mapping.values() if k != "Unknown"]):
        mask = (roi_mask == i + 1)
        flattened_data = [(key, np.array(fmri_scan[int(value) - 1, :])[mask]) for key, values in fMRI_row_mapping.items() for value in values] # -1 ? 
        np.random.shuffle(flattened_data)
        X = np.array([data for _, data in flattened_data])
        Y = np.array([int(label) for label, _ in flattened_data])
        model = LinearRegression()
        if X.shape[1] == 0:
            print(f"No data for ROI {roi}")
            continue
        scores[roi] = cross_val_score(model, X, Y, cv=5, scoring='r2').mean()
        print(f"Subject {str(subject_id)} cross-validated R^2 value for ROI {roi}: {np.mean(scores[roi])}") if getenv('PRINT', 0) else None
    return scores

if __name__ == "__main__":
    np.random.seed(6)
    hemisphere = "right"
    nsd_to_fmri_row_lookup = pd.read_csv(str(Path(__file__).parent.parents[0] / "fmri" / "872_nsd_to_subject_image_mapping.csv"))
    roi_mapping = get_roi_mapping(str(Path(__file__).parent.parents[0] / "fmri"), "1", ROI.PRF) 
    subject_averaged_scores = {roi: [] for roi in roi_mapping.values() if roi != "Unknown"}

    for i in range(1, 9):
        scores: dict = subject_regression(str(i), hemisphere=hemisphere)
        for roi, score in scores.items():
            subject_averaged_scores[roi].append(score)

    for key, value in subject_averaged_scores.items():
        print(f"R^2 {np.mean(value):.3f} ### ROI: {key}")

    import matplotlib.pyplot as plt

    roi_names = list(subject_averaged_scores.keys())
    averaged_scores = [np.mean(scores) for scores in subject_averaged_scores.values()]

    plt.bar(roi_names, averaged_scores)
    plt.xlabel('ROI')
    plt.ylabel('Average Score')
    plt.title('Average Scores for each ROI')
    plt.xticks(rotation=90)
    plt.show()
