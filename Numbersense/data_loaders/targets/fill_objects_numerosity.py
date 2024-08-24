import numpy as np


def _load_image_metadata_fill_objects_numerosity(total_data: list[tuple]):
    """"""
    total_counts = []
    for img_pair in total_data:
        label_first_img = int(img_pair[0].split("-")[1].split("_")[0])
        label_second_img = int(img_pair[1].split("-")[1].split("_")[0])
        total_counts.append((label_first_img, label_second_img))
        if not (label_first_img - label_second_img) in [-1, 0, 1]:
            print(img_pair)
    return total_counts


def _calculate_targets_fill_objects_numerosity(image_metadata: list[tuple]):
    """"""
    total_actions = []
    total_action_sizes = []
    for count_pair in image_metadata:
        action = count_pair[1] - count_pair[0]
        total_actions.append(action)
        total_action_sizes.append(abs(action))

    return total_actions, total_action_sizes
