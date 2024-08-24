import numpy as np


def _load_image_metadata_category_differentiation(total_data: list[tuple]):
    total_labels = []
    for img_pair in total_data:
        numerosity = int(img_pair[0].split("-")[2])
        category = img_pair[0].split("-")[3].split("_")[0]
        # We are already mapping to action integer, since loading hetereogenous
        # arrays will cause issues with pytorchs dataloader
        action = (
            -1
            if category == "Different"
            else 0
            if category == "Animals"
            else 1
            if category == "Tools"
            else -1
        )
        total_labels.append([numerosity, action])

    return total_labels


def _calculate_targets_category_differentiation(image_metadata: list[tuple]):
    total_actions = []
    total_action_sizes = []
    for label_pair in image_metadata:
        # Differnt:           -1
        # Animals:             0
        # Tools:               1
        # None                -1
        total_actions.append(label_pair[1])
        total_action_sizes.append(abs(label_pair[1]))

    return total_actions, total_action_sizes
