def _load_image_metadata_even_odd(total_data: list[tuple]):
    """"""
    total_labels = []
    for img_pair in total_data:
        numerosity = int(img_pair[0].split("-")[2])
        label = img_pair[0].split("-")[3].split("_")[0]

        total_labels.append((numerosity, label))

    return total_labels


def _calculate_targets_even_odd(image_metadata: list[tuple]):
    total_actions = []
    total_action_sizes = []
    for label_pair in image_metadata:
        action = (
            -1 if label_pair[1] == "different" else 0 if label_pair[1] == "even" else 1
        )
        total_actions.append(action)
        total_action_sizes.append(abs(action))

    return total_actions, total_action_sizes
