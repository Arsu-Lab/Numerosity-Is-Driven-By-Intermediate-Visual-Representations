def _load_image_metadata_target_number(total_data: list[tuple]):
    total_labels = []
    for img_pair in total_data:
        numerosity = int(img_pair[0].split("-")[2])
        target_count1 = int(img_pair[0].split("-")[3].split("_")[0])
        target_count2 = int(img_pair[1].split("-")[3].split("_")[0])
        total_labels.append([numerosity, target_count1 - target_count2])

    return total_labels


def _calculate_targets_target_number(image_metadata: list[tuple]):
    total_actions = []
    total_action_sizes = []
    for label_pair in image_metadata:
        action = -1 if label_pair[1] > 0 else 1 if label_pair[1] < 0 else 0
        total_actions.append(action)
        total_action_sizes.append(abs(action))

    return total_actions, total_action_sizes
