def _load_image_metadata_object_differentiation(total_data: list[tuple]):
    total_labels = []
    for img_pair in total_data:
        numerosity = int(img_pair[0].split("-")[2])
        fixed_obj_count = int(img_pair[0].split("-")[3].split("_")[0])
        total_labels.append([numerosity, fixed_obj_count])

    return total_labels


def _calculate_targets_object_differentiation(image_metadata: list[tuple]):
    total_actions = []
    total_action_sizes = []
    for label_pair in image_metadata:
        # if all objects are fixed:                1
        # if some objects are fixed                0
        # if no objets are fixed:                 -1
        # if there are no objects, we classify as -1
        action = (
            -1
            if label_pair[0] == 0
            else 1
            if label_pair[0] == label_pair[1]
            else 0
            if label_pair[1] != 0
            else -1
        )
        total_actions.append(action)
        total_action_sizes.append(abs(action))

    return total_actions, total_action_sizes
