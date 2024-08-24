"""Plot the action classification error for two models."""
import re

import numpy as np


def parse_correct_per_action_dictionary(
    correct_per_action_dict: dict, action_names: list[str]
):
    assert len(action_names) == 3, "`Action_names` names must be 3."
    action_metrics = [[] for _ in range(len(action_names))]
    # Parse the correct_per_action dict
    for key in correct_per_action_dict.keys():
        right_preds_per_key = correct_per_action_dict[key][0]
        total_preds_per_key = correct_per_action_dict[key][1]
        err_preds_per_key = (
            total_preds_per_key - right_preds_per_key + 1
        )  # + 1 required to not be undefined in plots
        # num := num. of objects
        if action_names[0] in key:
            num = int(float(key.strip(action_names[0])))
            action_metrics[0].append((num, err_preds_per_key, total_preds_per_key))
        elif action_names[1] in key:
            num = int(float(key.strip(action_names[1])))
            action_metrics[1].append((num, err_preds_per_key, total_preds_per_key))
        elif action_names[2] in key:
            num = int(float(key.strip(action_names[2])))
            action_metrics[2].append((num, err_preds_per_key, total_preds_per_key))
        else:
            raise Exception("Symbol not understood!")

    return action_metrics


def plot_action_error(
    correct_per_action_list: list[dict],
    axes: list,
    split_point: int,
    action_mapping: dict = None,
    compute_err_ratio: bool = False,
    colors: list = None,
    compare: bool = False,
):
    action_datas = []
    for correct_per_action_dict in correct_per_action_list:
        # Get all actions from dict:
        actions = set(
            [
                "".join(c for c in action_name if not c.isnumeric())
                for action_name in correct_per_action_dict.keys()
            ]
        )

        # Create action specific datasets:
        action_data = {}
        for action in actions:
            dataset = {
                int(key.split(action)[0]): value
                for key, value in correct_per_action_dict.items()
                if action in key
            }

            # Compute metrics
            if compute_err_ratio:
                dataset = {
                    key: 1 - value[0] / value[1] for key, value in dataset.items()
                }
            else:
                dataset = {key: value[2] for key, value in dataset.items()}

            action_data[action] = dataset
        action_datas.append(action_data)

    offset = 1e-6

    # Creating the plot
    for action_idx, (action_name, data) in enumerate(action_datas[0].items()):
        ax = axes[action_idx]
        ax.plot(
            data.keys(),
            data.values()
            if not compute_err_ratio
            else [value + offset for value in data.values()],
            color=colors[0],
            marker="o",
        )

        if compare:
            ax.plot(
                action_datas[1][action_name].keys(),
                action_datas[1][action_name].values()
                if not compute_err_ratio
                else [
                    value + offset for value in action_datas[1][action_name].values()
                ],
                color=colors[1],
                marker="o",
            )

        xlims = ax.get_xlim()
        ax.axvspan(xlims[0], split_point + 0.5, facecolor="white", alpha=0.3)
        ax.axvspan(
            split_point + 0.5, xmax=xlims[1] + 0.5, facecolor="lightgrey", alpha=0.2
        )

        if compute_err_ratio:
            ax.set_yscale("log")
        ax.tick_params(axis="y", labelsize=15)
        ax.tick_params(axis="x", labelsize=12)
        ax.set_xticks(np.unique(list(data.keys())))
        ax.set_xlim(min(data.keys()) - 0.5, max(data.keys()) + 0.5)
        if action_mapping:
            ax.set_title("{a}".format(a=action_mapping[action_name]), fontsize=18)
        else:
            ax.set_title("{a}".format(a=action_name), fontsize=18)

    return axes


# Remove later
def plot_action_error_deprecated(
    correct_per_action_lists: list[dict],
    axes: list,
    split_point: int,
    plot_params: dict = None,
    colors: list = None,
    compare: bool = False,
):
    if plot_params is None:
        plot_params = {}
    psm = plot_params.get("post_split_marker", "-o")
    # Get all actions from dict:
    action_names = set(
        [
            "".join(c for c in action_name if not c.isnumeric())
            for action_name in correct_per_action_lists[
                0
            ].keys()  # We assume structure is the same for all datasets.
        ]
    )

    assert len(axes) == len(action_names)

    # Max_objs is the same for both datasets
    all_actions = {}

    for k in range(len(correct_per_action_lists)):
        # Get all values for x-axis:
        tmp_actions = [
            {
                "nums": [
                    int(re.findall(r"\b\d+\b", key)[0])
                    for key in correct_per_action_lists[k].keys()
                    if action in key
                ]
            }
            for action in action_names
        ]

        # Add entries to hold error values for each respective action and num. of objects
        for idx, action in enumerate(tmp_actions):
            for num in action["nums"]:
                tmp_actions[idx][str(num)] = []

        dataset = correct_per_action_lists[0]
        actions_metrics = parse_correct_per_action_dictionary(
            dataset, list(action_names)
        )

        # Run over all actions!
        for i, action in enumerate(actions_metrics):
            action = sorted(
                action, key=lambda x: x[0]
            )  # Sort by numerosity. of objects, least to greatest

            # Len. of below lists varies btw. 8 & 9 - corr. pred. for a num. of obj. in a dataset
            nums, err, total = zip(*action)
            err_ratios = np.array([i / j for i, j in zip(err, total)])

            # len(nums) := max_objs
            nums = np.array(nums)

            assert len(nums) == len(err_ratios) == len(total)

            # First case:
            if i == 0:
                for n in range(len(nums)):
                    if n + 2 < len(nums):
                        tmp_actions[i][str(n + 1)].append(err_ratios[n])
            # Second case:
            elif i == 1:
                for n in range(len(nums)):
                    tmp_actions[i][str(n)].append(err_ratios[n])
            # Third case:
            elif i == 2:
                for n in range(len(nums)):
                    tmp_actions[i][str(n)].append(err_ratios[n])

        all_actions[str(k + 1)] = tmp_actions

    overall_vals = all_actions["1"]  # Same values for all datasets
    for action_idx in range(len(overall_vals)):
        ax = axes[action_idx]
        action_name = action_names[action_idx]
        action_nums = overall_vals[action_idx]["nums"]

        print("\nCreating figure for action '" + action_name + "'...")

        low_lims_1, low_lims_2 = [], []
        mean_err_1, mean_err_2 = [], []
        high_lims_1, high_lims_2 = [], []
        keys = list(overall_vals[action_idx].keys())[1:]  # First indice is 'nums'

        for key in keys:
            low_lims_1.append(np.min(all_actions["1"][action_idx][key]))
            mean_err_1.append(np.mean(all_actions["1"][action_idx][key]))
            high_lims_1.append(np.max(all_actions["1"][action_idx][key]))

            low_lims_2.append(np.min(all_actions["2"][action_idx][key]))
            mean_err_2.append(np.mean(all_actions["2"][action_idx][key]))
            high_lims_2.append(np.max(all_actions["2"][action_idx][key]))

        if action_name == action_names[0]:
            ax.plot(
                action_nums[0:split_point],
                mean_err_1[0:split_point],
                psm,
                color=colors[0],
            )
            ax.plot(
                action_nums[split_point:],
                mean_err_1[split_point:],
                psm,
                color=colors[0],
            )
            if compare:
                ax.plot(
                    action_nums[0:split_point],
                    mean_err_2[0:split_point],
                    psm,
                    color=colors[1],
                )
                ax.plot(
                    action_nums[split_point:],
                    mean_err_2[split_point:],
                    psm,
                    color=colors[1],
                )

            # Plot horizontal lines
            ax.plot(action_nums, [1 for _ in action_nums], "--", color="gold")
            ax.plot(action_nums, [0.1 for _ in action_nums], "--", color="orchid")
            ax.plot(
                action_nums,
                [0.01 for _ in action_nums],
                "--",
                color="mediumspringgreen",
            )
            ax.plot(action_nums, [0.001 for _ in action_nums], "--", color="chocolate")

            # Mark difference between predictions
            ax.fill_between(
                action_nums, low_lims_1, high_lims_1, alpha=0.3, color=colors[0]
            )
            if compare:
                ax.fill_between(
                    action_nums, low_lims_2, high_lims_2, alpha=0.3, color=colors[1]
                )

            if plot_params.get("training_limit_shadow_visible", True):
                xlims = ax.get_xlim()
                ax.axvspan(xlims[0], split_point + 0.5, facecolor="white", alpha=0.3)
                ax.axvspan(
                    split_point + 0.5, xlims[1], facecolor="lightgrey", alpha=0.2
                )

        elif action_name == action_names[1]:
            # Split between training and test region
            ax.plot(
                action_nums[0 : split_point + 1],
                mean_err_1[0 : split_point + 1],
                psm,
                color=colors[0],
            )
            ax.plot(
                action_nums[split_point + 1 :],
                mean_err_1[split_point + 1 :],
                psm,
                color=colors[0],
            )
            if compare:
                ax.plot(
                    action_nums[0 : split_point + 1],
                    mean_err_2[0 : split_point + 1],
                    psm,
                    color=colors[1],
                )
                ax.plot(
                    action_nums[split_point + 1 :],
                    mean_err_2[split_point + 1 :],
                    psm,
                    color=colors[1],
                )

            # Plot horizontal lines
            ax.plot(action_nums, [1 for _ in action_nums], "--", color="gold")
            ax.plot(action_nums, [0.1 for _ in action_nums], "--", color="orchid")
            ax.plot(
                action_nums,
                [0.01 for _ in action_nums],
                "--",
                color="mediumspringgreen",
            )
            ax.plot(action_nums, [0.001 for _ in action_nums], "--", color="chocolate")

            # Mark difference between predictions
            ax.fill_between(
                action_nums, low_lims_1, high_lims_1, alpha=0.3, color=colors[0]
            )
            if compare:
                ax.fill_between(
                    action_nums, low_lims_2, high_lims_2, alpha=0.3, color=colors[1]
                )

            if plot_params.get("training_limit_shadow_visible", True):
                # Shadow beyond training limit
                xlims = ax.get_xlim()
                ax.axvspan(xlims[0], split_point + 0.5, facecolor="white", alpha=0.3)
                ax.axvspan(
                    split_point + 0.5, xlims[1], facecolor="lightgrey", alpha=0.2
                )

        elif action_name == action_names[2]:
            ax.plot(
                action_nums[0:split_point],
                mean_err_1[0:split_point],
                psm,
                color=colors[0],
            )
            ax.plot(
                action_nums[split_point:],
                mean_err_1[split_point:],
                psm,
                color=colors[0],
            )
            if compare:
                ax.plot(
                    action_nums[0:split_point],
                    mean_err_2[0:split_point],
                    psm,
                    color=colors[1],
                )
                ax.plot(
                    action_nums[split_point:],
                    mean_err_2[split_point:],
                    psm,
                    color=colors[1],
                )

            # Plot horizontal lines
            ax.plot(action_nums, [1 for _ in action_nums], "--", color="gold")
            ax.plot(action_nums, [0.1 for _ in action_nums], "--", color="orchid")
            ax.plot(
                action_nums,
                [0.01 for _ in action_nums],
                "--",
                color="mediumspringgreen",
            )
            ax.plot(action_nums, [0.001 for _ in action_nums], "--", color="chocolate")

            # Mark difference between predictions
            ax.fill_between(
                action_nums, low_lims_1, high_lims_1, alpha=0.3, color=colors[0]
            )
            if compare:
                ax.fill_between(
                    action_nums, low_lims_2, high_lims_2, alpha=0.3, color=colors[1]
                )

            if plot_params.get("training_limit_shadow_visible", True):
                xlims = ax.get_xlim()
                ax.axvspan(
                    xlims[0], split_point - 1 + 0.5, facecolor="white", alpha=0.3
                )
                ax.axvspan(
                    split_point - 1 + 0.5, xlims[1], facecolor="lightgrey", alpha=0.2
                )

        x_labels = np.unique(action_nums)

        ax.set_yscale("log")
        ax.tick_params(axis="y", labelsize=15)
        ax.tick_params(axis="x", labelsize=12)
        ax.set_xticks(np.unique(action_nums))
        ax.set_xlim(x_labels[0] - 1, x_labels[-1] + 1)
        ax.set_title("{a}".format(a=action_name), fontsize=18)

    return axes
