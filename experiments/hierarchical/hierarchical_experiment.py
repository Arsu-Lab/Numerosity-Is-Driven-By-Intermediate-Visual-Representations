import json
import matplotlib.pyplot as plt
from typing import Optional

from Numbersense.config import ObjectiveFunction
from Numbersense.analysis.analyze_results import total_embedding_score, total_classification_accuracy

def bin_entries(models: dict):
    perf_dicts = {}
    for model in models:
        backbone, layer, trained_mode, weight_mode = model.split("-")[0], model.split("-")[1], model.split("-")[2], model.split("-")[3]
        if backbone not in perf_dicts:
            perf_dicts[backbone] = {}
        if trained_mode not in perf_dicts[backbone]:
            perf_dicts[backbone][trained_mode] = {}
        if weight_mode not in perf_dicts[backbone][trained_mode]:
            perf_dicts[backbone][trained_mode][weight_mode] = {}
        perf_dicts[backbone][trained_mode][weight_mode][layer] = models[model]
    return perf_dicts

def plot_metric(perf: dict, is_cross_validation: bool = False, save_path: Optional[str] = None):
    _, axs = plt.subplots(4, 4, figsize=(30, 20), gridspec_kw={'wspace': 0.1})
    y_lims = [[] for _ in range(4)]
    for objective_function in perf.keys():
        for model_key in perf[objective_function].keys() if is_cross_validation else [None]:
            models = perf[objective_function][model_key] if is_cross_validation else perf[objective_function]
            bins = bin_entries(models)
            for mi, model in enumerate(sorted(bins.keys())):
                first = True
                for ti, training_mode in enumerate(sorted(bins[model].keys())):
                    for wi, weight_mode in enumerate(sorted(bins[model][training_mode].keys())):
                        ax = axs[mi, 2*ti + wi]
                        keys = sorted([int(k) for k in bins[model][training_mode][weight_mode].keys()])
                        values = [bins[model][training_mode][weight_mode][str(k)]["average"] for k in keys]
                        std_devs = [bins[model][training_mode][weight_mode][str(k)]["std_deviation"] for k in keys]
                        if first:
                            ax.set_ylabel('Accuracy in %')
                        ax.set_xlabel('Layers / Blocks')
                        ax.set_title(f'{model}-{training_mode}-{weight_mode}')
                        ax.grid(True)
                        color = next(ax._get_lines.prop_cycler)['color']  # Get the next color in the cycle for each outer iteration
                        ax.scatter(keys, values, color=color)
                        ax.plot(keys, values, label=model_key if is_cross_validation else "Dataset 4" if "background" in objective_function else "Dataset 3" if "object" in objective_function else "Dataset 2" if "fixed_image" in objective_function else "Dataset 1", color=color)
                        ax.fill_between(keys, [v - s for v, s in zip(values, std_devs)], [v + s for v, s in zip(values, std_devs)], color=color, alpha=0.2)
                        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Ensure x ticks are shown as integers
                        y_lims[mi].append(ax.get_ylim())
                        first = False
    for mi in range(4):
        for ax in axs[mi]:
            ax.set_ylim(min([ylim[0] for ylim in y_lims[mi]]), max([ylim[1] for ylim in y_lims[mi]]))
            if ax == axs[3][-1]:
                ax.legend()
    plt.savefig(save_path) if save_path else plt.show()

experiment_path = "/scratch/modelrep/sadiya/students/elias/hierarchical_experiment"
metric = "accuracy" # embedding_sharpeness or "explained_variance"

perf_dicts = {}
for obj_function in [ObjectiveFunction.VARYING_SIZE, 
                     ObjectiveFunction.VARYING_SIZE_FIXED_IMAGE_FIXED_BETWEEN,
                     ObjectiveFunction.VARYING_SIZE_AND_OBJECT,
                     ObjectiveFunction.VARYING_SIZE_AND_OBJECT_AND_BACKGROUND]:
    cross_validation = None # [obj_function.value, obj_function.value+"_4val"]
    if cross_validation:
        if obj_function.value not in perf_dicts:
            perf_dicts[obj_function.value] = {}
        for cross_obj_func in cross_validation:
            if metric == "accuracy":
                perf_dicts[obj_function.value][cross_obj_func] = total_classification_accuracy(experiment_path, obj_function.value, cross_obj_func) 
            else:
                perf_dicts[obj_function.value][cross_obj_func] = total_embedding_score(experiment_path, obj_function.value, cross_obj_func, get_explained_variance=(metric == "explained_variance"))
    else:
        perf_dicts[obj_function.value] = total_classification_accuracy(experiment_path, obj_function.value) if metric == "accuracy" else total_embedding_score(experiment_path, obj_function.value, get_explained_variance=(metric == "explained_variance"))

plot_metric(perf_dicts, cross_validation is not None, f"hierarchical_{metric}_comparison")