import matplotlib.pyplot as plt
from typing import Optional

from Numbersense.analysis.analyze_results import total_classification_accuracy
from Numbersense.utilities.helpers import getenv

task_grouping = {
    "2D": ["segment_unsup2d", "edge_texture", "keypoints2d", "autoencoding", "inpainting", "colorization", "denoising"],
    "3D": ["keypoints3d", "depth_zbuffer", "reshading", "curvature", "depth_euclidean", "normal", "segment_unsup25d", "edge_occlusion"],
    "Semantic": ["segment_semantic", "class_object", "class_scene"]
}

def plot_total_classification_accuracy(perf: dict, violin_plot: bool = True, save_path: Optional[str] = None):
    _, ax = plt.subplots(figsize=(10, 8))
    models_sorted_by_average = sorted(perf.items(), key=lambda x: x[1]['average'])
    for i, (model, stats) in enumerate(models_sorted_by_average):
        data = stats["list"]
        if any(task in model for task in task_grouping["3D"]): 
            color = "red"
        elif any(task in model for task in task_grouping["2D"]): 
            color = "green"
        else: 
            color = "orange"

        if violin_plot:
            ax.violinplot(data, positions=[i], showmeans=True)
        else:
            ax.errorbar(i, stats["average"], yerr=stats["std_deviation"], fmt='o', color=color)
    ax.set_title('Total Classification Accuracy')
    ax.set_xticks(range(len(perf)))
    ax.set_xticklabels(["-".join(model.split("-")[1:]) for model, _ in models_sorted_by_average], rotation=90)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(45, 70)
    if not violin_plot:
        import matplotlib
        red_patch = matplotlib.patches.Patch(color='red', label='3D')
        green_patch = matplotlib.patches.Patch(color='green', label='2D')
        orange_patch = matplotlib.patches.Patch(color='orange', label='Semantic')
        ax.legend(handles=[red_patch, green_patch, orange_patch], loc='lower right')
    plt.savefig(save_path, bbox_inches='tight') if save_path else plt.show()

if __name__ == "__main__":
    EXPERIMENT_PATH = getenv("EXPERIMENT_PATH", "taskonomy")
    OBJ_FUNCTION = "varying_size_fixed_image_fixed_between" # Dataset 2 ID
    VIOLIN = getenv("VIOLIN", False)
    SAVE_DICT_PATH = getenv("SAVE_DICT_PATH", "taskonomy_performance.json")

    perf_dict = total_classification_accuracy(EXPERIMENT_PATH, OBJ_FUNCTION, save_path=SAVE_DICT_PATH)
    plot_total_classification_accuracy(perf_dict, violin_plot=VIOLIN, save_path=f"taskonomy_total_accuracy_{'violin' if VIOLIN else 'errorbar'}")
