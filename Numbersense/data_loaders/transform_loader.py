import importlib
import os


def _load_target_transforms(objective_function: str) -> str:
    """
    Loads the objective function specific target transform
    --------------------------------

    :param objective_function str: The objective function of the target transform

    :returns `None`
    """
    targets_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "targets")
    for transform in os.listdir(targets_path):
        if transform.endswith(".py") and transform.split(".")[0] == objective_function:
            print(f"Found data generation transform {objective_function}")

            try:
                module_path = ".".join(
                    [
                        "Numbersense",
                        "data_loaders",
                        "targets",
                        transform.split(".py")[0],
                    ]
                )
                module = importlib.import_module(module_path)
                metadata_transform = getattr(
                    module, f"_load_image_metadata_{objective_function}"
                )
                calculate_targets = getattr(
                    module, f"_calculate_targets_{objective_function}"
                )
                return metadata_transform, calculate_targets
            except ImportError as e:
                print(f"Failed to import module: {module_path}")
                print("ERROR:\n")
                print(e)
                return
    print(f"Could not find transforms for {objective_function}")
