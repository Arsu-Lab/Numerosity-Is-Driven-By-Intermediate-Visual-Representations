import os


class Navigator:
    @staticmethod
    def get_dataset_path(
        base_path: str,
        objective_function: str,
        mode: str,
        background: str,
        with_set: bool = False,
        set: int = 0,
    ) -> str:
        assert os.path.isdir(base_path), "Must be valid path dir"
        assert mode in ["train", "test"], "Unknown mode. Must be either train or test"
        assert type(set) == int, "The set ID must be an integer"

        return os.path.join(
            base_path,
            "datasets",
            objective_function,
            mode,
            background,
            f"set_{set}" if with_set else "",
        )

    @staticmethod
    def get_model_path(
        base_path: str,
        objective_function: str,
        background: str,
        model_name: str,
        with_set: bool = False,
        set: int = 0,
    ):
        assert os.path.isdir(base_path), "Must be valid path dir"
        assert type(model_name) == str, "Model name must be a string"
        assert type(set) == int, "The set ID must be an integer"

        return os.path.join(
            base_path,
            "trained_models",
            objective_function,
            background,
            model_name,
            f"set_{set}" if with_set else "",
        )

    @staticmethod
    def get_next_set(path: str):
        assert os.path.isdir(path), "When looking for next set: Given path invalid"

        sets = [-1]
        sets.extend([int(x[-1:]) for x in os.listdir(path) if x.startswith("set_")])
        highest_set = max(sets)

        return os.path.join(path, f"set_{str(highest_set + 1)}")

    @staticmethod
    def get_analysis_path(
        base_path: str,
        objective_function: str,
        background: str,
        model_name: str,
        state: str,
        with_set: bool = False,
        set: int = 0,
    ):
        assert os.path.isdir(base_path), "Must be valid path dir"
        assert type(set) == int, "The set ID must be an integer"

        return os.path.join(
            base_path,
            "analysis_results",
            objective_function,
            background,
            f"set_{set}" if with_set else "",
            model_name,
            state,
        )
