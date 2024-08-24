import os

from Numbersense.config import (
    Background,
    Dataloader_Parameters,
    Experiment_Parameters,
    ObjectiveFunction,
)
from Numbersense.datagen.generator import Generator

os.environ["STATISTICS"] = "1"

if __name__ == "__main__":
    exp_params = Experiment_Parameters(
        save_dir="/scratch/modelrep/sadiya/students/elias/hierarchical_experiment",
        render_dir="/scratch/modelrep/sadiya/students/elias/Renders",
    )

    dataloader_params = Dataloader_Parameters(train_max_objs=4)
    gen = Generator(
        experiment_parameters=exp_params, dataloader_parameters=dataloader_params
    )

    gen.generate_dataset(
        objective_function=ObjectiveFunction.VARYING_SIZE,
        background=Background.REAL,
        validation=False,
        seed=3,
        device="cpu",
        cores=80,
    )

    gen.dataloader_parameters.test_max_objs = 4
    gen.generate_dataset(
        objective_function=ObjectiveFunction.VARYING_SIZE,
        background=Background.REAL,
        validation=True,
        seed=4,
        device="cpu",
        cores=60,
    )

    gen.dataloader_parameters.test_max_objs = 8
    gen.generate_dataset(
        objective_function=ObjectiveFunction.VARYING_SIZE,
        background=Background.REAL,
        validation=True,
        seed=4,
        device="cpu",
        cores=60,
    )
