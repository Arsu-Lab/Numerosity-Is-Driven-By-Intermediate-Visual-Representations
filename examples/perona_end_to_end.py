# NOTE: REMOVE AFTER REFACTOR OF RUNTIME
import os
import numpy as np
from tqdm.std import trange
from Numbersense.config import Background, Dataloader_Parameters, Experiment_Parameters, ObjectiveFunction
from Numbersense.figures.plot import Plot
from Numbersense.training.training import Training

from Numbersense.datagen.cpu.RenderEngine_BINARY import RenderEngine
from Numbersense.utilities.helpers import getenv

def generate_dataset(seed:int, size:int, max_num:int, save_path:str, type:str = "training"):
    engine = RenderEngine(seed=seed)
    num = 0
    os.makedirs(save_path, exist_ok=True)
    for pair_idx in trange(size, desc=f"Generate for {type}"):
        engine.create_stimulus_image(num, f"{pair_idx*2+1}-{num}-img", save_path)
        action = np.random.choice([0, 1]) if num == 0 else np.random.choice([0, -1]) if num == max_num else np.random.choice([-1, 0, 1])
        num += action
        engine.create_stimulus_image(num, f"{pair_idx*2+2}-{num}-img", save_path)

if __name__ == "__main__":
    save_path = getenv("EXPERIMENT_PATH", "/home/elwa/repos/experiment")
    if getenv("GENERATE", 1):
        generate_dataset(getenv('TRAIN_SEED', 0), getenv('TRAINING_SIZE', 162000), getenv('TRAINING_MAX_NUM', 3), save_path=os.path.join(save_path,"datasets", "numerosity", "train", "plain", "set_0"))
        generate_dataset(getenv('VALIDATION_SEED', 1), getenv('VALIDATION_SIZE', 16200), getenv('VALIDATION_MAX_NUM', 8), save_path=os.path.join(save_path, "datasets", "numerosity", "test", "plain", "set_0"), type="validation")

    os.environ["COMPUTE"] = "cuda"
    os.environ["SEED"] = "0" # Weight init seed
    os.environ["MIXED_PRECISION_TRAINING"] = "1"

    if getenv("EMBEDDINGS", 1):
        dl_params = Dataloader_Parameters()
        exp_params = Experiment_Parameters(save_path)
        trainer, plotter = Training(dl_params, exp_params), Plot(dl_params, exp_params)
        trainer.train(objective_function=ObjectiveFunction.NUMEROSITY, background=Background.PLAIN, set_num=0, model_id="vanilla-perona")
        plotter.plot_embeddings(model_objective_function=ObjectiveFunction.NUMEROSITY, background=Background.PLAIN, set_num=0, model_num=0, model_id="vanilla-perona", only_embeddings=True)
    
