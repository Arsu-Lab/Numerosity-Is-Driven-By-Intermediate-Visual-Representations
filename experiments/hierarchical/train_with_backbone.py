import os, sys

import torch, cornet
from torch import nn
from torchvision.models import AlexNet_Weights, ResNet18_Weights, VGG19_Weights, alexnet, resnet18, vgg19
from visualpriors.taskonomy_network import TaskonomyEncoder

from Numbersense.analysis.analyze import Analysis
from Numbersense.config import Background, Dataloader_Parameters, Experiment_Parameters, ObjectiveFunction
from Numbersense.figures.plot import Plot
from Numbersense.model.networks import EmbeddingNet
from Numbersense.training.training import Training
from Numbersense.utilities.helpers import download_weights

def check(string: str):
    if string.lower() == "true": return True
    elif string.lower() == "false": return False
    else: raise ValueError("Invalid input. Expected 'true' or 'false'.")

# Get arguments from shell script
experiment_path = sys.argv[1]
render_path = sys.argv[2]
objc_function = sys.argv[3]
cross_val_function = sys.argv[4]
background = sys.argv[5]
model_id = sys.argv[6]
pretrained = check(sys.argv[7])
freeze_backbone = check(sys.argv[8])
block = int(sys.argv[9])
training_run = int(sys.argv[10])
model = sys.argv[11]
acc_number = int(sys.argv[12])
compute = "cuda"
dataloader_workers = 8

# Set training env variables
os.environ["COMPUTE"] = str(compute)
os.environ["ACC_NUMBER"] = str(acc_number)
os.environ["NUM_WORKERS"] = str(dataloader_workers)
os.environ["VERBOSE"] = "1"
os.environ["SEED"] = str(training_run)
os.environ["MIXED_PRECISION_TRAINING"] = "0"
os.environ["WANDB"] = "0"

for val in ObjectiveFunction:
    if val.value == objc_function:
        objc_function = val
        break

for val in ObjectiveFunction:
    if val.value == cross_val_function:
        cross_val_function = val
        break

experiment_parameters = Experiment_Parameters(experiment_path, render_path)
dataloader_parameters = Dataloader_Parameters(
    train_max_objs=4, test_max_objs=8, train_epochs=30, train_num_batches=15, train_batch_sizes=360
)

BASE_URL = "https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/"
taskonomy_encoder_weight_lookup = {
    "segment_unsup2d": "segment_unsup2d_encoder-b679053a920e8bcabf0cd454606098ae85341e054080f2be29473971d4265964.pth",
    "edge_texture": "edge_texture_encoder-be2d686a6a4dfebe968d16146a17176eba37e29f736d5cd9a714317c93718810.pth",
    "keypoints2d": "keypoints2d_encoder-6b77695acff4c84091c484a7b128a1e28a7e9c36243eda278598f582cf667fe0.pth",
    "autoencoding": "autoencoding_encoder-e35146c09253720e97c0a7f8ee4e896ac931f5faa1449df003d81e6089ac6307.pth",
    "inpainting": "inpainting_encoder-bf96fbaaea9268a820a19a1d13dbf6af31798f8983c6d9203c00fab2d236a142.pth",
    "keypoints3d": "keypoints3d_encoder-7e3f1ec97b82ae30030b7ea4fec2dc606b71497d8c0335d05f0be3dc909d000d.pth",
    "segment_unsup25d": "segment_unsup25d_encoder-7d12d2500c18c003ffc23943214f5dfd74932f0e3d03dde2c3a81ebc406e31a0.pth",
    "depth_zbuffer": "depth_zbuffer_encoder-cc343a8ed622fd7ee3ce54398be8682bbbbfb5d11fa80e8d03a56a5ae4e11b09.pth",
    "reshading": "reshading_encoder-de456246e171dc8407fb2951539aa60d75925ae0f1dbb43f110b7768398b36a6.pth",
    "segment_semantic": "segment_semantic_encoder-bb3007244520fc89cd111e099744a22b1e5c98cd83ed3f116fbff6376d220127.pth",
    "class_object": "class_object_encoder-4a4e42dad58066039a0d2f9d128bb32e93a7e4aa52edb2d2a07bcdd1a6536c18.pth",
    "class_scene": "class_scene_encoder-ad85764467cddafd98211313ceddebb98adf2a6bee2cedfe0b922a37ae65eaf8.pth",
    "colorization": "colorization_encoder-5ed817acdd28d13e443d98ad15ebe1c3059a3252396a2dff6a2090f6f86616a5.pth",
    "curvature": "curvature_encoder-3767cf5d06d9c6bca859631eb5a3c368d66abeb15542171b94188ffbe47d7571.pth",
    "denoising": "denoising_encoder-b64cab95af4a2c565066a7e8effaf37d6586c3b9389b47fff9376478d849db38.pth",
    "depth_euclidean": "depth_euclidean_encoder-88f18d41313de7dbc88314a7f0feec3023047303d94d73eb8622dc40334ef149.pth",
    "edge_occlusion": "edge_occlusion_encoder-5ac3f3e918131f61e01fe95e49f462ae2fc56aa463f8d353ca84cd4e248b9c08.pth",
    "normal": "normal_encoder-f5e2c7737e4948e3b2a822f584892c342eaabbe66661576ba50db7cdd40561c5.pth",
}

if model == "VGG19":
    vgg = vgg19(weights=VGG19_Weights.DEFAULT if pretrained else None)
    features = list(vgg.features.children())
    configs = [nn.Sequential(*features[:i]) for i in [5, 10, 19, 28, 36]]
elif model == "ResNet18":
    rn18 = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    features = [
        rn18.conv1,
        rn18.bn1,
        rn18.relu,
        rn18.maxpool,
        rn18.layer1,
        rn18.layer2,
        rn18.layer3,
        rn18.layer4,
    ]
    configs = [nn.Sequential(*features[:i]) for i in [5, 6, 7, 8]]
elif "ResNet50Taskonomy" in model:
    assert (task := model.split("-")[1]) in taskonomy_encoder_weight_lookup.keys(), f"Taskonomy encoder not found: {task}"
    w_link = BASE_URL + taskonomy_encoder_weight_lookup[task]
    rn50 = TaskonomyEncoder(eval_only=False, train=True, normalize_outputs=False)
    if "colorization" in model:
        rn50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=0, bias=False) # 1 channel input for colorization
        os.environ["GRAYSCALE"] = "1" # Inform dataloader to load grayscale

    if pretrained: rn50.load_state_dict(torch.load(download_weights(w_link))["state_dict"])
    features = [
        rn50.conv1,
        rn50.bn1,
        rn50.relu,
        rn50.maxpool,
        rn50.layer1,
        rn50.layer2,
        rn50.layer3,
        rn50.layer4,
    ]
    configs = [nn.Sequential(*features[:i]) for i in [5, 6, 7, 8]]
elif model == "CORnetS":
    crnet = cornet.cornet_s(pretrained=True if pretrained else False).module
    features = [crnet.V1, crnet.V2, crnet.V4, crnet.IT]
    configs = [nn.Sequential(*features[:i]) for i in [1, 2, 3, 4]]
elif model == "AlexNet":
    anet = alexnet(weights=AlexNet_Weights.DEFAULT if pretrained else None)
    features = list(anet.features.children())
    configs = [nn.Sequential(*features[:i]) for i in [3, 6, 8, 10, 13]]
else:
    print("Backbone model not found!")

embNet = EmbeddingNet(
    embedding_dimension=2,
    backbone=configs[block - 1],
    freeze_backbone=freeze_backbone,
)
if __name__ == "__main__":
    trainer = Training(dataloader_parameters, experiment_parameters)
    trainer.train(
        objective_function=objc_function,
        background=Background.REAL,
        embedding_net=embNet,
        model_id=model_id,
        model_num=training_run - 1,
    )
    del trainer

    analyzer = Analysis(dataloader_parameters, experiment_parameters)
    analyzer.validate_accuracy(
        set_num=0,
        model_num=training_run - 1,
        background=Background.REAL,
        model_objective_function=objc_function,
        model_id=model_id,
        embedding_net=embNet,
    )
    del analyzer

    plotter = Plot(dataloader_parameters, experiment_parameters)
    plotter.plot_embeddings(
        model_objective_function=objc_function,
        background=Background.REAL,
        set_num=0,
        model_num=training_run - 1,
        model_id=model_id,
        embedding_net=embNet,
    )
    del plotter

    analyzer = Analysis(dataloader_parameters, experiment_parameters)
    analyzer.validate_accuracy(
        set_num=0,
        model_num=training_run - 1,
        background=Background.REAL,
        model_objective_function=objc_function,
        data_objective_function=cross_val_function,
        model_id=model_id,
        embedding_net=embNet,
    )
    del analyzer

    plotter = Plot(dataloader_parameters, experiment_parameters)
    plotter.plot_embeddings(
        model_objective_function=objc_function,
        data_objective_function=cross_val_function,
        background=Background.REAL,
        set_num=0,
        model_num=training_run - 1,
        model_id=model_id,
        embedding_net=embNet,
    )
    del plotter
