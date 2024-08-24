import os
import requests
from pathlib import Path
import torch


def getenv(key: str, default=0):
    return type(default)(os.getenv(key, default))


def get_compute():
    compute = getenv("COMPUTE", "cuda")
    acc_number = getenv("ACC_NUMBER", 0)
    if compute == "cuda":
        device = torch.device(
            f"cuda:{acc_number}" if torch.cuda.is_available() else "cpu"
        )
        # Optimize cnn kernel use
        torch.backends.cudnn.benchmark = True
    elif compute == "mps":
        device = torch.device(f"mps:{acc_number}" if torch.has_mps else "cpu")
    else:
        device = torch.device("cpu")
    print("\nUsing " + str(device) + "...")
    return device


def _get_wandb_token() -> str:
    from dotenv import load_dotenv

    if not os.path.isfile(".env"):
        raise FileNotFoundError("The .env file is missing.")

    load_dotenv()
    login_token = os.environ.get("WANDB_LOGIN")
    # Must have been set with "export MODEL_ID=... in shell script
    model_id = os.environ.get("WANDB_MODEL_ID")

    if not login_token or not model_id:
        raise ValueError("Some variables are not set.")

    return login_token, model_id

def download_weights(url, cache_dir=os.path.join(os.path.expanduser('~'), ".cache/pytorch")):
    filename = url.split("/")[-1]
    file_path = os.path.join(cache_dir, filename)

    if not os.path.exists(file_path):
        print("Downloading weights...")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_path, "wb") as file:
                for data in response.iter_content(1024): #1 Kibibyte
                    file.write(data)
        else:
            raise Exception(f"Failed to download file from {url}")

    return file_path
