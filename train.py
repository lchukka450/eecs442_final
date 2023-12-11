import torch
import torch.nn as nn
import torch.optim as optim
import dataset 
import os 
from PIL import Image
import json

from model import AffordEnc
from dataset import PhysObsDataset


def train(model: AffordEnc,
          train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader,
          num_epochs: int = 10,
          device: str = "cuda",
          save_dir: str = "exps",
          lr: float = 1e-3,
          weight_decay: float = 1e-5,
          log_every: int = 100,
          save_every: int = 1):
    
    # create exps if it doesn't exist
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # create checkpoints and logs if they don't exist
    if not os.path.exists(os.path.join(save_dir, "checkpoints")):
        os.mkdir(os.path.join(save_dir, "checkpoints"))
    if not os.path.exists(os.path.join(save_dir, "logs")):
        os.mkdir(os.path.join(save_dir, "logs"))

    # write hyperparams to a json file
    hyperparams = {
        "num_epochs": num_epochs,
        "device": device,
        "save_dir": save_dir,
        "lr": lr,
        "weight_decay": weight_decay,
        "log_every": log_every,
        "save_every": save_every
    }
    with open(os.path.join(save_dir, "hyperparams.json"), "w") as f:
        json.dump(hyperparams, f)

    # define loss function and optimizer