import os
from datetime import datetime
import numpy as np
import torch
from pytorch_lightning import seed_everything
from torchvision import transforms

# Fetch Current System Details
DATE = datetime.now().strftime("%d%m%Y")

# Dataset Parameters
DATASET_NAME = "dataset_full"
DATA_CSV = os.path.join(
DATA_DIR = "./../dataset/caxton_dataset/caxton_dataset_filtered.csv"
DATASET_MEAN = [0.2915257, 0.27048784, 0.14393276]
DATASET_STD = [0.066747, 0.06885352, 0.07679665]       

# Training Parameters 
INITIAL_LR = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 15

# GPU and Accelerator Parameters
NUM_NODES = 1
NUM_GPUS = 1
ACCELERATOR = "ddp"

# Set Seeds for Training
def set_seed(seed):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    seed_everything(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

# Set Make Dir Path
def make_dirs(path):
    try:
        os.makedirs(path)
    except:
        pass

# Set Pre-processing Parameters
preprocess = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.2915257, 0.27048784, 0.14393276],
            [0.2915257, 0.27048784, 0.14393276],
        )
    ],
)
