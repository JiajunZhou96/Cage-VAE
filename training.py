# to train and save a model
import random
import numpy as np
import pandas as pd
import torch

import utils
from dataset import data_prep
import encoding_utils as eutils
import VAE

import warnings
warnings.filterwarnings("ignore")

config = utils.get_config(print_dict = False)
seed = config["seed"]
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# for dataloader
def seed_worker(seed):
    worker_seed = torch.manual_seed(seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

G = torch.Generator()
G.manual_seed(seed)

# loading of supervised learning dataset
dataset = pd.read_csv(config["original_dataset"])
# loading of unsupervised learning dataset
undataset = pd.read_csv(config["augmented_dataset"])

train_dataloader, test_dataloader = data_prep(dataset, undataset)

# VAE
model = VAE.load_VAE(pretrained = False)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

model.train()
VAE.train(model = model, optimizer = optimizer, 
        train_dataloader = train_dataloader, test_dataloader = test_dataloader, 
        figures = False)

