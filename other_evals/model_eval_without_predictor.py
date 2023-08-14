import os
import random
import numpy as np
import pandas as pd
import torch

os.chdir(os.pardir)

import utils
from dataset import data_prep
import encoding_utils as eutils
import analysis_utils as autils
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

def seed_worker(seed):
    worker_seed = torch.manual_seed(seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

G = torch.Generator()
G.manual_seed(seed)

dataset = pd.read_csv(config["original_dataset"])
undataset = pd.read_csv(config["augmented_dataset"])
train_dataloader, test_dataloader = data_prep(dataset, undataset)

model = VAE.load_VAE_only(pretrained = True)
model.eval()

autils.PCA_latent_space(data_prep.supervised_input, 
                        model, seed, 'ground_truth', True, './figures/PCA_without_predictor.png')