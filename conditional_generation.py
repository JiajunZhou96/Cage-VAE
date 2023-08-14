import os
import random
import numpy as np
import pandas as pd
import torch

from rdkit.Chem import AllChem
from rdkit import Chem

import utils
from dataset import data_prep
import encoding_utils as eutils
import analysis_utils as autils
import generation_utils as g_utils
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

# loading of supervised learning dataset
dataset = pd.read_csv(config["original_dataset"])
# loading of unsupervised learning dataset
undataset = pd.read_csv(config["augmented_dataset"])
train_dataloader, test_dataloader = data_prep(dataset, undataset)

model = VAE.load_VAE(pretrained = True)
model.eval()

index_to_smile = data_prep.index_to_smile
ordinalenc = data_prep.ordinalenc


############################### interpolation

nc_idx, c_idx = VAE.extract_high_prob(data_prep.supervised_input, model, threshold = 0.8)


g_utils.interpolation_generation(lerp_type = "slerp-in", steps = config["lerp_steps"], threshold = config["lerp_threshold"], model = model,
                                    data_input = data_prep.supervised_input,
                                    nc_idx = nc_idx, c_idx = c_idx,
                                    index_to_smile = index_to_smile, ordinalenc = ordinalenc,
                                    dataset = dataset,
                                    batch = config["intp_batch"],
                                    current_batch = config["current_batch"],
                                    log_file = config["intp_file"], use_filter = True)

############################### bayesian optimization

from scipy.stats import multivariate_normal
import json

bo_domain = g_utils.obtain_domain(data_prep.supervised_input, model)

def molecular_optimization(batch, model, start_mol_idx, bo_domain, log_file):
    p_weight = 0.0025
    x_mol, _ = model.input_to_latent(data_prep.supervised_input[start_mol_idx])
    x_mol = x_mol.detach().cpu().numpy()
    mvn = multivariate_normal(mean = np.zeros(128), cov = np.identity(128))
    prior_x = mvn.logpdf(x_mol + 1e-9).reshape(-1, 1)
    log_y = torch.log(model.latent_to_prob(torch.Tensor(x_mol).cuda())).detach().cpu().numpy()- p_weight * prior_x.reshape(-1,1)
    
    if not os.path.exists('./log'):
        os.makedirs('./log')
    with open(log_file, 'w') as bo_log:
        bo_log.write("Gaussian Process Starting Point" + "\n" + 
                    "Initial bb1 smile: " + str(dataset['bb1_smile'][start_mol_idx]) + "\n" +
                    "Initial bb2 smile: " + str(dataset['bb2_smile'][start_mol_idx]) + "\n" +
                    "Initial reaction: " + str(dataset['reaction'][start_mol_idx]) + "\n" )

    with open(log_file, 'a') as bo_log:
        bo_log.write("Initial X: " + "\n")
        json.dump(x_mol.tolist(), bo_log)
        bo_log.write("acquisition prior weight: " + str(p_weight) + "\n")
        bo_log.write("=============================================================================" + "\n\n")
        
    g_utils.multi_bo(batch = batch, start_x = x_mol, start_y = log_y, 
                        model = model, bo_domain = bo_domain, 
                        dataset = dataset,
                        index_to_smile = index_to_smile,  ordinalenc = ordinalenc,
                        log_file = log_file, use_filter = True)

molecular_optimization(batch = config["bo_batch"], start_mol_idx = config["initial_idx"], log_file = config["bo_file"])