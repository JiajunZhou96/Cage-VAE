import os
import random
import numpy as np
import pandas as pd
import torch

os.chdir(os.pardir)

from rdkit.Chem import AllChem
from rdkit import Chem

import utils
from dataset import data_prep_selfie
import encoding_utils as eutils
import analysis_utils as autils
import VAE

import warnings
warnings.filterwarnings("ignore")

config = utils.get_config(print_dict = False)
seed = config["seed2"]
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


dataset = pd.read_csv(config["original_dataset"])
undataset = pd.read_csv(config["augmented_dataset"])

train_dataloader, test_dataloader = data_prep_selfie(dataset, undataset)

model = VAE.load_VAE_selfie(pretrained = True)
model.eval()

index_to_selfie = data_prep_selfie.index_to_selfie
ordinalenc = data_prep_selfie.ordinalenc


gen_results = VAE.random_sampling_selfie(model = model, batch_size = 1000, index_to_selfie = index_to_selfie, ordinal_encoder = ordinalenc)
############### evaluations
validity, idx_valid, validity_list = autils.validity_smiles(gen_results['bb2_sk'])
precursor_validity = autils.precursor_validity(gen_results['bb2_sk'], validity_list)
symmetry_smiles = autils.symmetry_smiles(gen_results['bb2_sk'], validity_list)  # symmetry 方面要调整

novelty1 = autils.novelty_cages(gen_results['bb2_sk'], gen_results['bb1_sk'], gen_results['reaction type'], idx_valid, 
            (dataset['bb2_skeleton'].tolist()), 
            (dataset['bb1_skeleton'].tolist()), 
            (dataset['reaction'].tolist()))

novelty2 = autils.novelty_cages(gen_results['bb2_sk'], gen_results['bb1_sk'], gen_results['reaction type'], idx_valid, 
            (dataset['bb2_skeleton'].tolist() + undataset['bb2_skeleton'].tolist()), 
            (dataset['bb1_skeleton'].tolist() + undataset['bb1_skeleton'].tolist()), 
            (dataset['reaction'].tolist() + undataset['reaction'].tolist()))
uniqueness = autils.uniqueness_cages(gen_results['bb2_sk'], gen_results['bb1_sk'], gen_results['reaction type'], idx_valid)
print("validity", validity) # precursor/cage
print("precursor_validity", precursor_validity) # precursor
print("symmetry_smiles", symmetry_smiles) # precursor
print("novelty(original)", novelty1) # cage
print("novelty(original + augmented)", novelty2) # cage
print("uniqueness", uniqueness) # cage