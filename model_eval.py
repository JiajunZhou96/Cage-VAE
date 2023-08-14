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

# VAE
model = VAE.load_VAE(pretrained = True)
model.eval()

index_to_smile = data_prep.index_to_smile
ordinalenc = data_prep.ordinalenc

############### random sampling 1000 molecules
gen_results = VAE.random_sampling(model = model, batch_size = 1000, index_to_smile = index_to_smile, ordinal_encoder = ordinalenc)


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

################# interpolation

# 1.interpolation between two random latent vector(lienar/slerp mode)

model.eval()
z_1 = torch.randn([config["latent_dim"]]).numpy()
z_2 = torch.randn([config["latent_dim"]]).numpy()
intp_results_rand, rand_z = VAE.interpolation(z_1, z_2, 6, model, "linear", index_to_smile, ordinalenc)
print("interpolation results between two random latent vectors")
print("distance between each interpolation", utils.dist_measure(rand_z))
print("interpolation results", intp_results_rand)

# 2.interpolation between two known latent vector

model.eval()
index_1 = 2020
index_2 = 3450
mu_1, _ = model.input_to_latent(data_prep.supervised_input[index_1])
mu_2, _ = model.input_to_latent(data_prep.supervised_input[index_2])
intp_results_known, known_z = VAE.interpolation(mu_1.cpu().detach().numpy().T.reshape(-1,), mu_2.cpu().detach().numpy().T.reshape(-1,), 7, model, 'linear', index_to_smile, ordinalenc)
print("interpolation results between two pre-defined latent vectors")
print("distance between (i) and (i-1)th interpolation", utils.dist_measure(known_z, 'neighbor'))
print("distance between (i)th and the origin(1)st", utils.dist_measure(known_z, 'first'))
print("interpolation results", intp_results_known)

#################### reconstruction samples evaluations

# 1. reconstruction a batch of molecules
original = data_prep.train_mixed_data[:1000]
recon = VAE.reconstruct_molecules_batch(original, model, index_to_smile, ordinalenc)
recon_acc = VAE.reconstruction_accuracy(original, recon, index_to_smile)
print("reconstruction accuracy of a batch", recon_acc)

# 2. reconstruction around a molecule multiple times
recon_multiple, recon_mu, recon_z = VAE.reconstruct_around_single_molecule_repeat(data_prep.supervised_input, 500, 1000, model, index_to_smile, ordinalenc)
ave_dist = VAE.single_reconstruction_analysis_tool(recon_multiple, recon_mu, recon_z)
print("reconstruction results around a single molecule", recon_multiple)

##################### PCA analysis

autils.PCA_latent_space(np.vstack((data_prep.supervised_input, data_prep.unsupervised_input)), 
                        model, seed, 'prediction', True, './figures/PCA_comb_pd.png')

autils.PCA_latent_space(data_prep.supervised_input, 
                        model, seed, 'grount_truth', True, './figures/PCA_sup_gt.png')

autils.PCA_latent_space(data_prep.supervised_input, 
                        model, seed, 'prediction', True, './figures/PCA_sup_pd.png')

