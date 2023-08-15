import os
from collections import Counter
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

import utils
import VAE

config = utils.get_config(print_dict = False)

def validity_smiles(smiles_list):

    m = [Chem.MolFromSmiles(mol) for mol in smiles_list]
    idx_valid = [i for i,mol in enumerate(m) if mol != None]
    validity = len(idx_valid) / len(smiles_list)
    validity_list = [smiles_list[i] for i in idx_valid]

    return validity, idx_valid, validity_list

def novelty_smiles(smiles_list, validity_list, smiles_dataset):

    m_dataset = [Chem.MolToSmiles(Chem.MolFromSmiles(smile)) for smile in smiles_dataset]
    m = [Chem.MolToSmiles(Chem.MolFromSmiles(smile)) for smile in validity_list]
    novel = [mol for mol in m if mol not in m_dataset]

    return len(novel) / len(smiles_list)

def uniqueness_smiles(smiles_list, validity_list):

    m = [Chem.MolToSmiles(Chem.MolFromSmiles(smile)) for smile in validity_list]
    unique = list(set(m))

    return len(unique) / len(smiles_list)


def novelty_cages(smiles_list, bb1_list, reaction_list, idx_valid, smiles_dataset, bb1_dataset, reaction_dataset):

    smiles_valid = [smiles_list[i] for i in idx_valid]
    bb1_valid = [bb1_list[i] for i in idx_valid]
    reaction_valid = [reaction_list[i] for i in idx_valid]
    cage_valid = [(str(smiles_valid[i]) + str(reaction_valid[i]) + str(bb1_valid[i])) for i in range(0, len(smiles_valid))]

    cage_dataset = [(str(smiles_dataset[i]) + str(reaction_dataset[i]) + str(bb1_dataset[i])) for i in range(0, len(smiles_dataset))]
    novel_cage = [cage for cage in cage_valid if cage not in cage_dataset]

    return len(novel_cage)/ len(smiles_list)

def uniqueness_cages(smiles_list, bb1_list, reaction_list, idx_valid):

    smiles_valid = [smiles_list[i] for i in idx_valid]
    bb1_valid = [bb1_list[i] for i in idx_valid]
    reaction_valid = [reaction_list[i] for i in idx_valid]

    cage_valid = [(str(smiles_valid[i]) + str(reaction_valid[i]) + str(bb1_valid[i])) for i in range(0, len(smiles_valid))]
    unique_cage = list(set(cage_valid))

    return len(unique_cage) / len(smiles_list)

def precursor_validity(smiles_list, validity_list):

    # num = 2
    num_react = [smile.count("[Lr]") for smile in validity_list]

    count_dict = dict(Counter(num_react))

    return count_dict[2] / len(smiles_list)

import re
import encoding_utils as eutils
import pymatgen.core
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
def point_group_symmetry(smiles):
    
    rdkit_mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(rdkit_mol)
    rdkit_xyz = Chem.rdmolfiles.MolToXYZBlock(rdkit_mol)
    pm_mol = pymatgen.core.Molecule.from_str(rdkit_xyz, fmt = 'xyz')
    finder = PointGroupAnalyzer(pm_mol)
    point_result = PointGroupAnalyzer.get_pointgroup(finder)
    result = str(point_result)
    
    if result in ["C2", "C*v", "C2h", "C2v", "D*h", "D2h"]:
        return True
    
    else:
        return False

def symmetry_smiles(smiles_list, validity_list, sym = "point_group"):
    
    num_react = [smile.count("[Lr]") for smile in validity_list]
    index_candidate = [i for i, v in enumerate(num_react) if v == 2]
    
    candidates = [validity_list[i] for i in index_candidate]
    
    if sym == "point_group":
        m = [point_group_symmetry(smile) for smile in candidates]
    
    return len(m)/len(smiles_list)
    
def wrong_reactionsite_detect(smiles, reaction_type):
    
    mol = Chem.MolFromSmiles(smiles)
    key = reaction_type[:reaction_type.find("2")]
    smarts_lib = {"amine":"[NH2]", "aldehyde": "[CX3H1](=O)", "alkene":"C=[CH2]", "carboxylic_acid": "[CX3](=O)[OX2H1]", "alkyne": "C#[CH1]"}
    
    matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts_lib[key]))
    
    if len(matches) == 2:
        return True
    
    else:
        return False

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def PCA_latent_space(input_data, model, random_seed, mode, figure = True, file_name = None):
    
    model.eval()
    iter = input_data.shape[0] // 1000
    
    for i in range(1, iter):
        
        _, z = model.input_to_latent(input_data[1000*(i-1): 1000*i])
        z_batch = z.detach().cpu().numpy()
        
        if i == 1:
            z_input = z_batch
        else:
            z_input = np.vstack((z_input, z_batch))
        
        if mode == 'ground_truth':
            pass
            
        elif mode == 'prediction':
            pred = model.latent_to_prob(z)
            pred_batch = pred.detach().cpu().numpy()
        
            if i == 1:
                pred_input = pred_batch
            else:
                pred_input = np.hstack((pred_input, pred_batch))
            
        torch.cuda.empty_cache()
    
    _, z = model.input_to_latent(input_data[1000*(iter - 1):])
    z_batch = z.detach().cpu().numpy()
    z_input = np.vstack((z_input,z_batch))
    if mode == 'ground_truth':
            pass
            
    elif mode == 'prediction':
        pred = model.latent_to_prob(z)
        pred_batch = pred.detach().cpu().numpy()
        pred_input = np.hstack((pred_input, pred_batch))
    
    pca = PCA(n_components=2, random_state = random_seed)
    x_pca = pca.fit_transform(z_input)
    
    if figure:
        
        cm = plt.cm.get_cmap('viridis')
        plt.figure(figsize=(16, 16))
        plt.xticks(size = 22)
        plt.yticks(size = 22)
        plt.xlabel('PCA Reduced Dimension 1',fontproperties = 'Times New Roman', size = 24)
        plt.ylabel('PCA Reduced Dimension 2',fontproperties = 'Times New Roman', size = 24)
        if mode == 'ground_truth':
            y = input_data[:, config["max_len"]+3]
            plt.scatter(x_pca[:, 0], x_pca[:,1],c= y, cmap=cm)
            plt.colorbar()
        elif mode == 'prediction':
            pred_input = pred_input.tolist()
            plt.scatter(x_pca[:, 0], x_pca[:,1],c= pred_input, cmap=cm)
            plt.colorbar()
        if not os.path.exists('./log'):
            os.makedirs('./log')
        plt.savefig(file_name)
        plt.show()
        plt.clf()
