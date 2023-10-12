import re
import itertools
import numpy as np
import pandas as pd

import rdkit
from rdkit import Chem 
from rdkit.Chem import AllChem
import stk

def ligand_substitutions(mol_library, repl_library, patt = '[H]'):
    
    mols = [Chem.MolFromSmiles(mol) for mol in mol_library]
    
    mol_Hs = [Chem.AddHs(mol) for mol in mols if mol is not None]
    
    repls = [Chem.MolFromSmiles(repl) for repl in repl_library]
    
    pattern = Chem.MolFromSmarts(patt)  # any hydrogen
    
    new_mols = []
    for mol in mol_Hs:
        for repl in repls:
            new_mol = AllChem.ReplaceSubstructs(mol, pattern, repl, replaceAll = False)
            new_mols.append(new_mol)
    
    new_dim_mols = []
    for i in range(0, len(new_mols)):
        mol = [Chem.MolToSmiles(Chem.RemoveHs(m)) for m in new_mols[i]]  # remove explicit H
        new_dim_mols.append(mol)
        
    
    return list(itertools.chain(*new_dim_mols))

def construct_skeleton(core_smile, linker_smile):
    
    # lk + cr + lk
    if core_smile != " " and linker_smile != " ":
        
        cr = stk.BuildingBlock(core_smile, [stk.IodoFactory()])
        lk = stk.BuildingBlock(linker_smile, [stk.IodoFactory()])
        
        molecule = stk.ConstructedMolecule(
            topology_graph=stk.polymer.Linear(
                                              building_blocks=(lk, cr, lk), repeating_unit='ABA', num_repeating_units=1, 
                                              orientations=(0, 0, 0), num_processes=1)
                                            )
        
        writer = stk.MolWriter()
        mol_smile = Chem.MolToSmiles(Chem.MolFromMolBlock(writer.to_string(molecule=molecule)))
    
    # lk + lk
    elif core_smile == " " and linker_smile != " ":
        
        lk = stk.BuildingBlock(linker_smile, [stk.IodoFactory()])
        
        molecule = stk.ConstructedMolecule(
            topology_graph=stk.polymer.Linear(
                                              building_blocks=(lk, lk), repeating_unit='AA', num_repeating_units=1, 
                                              orientations=(0, 0), num_processes=1)
                                            )
        writer = stk.MolWriter()
        mol_smile = Chem.MolToSmiles(Chem.MolFromMolBlock(writer.to_string(molecule=molecule)))
        
    # core only
    elif core_smile != " " and linker_smile == " ":
        
        core = Chem.MolFromSmiles(core_smile)
        pattern = Chem.MolFromSmarts("I")
        repl = Chem.MolFromSmiles('[Lr]')
        mol = AllChem.ReplaceSubstructs(core, pattern, repl, replaceAll = True, replacementConnectionPoint = 0)
        mol_smile = Chem.MolToSmiles(mol[0])
        
    elif core_smile == " " and linker_smile == " ":
        
        mol_smile = " "
    
    return mol_smile

def obtain_skeleton(smile, bb_type = 2):

    mol = Chem.MolFromSmiles(smile) 

    smarts = ["[NH2]", "[CX3H1](=O)", "C#[CH1]", "C=[CH2]", "[CX3](=O)[OX2H1]"]
    patts = [Chem.MolFromSmarts(smart) for smart in smarts]
    matches = [mol.GetSubstructMatches(patt) for patt in patts]

    match = max(matches, key=len)
    idx = matches.index(match)
    pattern = patts[idx]

    repl = Chem.MolFromSmiles('[Lr]')

    if len(match) >= bb_type:
        replaced_mol = AllChem.ReplaceSubstructs(mol, pattern, repl, replaceAll = True, replacementConnectionPoint = 0)

    else:
        raise ValueError("No value has been applied.")

    return Chem.MolToSmiles(replaced_mol[0])

#https://github.com/tblaschke/autoencoder/blob/ffea9fae646b869ea48f116ec08aa9e25463706b/src/datareader.py#L89
# 'Br', 'Cl', 'Si' are represented by '$', '¥', '£'
def extract_char_lib(smile_list):

    smiles_char = []

    pattern = re.compile('.')

    for smile in smile_list:

        smile_char = ','.join(pattern.findall(smile))
        single_char = smile_char.split(',')
        smiles_char.extend(single_char)

    smiles_char = list(set(smiles_char))
    smiles_char.sort()
    special_char = ['$', '¥', '£']
    smiles_char = special_char + smiles_char

    return  smiles_char

def len_smiles(smiles:pd.DataFrame) -> list:

    length = []
    for smile in smiles:
        length.append(len(smile))

    return length

def smi_vocab_len(smile_char_list):

    return len(smile_char_list)

def double_to_single(smile:str) -> str:

    for s, w in zip(['Br', 'Cl', '[Lr]', 'Si'], ['R', 'G', 'X', 'J']):
        smile = smile.replace(s, w)

    return smile

def single_to_double(smile: str) -> str:

    for s, w in zip(['Br', 'Cl', '[Lr]', 'Si'], ['R', 'G', 'X', 'J']):
        smile = smile.replace(w, s)

    return smile

def smile_to_idx(smile_replace, max_len: int, smile_to_index) -> list:

    if len(smile_replace) < max_len:
        smile_replace += '$' * (max_len - len(smile_replace))
    else:
        pass

    label = [None] * max_len
    for i in range(0, len(label)):
        label[i] = smile_to_index[smile_replace[i]]

    return label

def smiles_to_idx(smiles_df, max_len: int, smile_to_index) -> list:

    labels = []
    for smile in smiles_df:
        label = smile_to_idx(smile, max_len, smile_to_index)
        labels.append(label)

    return labels


def idx_to_smile(index_mol: list, index_to_smile)-> list:

    smile = ""
    for i in range(0, len(index_mol)):
        smile += index_to_smile[index_mol[i]]
        smile = single_to_double(smile)
    return smile

def idx_to_smiles(index_mols: list, index_to_smile)-> list:

    smiles = [idx_to_smile(index_mol, index_to_smile) for index_mol in index_mols]

    return smiles


def to_canonical_smile(smile):

    return Chem.MolToSmiles(Chem.MolFromSmiles(smile))

def to_canonical_smiles(smiles):

    canonical_smiles = [to_canonical_smile(smile) for smile in smiles]

    return np.array(canonical_smiles)

import selfies as sf

def to_selfie(smile):

    smile_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smile)) # canonicalize
    selfie = sf.encoder(smile_canonical)

    return selfie

def to_selfies(smiles):

    selfies = [to_selfie(smile) for smile in smiles]

    return np.array(selfies)

def selfie_vocab_len(selfie_char_list):

    return len(selfie_char_list)

def selfies_to_idx(selfies_df:pd.DataFrame, max_len, selfie_to_index) -> list:

    labels = []
    for selfie in selfies_df:
        label, _ = sf.selfies_to_encoding(selfies=selfie, vocab_stoi = selfie_to_index, pad_to_len= max_len, enc_type="both")
        labels.append(label)

    return labels

def idx_to_selfie(index_mol: list, index_to_selfie)-> list:

    selfie = ""
    for i in range(0, len(index_mol)):
        selfie += index_to_selfie[index_mol[i]]
    return selfie

def idx_to_selfies(index_mols: list, index_to_selfie)-> list:

    selfies = [idx_to_selfie(index_mol, index_to_selfie) for index_mol in index_mols]

    return selfies


def add_eos(mol: str, mode = 'smile'):

    if mode == 'selfie':
        mol += '[eos]'

    elif mode == 'smile':
        mol += '£'

    return mol

def add_sos(mol: str, mode = 'smile'):

    if mode == 'selfie':
        mol = '[sos]' + mol

    elif mode == 'smile':
        mol = '¥' + mol

    return mol

def remove_sos_eos(mol: str, mode = 'smile'):

    if mode == 'selfie':
        mol = mol.replace('[sos]', '')
        mol = mol.replace('[eos]', '')

    elif mode == 'smile':
        mol = mol.replace('¥', '')
        mol = mol.replace('£', '')

    return mol

def remove_padding(mol:str, mode = 'smile'):
    
    if mode == 'smile':
        mol = mol.replace('$', '')

    return mol

def construct_bb(skeleton_smile, reaction_type):
    
    
    skeleton_mol = Chem.MolFromSmiles(skeleton_smile)
    
    num_react = skeleton_smile.count("[Lr]")
    if num_react == 2:
        key = reaction_type[:reaction_type.find("2")]
    elif num_react == 3:
        key = reaction_type[reaction_type.find("2") + 1:reaction_type.find("3")]
    
    # smarts
    smarts_lib = {"amine":"[NH2]", "aldehyde": "[CX3H1](=O)", "alkene":"C=[CH2]", "carboxylic_acid": "[CX3](=O)[OX2H1]", "alkyne": "C#[CH1]"}
    
    pattern = Chem.MolFromSmiles('[Lr]')
    repl = Chem.MolFromSmarts(smarts_lib[key])
    replaced_mol = AllChem.ReplaceSubstructs(skeleton_mol, pattern, repl, replaceAll = True, replacementConnectionPoint = 0)
    
    # functional groups
    functional_lib = {"amine": stk.PrimaryAminoFactory, "aldehyde": stk.AldehydeFactory, "alkene": stk.TerminalAlkeneFactory, "carboxylic_acid": stk.CarboxylicAcidFactory, "alkyne": stk.TerminalAlkyneFactory}
    functional_factory =  functional_lib[key]
    
    return Chem.MolToSmiles(replaced_mol[0]), functional_factory


def is_bb(skeleton_smile, bb_type = "bb2"):
    
    num_react = skeleton_smile.count("[Lr]")
    
    if num_react == 2 and bb_type == "bb2":
        return True
    elif num_react == 3 and bb_type == "bb1":
        return True
    else:
        return False
    
