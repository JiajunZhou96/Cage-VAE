import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from encoding_utils import ligand_substitutions, construct_skeleton, obtain_skeleton
from utils import save_dataset

import warnings
warnings.filterwarnings("ignore")

# iodine is a dummy atom
cr_lib = ["ISI"," ", "Cc1cc(I)c(O)c(I)c1", "CCC(C)c1cc(I)c(O)c(I)c1", "CCC(I)CI", "IC1CCCC1I",
    "FC(F)(F)C(I)CI", "IC1CCC(I)S1", "Ic1ccc(I)cc1", "INI","COc1c(I)cc(C(C)(C)C)cc1I","CC(C)C(I)CI", "IOI", "CC1CC(I)C(I)CC1C",
    "Ic2ccc1ccc(I)cc1c2", "COc1cc(I)c(OC)cc1I", "Ic2ccc1cc(I)ccc1c2", "Ic3ccc2c(ccc1cc(I)ccc12)c3","CC(I)CI",
    "IC#CI", "ICCI", "IC1CCCC(I)C1","IC13CC2CC(C1)CC(I)(C2)C3"
    ,"IC2=CC1C=C(I)SC1S2", "IC2=CC1SC(I)=CC1S2", "IC2CN(Cc1ccccc1)CC2I", "CCOC(=O)C(I)CI"
    ,"Ic1ccccc1I", "Ic3ccc2[nH]c1ccc(I)cc1c2c3", "Cc1c(O)c(I)cc(I)c1O", "OC(I)CI", "CC(I)CCCI", "Oc1cc(I)c(O)cc1I", "IC1CNCC1I"
    , "O=C1CC(I)C(O)CC1I", "IC1CC=CCC1I", "Ic1cc2ccc3cc(I)cc4ccc(c1)c2c34"
    , "Ic3cnc2c(ccc1cc(I)cnc12)c3"
    , "Ic2cccc3cc1cccc(I)c1cc23", "Ic3ccc2ccc1ccc(I)nc1c2n3", "ICC1CCC(CI)C1", "IC1CCCCC1I", "IC#Cc1ccc(C#CI)cc1"
    , "Oc1c(I)cccc1I", "IC1CSSCC1I", "ICOCI", "ICCCI", "Ic2cc(I)cc(N1CCCC1)c2", "IN=NI", "Clc1cc(I)cc(I)c1"
    , "Ic1ccc(I)o1", "O=C1CC(I)C1I" , "Ic1cc(I)cc(Br)c1", "OB(O)c1cc(I)cc(I)c1"
    , "IC1C=CC=CC1I","Ic2c1ccccc1c(I)c3ccccc23", "CC(C)C(I)CI", "IN(I)c1ccccc1", "ICc2ccc1ccc(CI)cc1c2"
    , "Ic1ccc2c5cccc4cccc(c3ccc(I)c1c23)c45", "CCOC(=O)C(I)CI"
    , "IC1=CCC=C1I", "IC1CNCC1I", "IC1=CCC=C(I)C1", "IC1CNCC(I)C1", "IC1CSCC(I)C1", "CCc1cc(I)c(O)c(I)c1"
    , "CC(C)c1cc(I)c(O)c(I)c1", "ICNCI", "ICI", "Brc1cc(I)cc(I)c1", "Fc1cc(I)cc(I)c1", "Cc1c(I)cc(O)cc1I", "Ic1c[nH]cc1I", "CC1CCC(C)C(I)C1I"
    , "ClC(Cl)(Cl)C(I)CI", "BrC(Br)(Br)C(I)CI", "Ic1cocc1I", "Ic2cc(I)cc(C1CCCC1)c2", "IC2CC(c1ccccc1)CC2I","CC(I)I", "CC(I)C(C)I"
    , "Ic3ccc2Cc1ccc(I)cc1c2c3"]

lk_lib = ["Ic1ccc([Lr])cc1"," ", "Ic1ccc([Lr])nc1", "ICC[Lr]", "Clc1cc([Lr])ccc1I","Clc1cc(I)ccc1[Lr]", "IC([Lr])c1ccccc1", "Fc1ccc(C(I)[Lr])cc1", "Oc2ccc(c1ccc(I)cc1)cc2[Lr]",
    "Ic1ccc([Lr])cn1", "CC(I)C(=O)NC[Lr]"
    , "Oc1cc([Lr])ccc1I","Oc1cc(I)ccc1[Lr]", "COc1cc([Lr])ccc1I","COc1cc(I)ccc1[Lr]", "OCC(I)[Lr]", "N#Cc1ccc(C(I)[Lr])cc1", "Ic2ccc(c1ccc([Lr])cc1)cc2"
    , "ISc1ccccc1[Lr]"
    , "Cc1cc(C)c(C(I)[Lr])c(C)c1", "CCCCCC(I)[Lr]", "ClC#Cc1cc(I)cc([Lr])c1"
    , "Oc1ccc([Lr])cc1I","Oc1ccc(I)cc1[Lr]", "SCC(I)[Lr]", "Ic1cccc([Lr])c1", "CC(C)(I)[Lr]"
    , "IC([Lr])c1cccc2ccccc12"
    , "Clc1cc([Lr])ccc1I","Clc1cc(I)ccc1[Lr]", "IN([Lr])c1ccccc1", "Cc1cc(C)c(C(I)[Lr])c(C)c1"
    , "IC[Lr]", "CCC(I)[Lr]", "CCCC(I)[Lr]", "CCCCC(I)[Lr]", "OCCC(I)[Lr]", "OCCCC(I)[Lr]", "OC(I)[Lr]"]

replacement_smiles = ["F", "Cl","Br", "C", "CC", "C(C)C","C#C", "O", "OC", "C=C", "C=O", "N", "CN", "C(=O)N", "C(=O)C", "S", "SC", "C#N",]

new_ligands = ligand_substitutions(lk_lib, replacement_smiles, patt = '[H]')
new_ligands.extend(lk_lib)
new_ligands_de = list(set(new_ligands))

skeleton_lib = []
for cr in cr_lib:
    for lk in new_ligands_de:
        skeleton = construct_skeleton(cr, lk)
        skeleton_lib.append(skeleton)
        
# less or equal than 56
skeleton_reduced = [smile for smile in skeleton_lib if len(smile) <= 56]

dataset = pd.read_csv('./datasets/dataset_original.csv')

# obtain skeletons
dataset['bb2_skeleton'] = dataset.apply(lambda x:obtain_skeleton(x['bb2_smile'], bb_type = 2), axis=1) 
dataset['bb1_skeleton'] = dataset.apply(lambda x:obtain_skeleton(x['bb1_smile'], bb_type = 2), axis=1) 
dataset_new = dataset.reset_index(drop = True)

bb2_skeleton = dataset_new['bb2_skeleton'].tolist()
bb2_skeleton_reduced  = set(bb2_skeleton)
skeleton_reduced = set(skeleton_reduced)
skeleton_remains = list(skeleton_reduced-bb2_skeleton_reduced)

# n multiplier
n = 10
skeleton_multiple = []
skeleton_multiple += n * skeleton_remains

augmented_skeleton = pd.DataFrame({"bb2_skeleton": skeleton_multiple})

ordinalenc = OrdinalEncoder()
ordinal_treat = pd.concat([dataset['bb1_skeleton'], dataset['reaction']], axis = 1)
ordinalenc.fit(ordinal_treat)

reaction_assign = np.random.randint(len(ordinalenc.categories_[1]), size=(augmented_skeleton.shape[0],))  # len: 6 
bb1_skeleton_assign = np.random.randint(len(ordinalenc.categories_[0]), size=(augmented_skeleton.shape[0],))  # len: 51
ordinal_inverse = np.vstack((bb1_skeleton_assign, reaction_assign)).T
bb1_and_reaction = ordinalenc.inverse_transform(ordinal_inverse)
augmented_skeleton['bb1_skeleton'] = bb1_and_reaction[:,0]
augmented_skeleton['reaction'] = bb1_and_reaction[:,1]
augmented_skeleton_drop = augmented_skeleton.drop(index = augmented_skeleton[augmented_skeleton["bb2_skeleton"] == " "].index, inplace=False)

save_dataset(augmented_skeleton_drop, path = None, file_name = 'dataset_self_augmented', idx = False)