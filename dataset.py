import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

import utils
import encoding_utils as eutils

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

config = utils.get_config(print_dict = False)

def data_prep(dataset, undataset):
    
    dataset['bb1_skeleton'] = dataset.apply(lambda x:eutils.obtain_skeleton(x['bb1_smile'], bb_type = 3), axis=1)
    dataset['bb2_skeleton'] = dataset.apply(lambda x:eutils.obtain_skeleton(x['bb2_smile'], bb_type = 2), axis=1)
    dataset['collapsed'].fillna(value = 1, inplace = True)

    dataset['bb2_replaced_smile'] = [eutils.double_to_single(smile) for smile in dataset['bb2_skeleton']]
    undataset['bb2_replaced_smile'] = [eutils.double_to_single(smile) for smile in undataset['bb2_skeleton']]

    dataset['bb2_smile_len'] =  dataset.apply(lambda x: len(x['bb2_replaced_smile']), axis = 1)
    undataset['bb2_smile_len'] =  undataset.apply(lambda x: len(x['bb2_replaced_smile']), axis = 1)

    smiles_new = dataset['bb2_replaced_smile']
    un_smiles_new = undataset['bb2_replaced_smile']

    max_len = config["max_len"]
    smile_char_list = utils.load_vocab(config["vocab"])

    data_prep.smile_to_index = dict((c,i) for i,c in enumerate(smile_char_list)) 
    data_prep.index_to_smile = dict((i,c) for i,c in enumerate(smile_char_list))

    vocab_length = eutils.smi_vocab_len(smile_char_list)

    smiles_sos = np.array([eutils.add_sos(smile, mode = 'smile') for smile in smiles_new])
    smiles_sos_eos = np.array([eutils.add_eos(smile, mode = 'smile') for smile in smiles_sos])

    un_smiles_sos = np.array([eutils.add_sos(smile, mode = 'smile') for smile in un_smiles_new])
    un_smiles_sos_eos = np.array([eutils.add_eos(smile, mode = 'smile') for smile in un_smiles_sos])

    length_list = [len(smile) for smile in smiles_sos_eos]
    smile2id_list = eutils.smiles_to_idx(smiles_sos_eos, max_len, data_prep.smile_to_index)

    un_length_list = [len(smile) for smile in un_smiles_sos_eos]
    un_smile2id_list = eutils.smiles_to_idx(un_smiles_sos_eos, max_len, data_prep.smile_to_index)

    data_prep.ordinalenc = OrdinalEncoder()
    ordinal_treat = pd.concat([dataset['bb1_skeleton'], dataset['reaction']], axis = 1)

    data_prep.ordinalenc.fit(ordinal_treat)
    features_list = data_prep.ordinalenc.transform(ordinal_treat)
    un_ordinal_treat = pd.concat([undataset['bb1_skeleton'], undataset['reaction']], axis = 1)
    un_features_list = data_prep.ordinalenc.transform(un_ordinal_treat)

    collapse_label = dataset['collapsed']
    undataset['collapsed'] = -1
    uncollapse_label = undataset['collapsed']

    dataset['indicator'] = 1
    undataset['indicator'] = 0

    # [:,:max_len],  smile2id_list，
    # [:, max_len],  length_list，
    # [:, max_len + 1:max_len + 3],  features_list，
    # [:, max_len + 3],  collapse_label，
    # [:, max_len + 4],  indicator(shape persistency)
    data_prep.supervised_input = np.hstack((np.array(smile2id_list), 
                                    np.array(length_list)[:, np.newaxis], 
                                    features_list, 
                                    np.array(collapse_label)[:, np.newaxis],  
                                    dataset['indicator'].to_numpy()[:, np.newaxis]))

    data_prep.unsupervised_input = np.hstack((np.array(un_smile2id_list), 
                                    np.array(un_length_list)[:, np.newaxis], 
                                    un_features_list, 
                                    np.array(uncollapse_label)[:, np.newaxis],  
                                    undataset['indicator'].to_numpy()[:, np.newaxis]))

    supervised_train, supervised_test = train_test_split(data_prep.supervised_input, test_size = 0.1, random_state = seed) #(32221, 57)， (1190304, 57)
    unsupervised_train, unsupervised_test = train_test_split(data_prep.unsupervised_input, test_size = 0.002, random_state = seed)  # (3581, 57)， (2386, 57)

    # mix
    data_prep.train_mixed_data = np.vstack((supervised_train, unsupervised_train))
    data_prep.test_mixed_data = np.vstack((supervised_test, unsupervised_test))

    # shuffle
    np.random.shuffle(data_prep.train_mixed_data)
    np.random.shuffle(data_prep.test_mixed_data)

    train_data = torch.from_numpy(data_prep.train_mixed_data[:,:max_len]).type(torch.LongTensor)
    test_data = torch.from_numpy(data_prep.test_mixed_data[:,:max_len]).type(torch.LongTensor)

    train_len = torch.from_numpy(data_prep.train_mixed_data[:,max_len]).type(torch.LongTensor)
    test_len = torch.from_numpy(data_prep.test_mixed_data[:,max_len]).type(torch.LongTensor)

    train_features = torch.from_numpy(data_prep.train_mixed_data[:,max_len + 1:max_len + 3]).type(torch.LongTensor)
    test_features = torch.from_numpy(data_prep.test_mixed_data[:,max_len + 1:max_len + 3]).type(torch.LongTensor)

    train_target = torch.from_numpy(data_prep.train_mixed_data[:,max_len + 3]).type(torch.LongTensor)
    test_target = torch.from_numpy(data_prep.test_mixed_data[:,max_len + 3]).type(torch.LongTensor)

    train_indicator = torch.from_numpy(data_prep.train_mixed_data[:,max_len + 4]).type(torch.LongTensor)
    test_indicator = torch.from_numpy(data_prep.test_mixed_data[:,max_len + 4]).type(torch.LongTensor)

    all_train_data = TensorDataset(train_data, train_len, train_features, train_target, train_indicator)
    all_test_data = TensorDataset(test_data, test_len, test_features, test_target, test_indicator)

    # dataloader
    train_dataloader = DataLoader(all_train_data, batch_size = 128, shuffle = True, worker_init_fn = seed_worker, generator = G)
    test_dataloader = DataLoader(all_test_data, batch_size = len(all_test_data), shuffle = True, worker_init_fn = seed_worker, generator = G)
    
    return train_dataloader, test_dataloader

import selfies as sf

def data_prep_selfie(dataset, undataset):
    
    dataset['bb1_skeleton'] = dataset.apply(lambda x:eutils.obtain_skeleton(x['bb1_smile'], bb_type = 3), axis=1)
    dataset['bb2_skeleton'] = dataset.apply(lambda x:eutils.obtain_skeleton(x['bb2_smile'], bb_type = 2), axis=1)
    dataset['collapsed'].fillna(value = 1, inplace = True)
    
    dataset['bb2_selfie_sk'] = eutils.to_selfies(dataset['bb2_skeleton'])
    undataset['bb2_selfie_sk'] = eutils.to_selfies(undataset['bb2_skeleton'])
    
    max_len = 57
    
    #selfie_char = list(sf.get_alphabet_from_selfies(undataset['bb2_selfie_sk'].append( dataset['bb2_selfie_sk'])))
    #special_char = ['[nop]','[sos]','[eos]']
    #selfie_char_list = special_char + selfie_char
    selfie_char_list = utils.load_vocab("./vocab/selfie_vocab.npy")
    
    data_prep_selfie.selfie_to_index = dict((c,i) for i,c in enumerate(selfie_char_list))
    data_prep_selfie.index_to_selfie = dict((i,c) for i,c in enumerate(selfie_char_list))

    vocab_length = eutils.selfie_vocab_len(selfie_char_list)

    selfies_sos = np.array([eutils.add_sos(smile, mode = 'selfie') for smile in dataset['bb2_selfie_sk'] ])
    selfies_sos_eos = np.array([eutils.add_eos(smile, mode = 'selfie') for smile in selfies_sos])

    un_selfies_sos = np.array([eutils.add_sos(smile, mode = 'selfie') for smile in undataset['bb2_selfie_sk'] ])
    un_selfies_sos_eos = np.array([eutils.add_eos(smile, mode = 'selfie') for smile in un_selfies_sos])

    length_list = [len(smile) for smile in selfies_sos_eos]
    selfie2id_list = eutils.selfies_to_idx(selfies_sos_eos, max_len, data_prep_selfie.selfie_to_index)

    un_length_list = [len(smile) for smile in un_selfies_sos_eos]
    un_selfie2id_list = eutils.selfies_to_idx(un_selfies_sos_eos, max_len, data_prep_selfie.selfie_to_index)

    data_prep_selfie.ordinalenc = OrdinalEncoder()
    ordinal_treat = pd.concat([dataset['bb1_skeleton'], dataset['reaction']], axis = 1)

    data_prep_selfie.ordinalenc.fit(ordinal_treat)
    features_list = data_prep_selfie.ordinalenc.transform(ordinal_treat)

    un_ordinal_treat = pd.concat([undataset['bb1_skeleton'], undataset['reaction']], axis = 1)
    un_features_list = data_prep_selfie.ordinalenc.transform(un_ordinal_treat)

    collapse_label = dataset['collapsed'] # shape persistency 输入！！

    undataset['collapsed'] = -1  # 设置为 -1
    uncollapse_label = undataset['collapsed'] # shape persistency 输入！！

    dataset['indicator'] = 1
    undataset['indicator'] = 0

    data_prep_selfie.supervised_input = np.hstack((np.array(selfie2id_list), 
                                    np.array(length_list)[:, np.newaxis], 
                                    features_list, 
                                    np.array(collapse_label)[:, np.newaxis],  
                                    dataset['indicator'].to_numpy()[:, np.newaxis])) 

    data_prep_selfie.unsupervised_input = np.hstack((np.array(un_selfie2id_list), 
                                    np.array(un_length_list)[:, np.newaxis], 
                                    un_features_list, 
                                    np.array(uncollapse_label)[:, np.newaxis],  
                                    undataset['indicator'].to_numpy()[:, np.newaxis]))

    supervised_train, supervised_test = train_test_split(data_prep_selfie.supervised_input, test_size = 0.1, random_state = seed)
    unsupervised_train, unsupervised_test = train_test_split(data_prep_selfie.unsupervised_input, test_size = 0.002, random_state = seed)

    data_prep_selfie.train_mixed_data = np.vstack((supervised_train, unsupervised_train))
    data_prep_selfie.test_mixed_data = np.vstack((supervised_test, unsupervised_test))

    np.random.shuffle(data_prep_selfie.train_mixed_data)
    np.random.shuffle(data_prep_selfie.test_mixed_data)

    train_data = torch.from_numpy(data_prep_selfie.train_mixed_data[:,:max_len]).type(torch.LongTensor)
    test_data = torch.from_numpy(data_prep_selfie.test_mixed_data[:,:max_len]).type(torch.LongTensor)

    train_len = torch.from_numpy(data_prep_selfie.train_mixed_data[:,max_len]).type(torch.LongTensor)
    test_len = torch.from_numpy(data_prep_selfie.test_mixed_data[:,max_len]).type(torch.LongTensor)

    train_features = torch.from_numpy(data_prep_selfie.train_mixed_data[:,max_len + 1:max_len + 3]).type(torch.LongTensor)
    test_features = torch.from_numpy(data_prep_selfie.test_mixed_data[:,max_len + 1:max_len + 3]).type(torch.LongTensor)

    train_target = torch.from_numpy(data_prep_selfie.train_mixed_data[:,max_len + 3]).type(torch.LongTensor)
    test_target = torch.from_numpy(data_prep_selfie.test_mixed_data[:,max_len + 3]).type(torch.LongTensor)

    train_indicator = torch.from_numpy(data_prep_selfie.train_mixed_data[:,max_len + 4]).type(torch.LongTensor)
    test_indicator = torch.from_numpy(data_prep_selfie.test_mixed_data[:,max_len + 4]).type(torch.LongTensor)

    all_train_data = TensorDataset(train_data, train_len, train_features, train_target, train_indicator)
    all_test_data = TensorDataset(test_data, test_len, test_features, test_target, test_indicator)

    train_dataloader = DataLoader(all_train_data, batch_size = 128, shuffle = True, worker_init_fn = seed_worker, generator = G)
    test_dataloader = DataLoader(all_test_data, batch_size = len(all_test_data), shuffle = True, worker_init_fn = seed_worker, generator = G)
    
    return train_dataloader, test_dataloader
    
