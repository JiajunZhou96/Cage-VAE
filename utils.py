import os
import json
import inspect
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import pandas as pd
import torch

from collections import Counter
from collections import OrderedDict

import encoding_utils as eutils

def get_config(path = None, print_dict = False):
    
    if path is None:
        file = './config.json'
    else:
        file = path

    f = open(file, 'r')
    line = f.read()
    config = json.loads(line)

    if print_dict:
        print(config)
    else:
        pass
    return config

def retrieve_df_name(dataframe):
    for fi in reversed(inspect.stack()):
        df_names = [df_name for df_name, df_val in fi.frame.f_locals.items() if df_val is dataframe]
        if len(df_names) > 0:
            return df_names[0]

def save_dataset(dataframe, path = None, file_name = None, idx = False):
    '''
    Use this function to save the dataframe
    '''
    if path is None:
        path = os.path.join(os.getcwd(), 'datasets')
    else:
        # path = path
        path = os.path.join(os.getcwd(), path)
    print('Current path is:', path)

    if os.path.exists(path) == True:
        pass
        print('Path already existed.')
    else:
        os.mkdir(path)
        print('Path created.')

    if file_name is None:

        dataframe.to_csv(path + '/' + retrieve_df_name(dataframe)+ '.csv', index = idx)
    else:
        dataframe.to_csv(path + '/' + file_name + '.csv', index = idx)

    print('Dataset saved successfully.')
    
    
def plot_len_bar(mols, mode = 'smile', x_font = 20, color_palette = None, file_name = None):

    if mode == 'smile':
        length = eutils.len_smiles(mols)

    sns.set(rc={'figure.figsize':(20,10)})
    sns.set_style(style='white')
    g = sns.countplot(x = length, palette = color_palette)
    if mode == 'smile':
        g.set_xlabel("Length of SMILES",  fontproperties = 'Times New Roman', fontsize = 20)
        g.set_ylabel("Number of SMILES",  fontproperties = 'Times New Roman', fontsize = 20)
        
    g.tick_params(labelsize = 20)
    g.set_xticklabels(g.get_xticklabels(), fontsize= x_font)
    if not os.path.exists('./figures'):
        os.makedirs('./figures')
    plt.savefig(file_name)  
    plt.show()
    plt.clf()

def duplicate_table(mols: np.array):

    '''
    plot the duplication mols and frequency in the dataframe
    '''

    count = Counter(mols)
    count = OrderedDict(sorted(count.items()))
    labels, values = zip(*count.items())
    labels = list(labels)
    values = list(values)
    dataframe =  pd.DataFrame({"molecules": labels, "Number": values})

    return dataframe

def plot_cavity_distribution(dataset):
    
    sns.set(rc={'figure.figsize':(20,10)})
    sns.set_style(style='white')
    
    g = sns.histplot(x = dataset,)
    g.set_xlabel("Cavity Size",  fontproperties = 'Times New Roman', fontsize = 20)
    g.set_ylabel("Number of Molecules",  fontproperties = 'Times New Roman', fontsize = 20)
    g.tick_params(labelsize = 20)
    if not os.path.exists('./figures'):
        os.makedirs('./figures')
    plt.savefig('./figures/fig_cavity_distribution.png')  
    plt.show()
    plt.clf()
    
def plot_persistency(dataset):
    
    sns.set(rc={'figure.figsize':(10,10)})
    sns.set_style(style='white')
    
    g = sns.countplot(x = dataset)
    g.xaxis.set_major_locator(ticker.MultipleLocator(1))
    g.set_xlabel("Shape Persistency(Collapsed?)",  fontproperties = 'Times New Roman', fontsize = 20)
    g.set_ylabel("Number of Molecules",  fontproperties = 'Times New Roman', fontsize = 20)
    g.tick_params(labelsize = 20)
    if not os.path.exists('./figures'):
        os.makedirs('./figures')
    plt.savefig('./figures/fig_persistence.png')  
    plt.show()
    plt.clf()
    
def save_vocab(list, vocab_name):
    list_array = np.array(list)
    np.save('./vocab/' + vocab_name + '.npy', list_array)

def load_vocab(path):

    vocab = np.load(path)
    vocab = vocab.tolist()

    return vocab

    
def plot_training(batches_list, test_recon_losses, test_c_recon_losses, KL_losses, test_prop_losses):
    
    plt.figure(figsize=(16, 8))
    plt.xticks(size = 22)
    plt.yticks(size = 22)
    plt.xlabel('Number of Total Batches', fontproperties = 'Times New Roman', fontsize = 24)
    plt.ylabel('Losses', fontproperties = 'Times New Roman', fontsize = 24)

    plt.semilogy(batches_list, test_recon_losses, color = 'blue', linestyle = '--', linewidth = 1, label = 'Test Reconstruction Loss')
    plt.semilogy(batches_list, test_c_recon_losses, color = 'cyan', linestyle = '--', linewidth = 1, label = 'Test Category Reconstruction Loss')
    plt.semilogy(batches_list, KL_losses, color = 'red', linestyle = '--', linewidth = 1, label = 'KL Loss')
    plt.semilogy(batches_list, test_prop_losses, color = 'orange', linestyle = '--', linewidth = 1, label = 'Test Property Loss')
    plt.legend(fontsize = 12)
    if not os.path.exists('./figures'):
        os.makedirs('./figures')
    plt.savefig('./figures/training.png')  
    plt.show()
    plt.clf()
    
def plot_schedular(epoches_list, constant_recon_schedule, recon_loss_linear_schedule, prop_loss_linear_schedule, cyclic_KL_schedule):
    
    f, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16, 8))
    ax.plot(epoches_list, constant_recon_schedule, color = 'blue', linestyle = '--', linewidth = 1, label = 'Recon Loss Schedular')
    ax.plot(epoches_list, recon_loss_linear_schedule, color = 'cyan', linestyle = 'dotted', linewidth = 2, label = 'Caterogy Recon Loss Schedular')
    ax.plot(epoches_list, prop_loss_linear_schedule, color = 'orange', linestyle = '--', linewidth = 1, label = 'Prop Loss Schedular')

    ax2.plot(epoches_list, cyclic_KL_schedule, color = 'red', linestyle = '--', linewidth = 1, label = 'KL Schedular')

    ax.set_ylim(0, 1.2)
    ax2.set_ylim(0,0.003)

    ax.tick_params(axis="x", labelsize = 20) 
    ax.tick_params(axis="y", labelsize = 20) 
    ax2.tick_params(axis="x", labelsize = 20)
    ax2.tick_params(axis="y", labelsize = 20)

    ax.legend(loc="lower right", prop={'size': 16})
    ax2.legend(loc="lower right", prop={'size': 16})

    plt.xlabel('Number of Epoches', fontproperties = 'Times New Roman', fontsize = 24)
    f.text(0.04, 0.5, 'Schedular Magnitude', fontproperties = 'Times New Roman', fontsize = 24,  va='center', rotation='vertical')
    if not os.path.exists('./figures'):
        os.makedirs('./figures')
    plt.savefig('./figures/scheduler.png') 
    plt.show()
    plt.clf()



def dist_measure(z_tensor, mode = 'neighbor'):
    # measure the distances of two torch tensors
    if mode == 'neighbor':
        
        dists = [torch.norm(z_tensor[i] - z_tensor[i+1]) for i in range(z_tensor.shape[0]-1)]
    
    elif mode == 'first':
        
        dists = [torch.norm(z_tensor[i] - z_tensor[1]) for i in range(z_tensor.shape[0]-1)]
    
    return [dist.item() for dist in dists]

def plot_frequency(recon_smiles, recon_bb1, recon_reaction, ordinal_encoder):
    
    ordinal_inverse = np.array([recon_bb1,recon_reaction]).T
    recon_bb1_and_reaction = ordinal_encoder.inverse_transform(ordinal_inverse)
    
    mols = []
    for i in range(len(recon_smiles)):
        mol = recon_smiles[i] + '_' + str(recon_bb1_and_reaction[i][0]) + '_' + str(recon_bb1_and_reaction[i][1])
        mols.append(mol)
    
    
    count = Counter(mols)
    count = OrderedDict(sorted(count.items()))
    mols, frequency = zip(*count.items())

    mols = list(mols)
    frequency = list(frequency)
    mols_freq = zip(mols, frequency)
    sorted_mols_freq = sorted(mols_freq, key=lambda x:x[1], reverse=True)
    result = zip(*sorted_mols_freq)
    sorted_mols, sorted_frequency = [list(x) for x in result]

    sorted_mols_select = sorted_mols[:5]
    sorted_frequency_select = sorted_frequency[:5]

    plt.figure(figsize=(20,10))
    plt.bar(sorted_mols_select, sorted_frequency_select)
    plt.xlabel('SMILES', fontproperties = 'Times New Roman', fontsize = 28, labelpad = 160)
    plt.ylabel('Frequency', fontproperties = 'Times New Roman', fontsize = 28)

    plt.tick_params(axis = 'both', which = 'major', labelsize = 10, labelbottom=False)
    if not os.path.exists('./figures'):
        os.makedirs('./figures')
    plt.savefig('./figures/figure_frequency.png')  
    plt.show()
    plt.clf()
    #print(sorted_mols_select)



