import matplotlib.pyplot as plt
import pandas as pd

from utils import plot_len_bar, duplicate_table, plot_cavity_distribution
from encoding_utils import to_canonical_smiles, obtain_skeleton

import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv('./datasets/dataset_original.csv')

dataset['bb1_skeleton'] = dataset.apply(lambda x:obtain_skeleton(x['bb1_smile'], bb_type = 3), axis=1)
dataset['bb2_skeleton'] = dataset.apply(lambda x:obtain_skeleton(x['bb2_smile'], bb_type = 2), axis=1)
dataset['collapsed'].fillna(value = 1, inplace = True)

bb1_sk_smiles = to_canonical_smiles(dataset['bb1_skeleton'])
bb2_sk_smiles = to_canonical_smiles(dataset['bb2_skeleton'])

bb1_smi_stats = duplicate_table(bb1_sk_smiles)
bb2_smi_stats = duplicate_table(bb2_sk_smiles)
print('The number of bb1 skeletons:', bb1_smi_stats.shape[0])
print('The number of bb2 skeletons:', bb2_smi_stats.shape[0])

# precursor length
plot_len_bar(dataset['bb1_smile'], mode = 'smile', x_font = 12.5, color_palette = None, file_name = './figures/bb1_smile.png')
plot_len_bar(dataset['bb2_smile'], mode = 'smile', x_font = 20, color_palette = None, file_name = './figures/bb2_smile.png')
# precursor skeleton length
plot_len_bar(dataset['bb1_skeleton'], mode = 'smile', color_palette = "flare", file_name = './figures/bb1_sk.png')
plot_len_bar(dataset['bb2_skeleton'], mode = 'smile', color_palette = "flare", file_name = './figures/bb2_sk.png')

plot_cavity_distribution(dataset[dataset["collapsed"] == 0]['cavity_size'])
