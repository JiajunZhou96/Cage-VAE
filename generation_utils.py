import os
import random
import numpy as np
import torch

import encoding_utils as eutils
import analysis_utils as autils
import VAE


def obtain_domain(input_data, model):
    
    model.eval()
    iter = input_data.shape[0] // 1000
    
    if iter > 2:
        
        for i in range(1, iter):
        
            z, _ = model.input_to_latent(input_data[1000*(i-1): 1000*i])
            z_batch = z.detach().cpu().numpy()
            
            if i == 1:
                z_input = z_batch
            else:
                z_input = np.vstack((z_input,z_batch))

        z, _ = model.input_to_latent(input_data[1000*(iter - 1):])
        z_batch = z.detach().cpu().numpy()
        z_input = np.vstack((z_input,z_batch))
        
    else:
        
        z, _ = model.input_to_latent(input_data)
        z_batch = z.detach().cpu().numpy()
        z_input = np.vstack((z_input,z_batch))

    z_all_max = np.max(z_input, axis = 0)
    z_all_min = np.min(z_input, axis = 0)
        
    z_domain = list(zip(list(z_all_min),list(z_all_max))) 
        
            
    return z_domain

import GPyOpt
from scipy.stats import multivariate_normal
import json


def multi_bo(batch = 20, start_x = None, start_y = None, 
             model = None, bo_domain = None, 
             dataset = None,
             index_to_smile = None, ordinalenc =None,
             log_file = None, use_filter = True):
    
    maxiter = 50
    p_weight = 0.0025
    mvn = multivariate_normal(mean=np.zeros(128), cov=np.identity(128))
    
    with open(log_file, 'a') as bo_log:
        bo_log.write("max iter for gaussian process: " + str(maxiter))
    
    for i in range(0, batch):
        
        while True:
            
            model.eval()
            def opt_func(z_opt):

                z_opt_cuda = torch.from_numpy(np.array(z_opt)).type(torch.FloatTensor).cuda()
                
                pred_log = torch.log(model.latent_to_prob(z_opt_cuda))
                pred_cpu = pred_log.detach().cpu().numpy()
            
                prior = mvn.logpdf(z_opt + 1e-9).reshape(-1, 1)

                return pred_cpu - p_weight * prior
    
            # 1. gaussian process
            bounds = [{'name': 'var_' + str(i), 'type': 'continuous', 'domain': bo_domain[i-1]} for i in range(1, 129)]
            opt_latent = GPyOpt.methods.BayesianOptimization(f= opt_func, domain=bounds)

            opt_latent.X = start_x
            opt_latent.Y = start_y
            opt_latent.run_optimization(max_iter = maxiter)
            
            # translate to molecules
            z_gauss = torch.from_numpy(np.array(opt_latent.x_opt)).type(torch.FloatTensor).unsqueeze(0).cuda()
            gauss_results = VAE.latent_to_mol(model = model, z_input = z_gauss, batch_size = 1, index_to_smile = index_to_smile, ordinal_encoder = ordinalenc)
            
            # filter
            if use_filter:
                
                validity, idx_valid, validity_list = autils.validity_smiles([gauss_results['bb2_sk'][0]])
                
                if validity == 1:
                    
                    if eutils.is_bb(gauss_results['bb2_sk'][0], bb_type = "bb2"):
                        
                        novelty = autils.novelty_cages([gauss_results['bb2_sk'][0]], [gauss_results['bb1_sk'][0]], [gauss_results['reaction type'][0]], 
                                                idx_valid, 
                                                dataset['bb2_skeleton'].tolist(), dataset['bb1_skeleton'].tolist(), dataset['reaction'].tolist())
                        if novelty == 1: 
                            
                            try:
                                bb1_string, bb1_factory = eutils.construct_bb(gauss_results['bb1_sk'][0], gauss_results['reaction type'][0])
                                bb2_string, bb2_factory = eutils.construct_bb(gauss_results['bb2_sk'][0], gauss_results['reaction type'][0])
                                
                                if autils.point_group_symmetry(gauss_results['bb2_sk'][0]):
                                    
                                    if autils.wrong_reactionsite_detect(bb2_string, gauss_results['reaction type'][0]):
                                        break
                                    
                                    else:
                                        continue
                                else:
                                    continue
                            except:
                                continue
                        else:
                            continue
                    else:
                        continue
                else:
                    continue
            else:
                pass

        with open(log_file, 'a') as bo_log:
            bo_log.write("***Gaussian Process epoch [" + str(i + 1) + "/" + str(batch) +"] \n" +
                        "P weight:" + str(p_weight) + "\n" +
                        "Query shape persistency: " + "shape persistent" + "\n" +
                        "bb1 skeleton: " + str(gauss_results['bb1_sk'][0]) + "\n" +
                        "bb2 skeleton: " + str(gauss_results['bb2_sk'][0]) + "\n" +
                        "reaction type: " + str(gauss_results['reaction type'][0]) + "\n")
            
        with open(log_file, 'a') as bo_log:
            bo_log.write("Optimized latent vector X: " + "\n")
            json.dump(opt_latent.x_opt.tolist(), bo_log)
            bo_log.write("\n" + "Optimized Acquisition Function Value Y: " + str(opt_latent.fx_opt) + "\n\n")
        
        if use_filter:
            validity, idx_valid, validity_list = autils.validity_smiles([gauss_results['bb2_sk'][0]])
            if validity == 0:
                with open(log_file, 'a') as bo_log:
                    bo_log.write('validity?: No' + "\n")
            else:
                with open(log_file, 'a') as bo_log:
                    bo_log.write('validity?: Yes' + "\n")
                    
                novelty = autils.novelty_cages([gauss_results['bb2_sk'][0]], [gauss_results['bb1_sk'][0]], [gauss_results['reaction type'][0]], idx_valid, 
                    dataset['bb2_skeleton'].tolist(), 
                    dataset['bb1_skeleton'].tolist(), 
                    dataset['reaction'].tolist())
                
                if novelty == 0:
                    with open(log_file, 'a') as bo_log:
                        bo_log.write('novelty?: No' + "\n")
                else:
                    
                    with open(log_file, 'a') as bo_log:
                        bo_log.write('novelty?: Yes' + "\n")
                
                try:
                    bb1_string, bb1_factory = eutils.construct_bb(gauss_results['bb1_sk'][0], gauss_results['reaction type'][0])
                    bb2_string, bb2_factory = eutils.construct_bb(gauss_results['bb2_sk'][0], gauss_results['reaction type'][0])

                    with open(log_file, 'a') as bo_log:
                        bo_log.write("bb1 string with reactive: " + str(bb1_string) + "\n" +
                                    "bb2 string with reactive: " + str(bb2_string) + "\n" +
                                    "bb1 factory: " + str(bb1_factory) + "\n" +
                                    "bb2 factory: " + str(bb2_factory) + "\n"
                                    )
                        
                except:
                    with open(log_file, 'a') as bo_log:
                        bo_log.write("problem construct cage" + "\n")
                        
                
            with open(log_file, 'a') as bo_log:
                bo_log.write("=============================================================================" + "\n\n")
                
                
########## interpolation

def intp_sampling(lerp_type = "slerp-in", steps = 6, threshold = 0.8, model = None,
                  data_input = None,
                  nc_idx = None, c_idx = None,
                  index_to_smile = None, ordinalenc = None
                  ):

    index_i, index_2 = np.random.choice(nc_idx, size = 2)
    index_3 = np.random.choice(c_idx)

    index_f = random.choice([index_2, index_3])

    mu_1, _ = model.input_to_latent(data_input[index_i])
    mu_2, _ = model.input_to_latent(data_input[index_f])
    intp_results, known_z = VAE.interpolation(mu_1.cpu().detach().numpy().T.reshape(-1,), mu_2.cpu().detach().numpy().T.reshape(-1,), steps, model, lerp_type, index_to_smile, ordinalenc)   # 这里的 6 是 distance 是一个可以调整的因素
    intp_results["probability"] = model.latent_to_prob(known_z).detach().cpu().numpy()
    
    intp_results['min'] = intp_results.groupby('bb2_sk')['probability'].transform('min')
    intp_results['max'] = intp_results.groupby('bb2_sk')['probability'].transform('max')

    intp_results['gap'] = intp_results['max'] - intp_results['min']
    intp_filtered = intp_results[intp_results['gap'] < 0.2]
    
    intp_filtered = intp_results[intp_results['probability'] < (1-threshold)]

    intp_deduplicated = intp_filtered.loc[intp_filtered.groupby('bb2_sk')['probability'].idxmin()]
    intp_deduplicated.drop(['min', 'max', 'gap'], axis=1, inplace = True)
    intp_deduplicated.reset_index(inplace = True)
    
    return intp_deduplicated


def interpolation_generation(lerp_type = "slerp-in", steps = 6, threshold = 0.8, model = None,
                             data_input = None,
                             nc_idx = None, c_idx = None,
                             index_to_smile = None, ordinalenc = None,
                             dataset = None,
                             batch = None,
                             current_batch = None,
                             log_file = None, use_filter = True,
):
    
    while True:
        
        intp_deduplicated = intp_sampling(lerp_type = lerp_type, steps = steps, threshold = threshold, model = model,
                        data_input = data_input,
                        nc_idx = nc_idx, c_idx = c_idx,
                        index_to_smile = index_to_smile, ordinalenc = ordinalenc
                        )

        
        num = 0
        for i in range(0, intp_deduplicated.shape[0]):
            
            if use_filter:

                validity, idx_valid, validity_list = autils.validity_smiles([intp_deduplicated['bb2_sk'].iloc[i]])
                
                if validity == 1:
                    
                    if eutils.is_bb(intp_deduplicated['bb2_sk'].iloc[i], bb_type = "bb2"):
                        
                        novelty = autils.novelty_cages([intp_deduplicated['bb2_sk'].iloc[i]], [intp_deduplicated['bb1_sk'].iloc[i]], [intp_deduplicated['reaction type'].iloc[i]], 
                                                idx_valid, 
                                                dataset['bb2_skeleton'].tolist(), dataset['bb1_skeleton'].tolist(), dataset['reaction'].tolist())
                        if novelty == 1: 
                            
                            try:
                                bb1_string, bb1_factory = eutils.construct_bb(intp_deduplicated['bb1_sk'].iloc[i], intp_deduplicated['reaction type'].iloc[i])
                                bb2_string, bb2_factory = eutils.construct_bb(intp_deduplicated['bb2_sk'].iloc[i], intp_deduplicated['reaction type'].iloc[i])
                                
                                if autils.point_group_symmetry(intp_deduplicated['bb2_sk'].iloc[i]):
                                    
                                    if autils.wrong_reactionsite_detect(bb2_string, intp_deduplicated['reaction type'].iloc[i]):
                                        
                                        num = num + 1
                                        
                                        if not os.path.exists('./log'):
                                            os.makedirs('./log')
                                        with open(log_file, 'a') as intp_log:
                                            intp_log.write("***Interpolation epoch [" + str(current_batch) + "/" + str(batch) +"] \n" +
                                                        "number in this batch: " + str(num) + "\n" +
                                                        "Query shape persistency: " + "shape persistent" + "\n" +
                                                        "bb1 skeleton: " + str(intp_deduplicated['bb1_sk'].iloc[i]) + "\n" +
                                                        "bb2 skeleton: " + str(intp_deduplicated['bb2_sk'].iloc[i]) + "\n" +
                                                        "reaction type: " + str(intp_deduplicated['reaction type'].iloc[i]) + "\n")
                                        
                                        if use_filter:
                                            validity, idx_valid, validity_list = autils.validity_smiles([intp_deduplicated['bb2_sk'].iloc[i]])
                                            if validity == 0:
                                                with open(log_file, 'a') as intp_log:
                                                    intp_log.write('validity?: No' + "\n")
                                            else:
                                                with open(log_file, 'a') as intp_log:
                                                    intp_log.write('validity?: Yes' + "\n")
                                                    
                                                novelty = autils.novelty_cages([intp_deduplicated['bb2_sk'].iloc[i]], [intp_deduplicated['bb1_sk'].iloc[i]], [intp_deduplicated['reaction type'].iloc[i]], idx_valid, 
                                                    dataset['bb2_skeleton'].tolist(), 
                                                    dataset['bb1_skeleton'].tolist(), 
                                                    dataset['reaction'].tolist())
                                                
                                                if novelty == 0:
                                                    with open(log_file, 'a') as intp_logg:
                                                        intp_log.write('novelty?: No' + "\n")
                                                else:
                                                    
                                                    with open(log_file, 'a') as intp_log:
                                                        intp_log.write('novelty?: Yes' + "\n")
                                                
                                                try:
                                                    bb1_string, bb1_factory = eutils.construct_bb(intp_deduplicated['bb1_sk'].iloc[i], intp_deduplicated['reaction type'].iloc[i])
                                                    bb2_string, bb2_factory = eutils.construct_bb(intp_deduplicated['bb2_sk'].iloc[i], intp_deduplicated['reaction type'].iloc[i])
                                                        
                                                    with open(log_file, 'a') as intp_log:
                                                        intp_log.write("bb1 string with reactive: " + str(bb1_string) + "\n" +
                                                                    "bb2 string with reactive: " + str(bb2_string) + "\n" +
                                                                    "bb1 factory: " + str(bb1_factory) + "\n" +
                                                                    "bb2 factory: " + str(bb2_factory) + "\n"
                                                                    )
                                                        
                                                except:
                                                    with open(log_file, 'a') as intp_log:
                                                        intp_log.write("problem construct cage" + "\n")
                                                        
                                                
                                            with open(log_file, 'a') as intp_log:
                                                intp_log.write("=============================================================================" + "\n\n")
                                            
                                    else:
                                        continue
                                else:
                                    continue
                            except:
                                continue
                        else:
                            continue
                    else:
                        continue
                else:
                    continue
            else:
                pass

        current_batch = current_batch + num
        
        if current_batch >= batch + 1:
            break
