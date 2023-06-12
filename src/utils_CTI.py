import sys
import scanpy as sc
import anndata
import pandas as pd
import numpy as np
import os
import time
import logging
from scipy.spatial import distance_matrix
import random
from numpy import array, dot
from qpsolvers import solve_qp
import qpsolvers
import itertools
# from qpsolvers.solvers.qpswift_ import qpswift_solve_qp
import ray

data_type = 'float32'

import matplotlib as mpl
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils_pyRCTD import *


def find_neighbors(position, q=0.004, p=1):
    pdist = distance_matrix(position,position,p = p)
    radius = np.quantile(pdist[pdist!=0],q)
    neighbors = (pdist <= radius) & (pdist > 0)
    return [np.where(neighbors[i] == 1)[0] for i in range(neighbors.shape[0])]

def KeepOrderUnique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def SlideseqP2Parser(p2_res):
    p2_res = p2_res.loc[p2_res['spot_class']!='reject']
    p2_res = p2_res.reset_index()
    p2_res.columns = ['spot_index']+p2_res.columns.tolist()[1:]
    p2_res_single = p2_res.loc[p2_res['spot_class']=='singlet']
    p2_res_single['cell_nums'] = 1
    p2_res_single['cell_index'] = p2_res_single['spot_index'].map(lambda x:x+'_0')
    p2_res_single['cell_index_n'] = '0'
    p2_res_single['cell_type'] = p2_res_single.apply(lambda x:x['first_type'] if x['first_type_weight']>x['second_type_weight'] else x['second_type'], axis=1)
    p2_res_double = p2_res.loc[p2_res['spot_class']!='singlet']
    p2_res_double['cell_nums'] = 2
    #offset = (np.linalg.norm((p2_res[['x','y']].values[1:]-p2_res[['x','y']].values[0]),axis=1).min()**2/2)**.5/10
    offset=0
    for row in p2_res_double.iterrows():
        for i in range(2):
            if i==0:
                row[1]['x'] -= offset/2
            else:
                row[1]['x'] += offset
            row[1]['cell_index'] = row[1]['spot_index']+'_{}'.format(i)
            row[1]['cell_index_n'] = str(i)
            if i==0:
                row[1]['cell_type'] = row[1]['first_type']
            else:
                row[1]['cell_type'] = row[1]['second_type']
            p2_res_single = p2_res_single.append(row[1])
    return p2_res_single.reset_index(drop=True)



@ray.remote
def UpdateCellLabel(args):
    Y,index,pos_celltype,alpha,nUMI,cell_num_total,nu = args
        
    if index.shape[0] == 1:
        score_vec = np.zeros(pos_celltype.shape[0])
        choose_from = pos_celltype
        scaler = alpha * nUMI / cell_num_total
        j_vector = df_j[index[0]]

        for k in range(choose_from.shape[0]):
            j = choose_from[k]
            mu_hat = signature_matrix.loc[:,cell_type_names[j]].values.squeeze()
            prediction = mu_hat * scaler
            likelihood = calc_log_l_vec(prediction, Y, likelihood_vars = likelihood_vars)
            if j_vector.shape[0] == 0:
                score_vec[k] = likelihood
            else:
                prior = init[j_vector][init[j_vector] != j].shape[0] * nu / j_vector.shape[0]
                score_vec[k] = likelihood + prior

        init[index] = choose_from[np.argmin(score_vec)]
        
    else:
        for h in np.random.choice(np.arange(len(com[index.shape[0] - 2])), len(com[index.shape[0] - 2]), replace = False):
            cells_index_small = com[index.shape[0] - 2][h]
            cell_index = index[np.array(cells_index_small)]
            other_index = init[index[~np.isin(index, cell_index)]]

            if other_index.shape[0] == 0:
                other_mu = 0
            else:
                other_mu = signature_matrix.loc[:,[cell_type_names[_] for _ in other_index]].values.sum(-1)
            scaler = alpha * nUMI / cell_num_total

            ct_all = [(x, y) for x in pos_celltype for y in pos_celltype]
            j_vector1 = df_j[cell_index[0]]
            j_vector2 = df_j[cell_index[1]]
            score_vec = np.zeros(len(ct_all))


            for p in np.random.choice(np.arange(len(ct_all)), len(ct_all), replace = False):
                ct = ct_all[p]
                mu_hat = other_mu + signature_matrix.loc[:,[cell_type_names[_] for _ in ct]].values.sum(-1)
                prediction = mu_hat * scaler
                likelihood = calc_log_l_vec(prediction, Y, likelihood_vars = likelihood_vars)

                if (j_vector1.shape[0] == 0) and (j_vector2.shape[0] == 0):
                    score_vec[p] = likelihood
                elif (j_vector1.shape[0] == 0) and (j_vector2.shape[0] != 0):
                    prior2 = init[j_vector2][init[j_vector2] != ct[1]].shape[0] * nu / j_vector2.shape[0]
                    score_vec[p] = likelihood + prior2
                elif (j_vector1.shape[0] != 0) and (j_vector2.shape[0] == 0):
                    prior1 = init[j_vector1][init[j_vector1] != ct[0]].shape[0] * nu / j_vector1.shape[0]
                    score_vec[p] = likelihood + prior1
                else:  
                    Ni = 0
                    for c in pos_celltype:
                        Ni = Ni + SmoothPrior(c, ct[1],init, j_vector1, cell_index[1],nu) / (SmoothPrior(ct[1], c, init, j_vector2, cell_index[0],nu))
                    logNi = np.log(Ni)
                    prior = SmoothPrior(ct[0], ct[1],init, j_vector1, cell_index[1],nu, logp = True) - logNi

                    score_vec[p] = likelihood - prior

            init[cell_index] = np.array(ct_all[np.argmin(score_vec)])
    return index,init[index]


def SingleCellTypeIdentification(InitProp, cell_locations, spot_index_name, Q_mat_all, X_vals_loc, nu = 0, n_epoch = 8, n_neighbo=10, loggings = None, hs_ST = False):
    
    global df_j,com,init,signature_matrix,cell_type_names,likelihood_vars
    

    if 'z' in cell_locations.columns and cell_locations['z'].dtype in [np.float64,np.float32,int]:
        df_j = find_neighbors(cell_locations.loc[:,['x', 'y', 'z']].values, q = n_neighbo/cell_locations.shape[0])
    else:
        df_j = find_neighbors(cell_locations.loc[:,['x', 'y']].values, q = n_neighbo/cell_locations.shape[0])
        
    sp_index = np.array(KeepOrderUnique(cell_locations[spot_index_name]))
    sp_index_table = pd.DataFrame(np.arange(sp_index.shape[0]),index = sp_index)

    if hs_ST:
        try:
            weights = InitProp['results']['weights'].loc[sp_index].values
            cell_type_names = InitProp['results']['weights'].columns.values
        except:
            weights = InitProp['results'].loc[sp_index].values
            cell_type_names = InitProp['results'].columns.values
    else:
        weights = InitProp['results'].loc[sp_index].values
        cell_type_names = InitProp['results'].columns.values
    
    alpha = weights.sum(1)
    spot_label = sp_index_table.loc[cell_locations[spot_index_name].values].values.squeeze()
    nUMI = InitProp['spatialRNA']['nUMI'].loc[sp_index].values.squeeze()
    signature_matrix = InitProp['cell_type_info']['renorm']['cell_type_means'].loc[InitProp['internal_vars']['gene_list_reg'],]

    cell_num_total = np.array([np.where(cell_locations[spot_index_name].values == sp_name)[0].shape[0] for sp_name in sp_index])

    weights_long = weights[spot_label,:]

    pos_celltype = []
    for i in range(weights.shape[0]):
        candidates = np.where(weights[i] > min(weights[i].sum() / (2 * cell_num_total.max()), weights[i].sum() / 5))[0]
        if candidates.shape[0] == 0:
            candidates = np.array([0,1,2])
        pos_celltype.append(candidates)

    pos_celltype_long = []
    for i in range(weights_long.shape[0]):
        candidates = np.where(weights_long[i] > min(weights_long[i].sum() / (2 * cell_num_total.max()), weights_long[i].sum() / 5))[0]
        if candidates.shape[0] == 0:
            candidates = np.array([0,1,2])
        pos_celltype_long.append(candidates)

    com = []
    for i in np.arange(2,cell_num_total.max() + 1):
        com.append(list(itertools.combinations(range(i), 2)))

    init = np.argmax(weights_long, axis = -1)

    sigma = InitProp['internal_vars']['sigma'] * 100
    # sigma = int(sigma)
    sigma = round(sigma)
    puck = InitProp['spatialRNA']
    MIN_UMI = InitProp['config']['UMI_min_sigma']
    
    puck_counts = puck['counts'].loc[:, sp_index]
    puck_nUMI = puck['nUMI'].loc[sp_index]

    N_fit = min(InitProp['config']['N_fit'],(puck_nUMI > MIN_UMI).sum().item())
    if N_fit == 0:
        raise ValueError('choose_sigma_c determined a N_fit of 0! This is probably due to unusually low UMI counts per bead in your dataset. Try decreasing the parameter UMI_min_sigma. It currently is {} but none of the beads had counts larger than that.'.format(MIN_UMI))
    fit_ind = np.random.choice(puck_nUMI[puck_nUMI > MIN_UMI].index, N_fit, replace = False)
    beads = puck_counts.loc[InitProp['internal_vars']['gene_list_reg'],fit_ind].values.T
    loggings.info('chooseSigma: using initial Q_mat with sigma = {}'.format(sigma/100))

    n0 = weights.shape[0]
    cellindex_table = pd.DataFrame(np.arange(n0), index = puck_counts.columns)
        
    for epoch in range(n_epoch):
        inp_args = []       
        likelihood_vars = {'Q_mat': Q_mat_all[str(sigma)], 'X_vals': X_vals_loc, 'N_X': Q_mat_all[str(sigma)].shape[1], 'K_val': Q_mat_all[str(sigma)].shape[0] - 3}        
        for i in np.random.choice(np.arange(sp_index.shape[0]), sp_index.shape[0], replace = False):
            index = np.where(spot_label == i)[0]
            Y = puck_counts.loc[InitProp['internal_vars']['gene_list_reg'], sp_index[i]].values
            inp_args.append((Y,index,pos_celltype[i],alpha[i],nUMI[i],cell_num_total[i],nu))

        init_update_res = ray.get([UpdateCellLabel.remote(arg) for arg in inp_args])
        for init_update_index,init_update in init_update_res:
            init[init_update_index] = init_update

        weights = np.zeros((N_fit, InitProp['cell_type_info']['renorm']['n_cell_types']))
        index_table = cellindex_table.loc[fit_ind].values.squeeze()
        for i in range(N_fit):
            index = np.where(spot_label == index_table[i])[0]
            for j in range(cell_num_total[index_table[i]]):
                weights[i, init[index][j]] = weights[i, init[index][j]] + 1
            weights[i] = weights[i]*alpha[index_table[i]] / cell_num_total[index_table[i]]
            
        prediction = InitProp['cell_type_info']['renorm']['cell_type_means'].loc[InitProp['internal_vars']['gene_list_reg'],:] @ weights.T * (puck_nUMI.loc[fit_ind]).values.squeeze()[None,:]
        print('Likelihood value: {}'.format(calc_log_l_vec(prediction.values.T.reshape(-1), beads.reshape(-1),likelihood_vars = likelihood_vars)))
        sigma_prev = sigma
        sigma = chooseSigma(prediction, beads.T, Q_mat_all, likelihood_vars['X_vals'], sigma)
        loggings.info('Sigma value: {}'.format(sigma/100))
        if sigma_prev == sigma  and epoch > 1:
            break
        
    InitProp['internal_vars']['sigma'] = sigma/100
    InitProp['internal_vars']['Q_mat'] = Q_mat_all[str(sigma)]
    InitProp['internal_vars']['X_vals'] = likelihood_vars['X_vals']
    cell_locations['discrete_label'] = init
    InitProp['discrete_label'] = cell_locations
    if hs_ST:
        try:
            InitProp['label2ct'] = pd.DataFrame(InitProp['results']['weights'].columns, index = np.arange(InitProp['results']['weights'].shape[1]))
        except:
            InitProp['label2ct'] = pd.DataFrame(InitProp['results'].columns, index = np.arange(InitProp['results'].shape[1]))
    else:
        InitProp['label2ct'] = pd.DataFrame(InitProp['results'].columns, index = np.arange(InitProp['results'].shape[1]))

    return InitProp

def SmoothPrior(i,j, init, j_vector, index_of_j, nu, logp = False):
    init_fack = init.copy()
    init_fack[index_of_j] = j
    U = -init_fack[j_vector][init_fack[j_vector] != i].shape[0] * nu / j_vector.shape[0]
    if logp:
        return U
    else:
        return np.exp(U)
    
    
## based on / inspired by DataPrep: https://github.com/dmcable/spacexr
def calculate_batch_effect(spatialRNA, reference, add_genes, max_cores = 4, test_mode = False, gene_cutoff = 0.000125, fc_cutoff = 0.5, gene_cutoff_reg = 0.0002, fc_cutoff_reg = 0.75, UMI_min = 20, UMI_max = 20000000, UMI_min_sigma = 300,
                         class_df = None, CELL_MIN_INSTANCE = 25, cell_type_names = None, MAX_MULTI_TYPES = 4, keep_reference = False, cell_type_info = None, loggings = None):
    config = {'gene_cutoff': gene_cutoff, 'fc_cutoff': fc_cutoff, 'gene_cutoff_reg': gene_cutoff_reg, 'fc_cutoff_reg': fc_cutoff_reg, 'UMI_min': UMI_min, 'UMI_min_sigma': UMI_min_sigma, 'max_cores': max_cores,
                 'N_epoch': 8, 'N_X': 50000, 'K_val': 100, 'N_fit': 1000, 'N_epoch_bulk' :30, 'MIN_CHANGE_BULK': 0.0001, 'MIN_CHANGE_REG': 0.001, 'UMI_max': UMI_max, 'MIN_OBS': 3, 'MAX_MULTI_TYPES': MAX_MULTI_TYPES}
    if test_mode:
        config = {'gene_cutoff': .00125, 'fc_cutoff': 0.5, 'gene_cutoff_reg': 0.002, 'fc_cutoff_reg': 0.75, 'UMI_min': 1000, 'UMI_min_sigma': 300, 'max_cores': 1,
                 'N_epoch': 1, 'N_X': 50000, 'K_val': 100, 'N_fit': 50, 'N_epoch_bulk' :4, 'MIN_CHANGE_BULK': 1, 'MIN_CHANGE_REG': 0.001, 'UMI_max': 200000, 'MIN_OBS': 3, 'MAX_MULTI_TYPES': MAX_MULTI_TYPES}
    if cell_type_names is None:
        cell_type_names = np.unique(reference['cell_types'].iloc[:,0])
    if cell_type_info is None:
        cell_type_info = {'info': process_cell_type_info(reference, cell_type_names = cell_type_names, CELL_MIN = CELL_MIN_INSTANCE, loggings = loggings), 'renorm': None}
    if not keep_reference:
        reference = create_downsampled_data(reference, n_samples = 5)
    puck_original = restrict_counts(spatialRNA, np.array(spatialRNA['counts'].index), UMI_thresh = config['UMI_min'], UMI_max = config['UMI_max'])
    loggings.info('calculate_batch_effect: getting regression differentially expressed genes: ')
    gene_list_reg = get_de_genes(cell_type_info['info'], puck_original, fc_thresh = config['fc_cutoff_reg'], expr_thresh = config['gene_cutoff_reg'], MIN_OBS = config['MIN_OBS'], loggings = loggings)
    if gene_list_reg.shape[0] == 0:
        raise ValueError("calculate_batch_effect: Error: 0 regression differentially expressed genes found")
    loggings.info('calculate_batch_effect: getting platform effect normalization differentially expressed genes: ')
    gene_list_bulk = get_de_genes(cell_type_info['info'], puck_original, fc_thresh = config['fc_cutoff'], expr_thresh = config['gene_cutoff'], MIN_OBS = config['MIN_OBS'], loggings = loggings)
    if gene_list_bulk.shape[0] == 0:
        raise ValueError("calculate_batch_effect: Error: 0 bulk differentially expressed genes found")
        
    gene_list_bulk = np.array(list(set(gene_list_bulk).union(set(add_genes))))
    puck = restrict_counts(puck_original, gene_list_bulk, UMI_thresh = config['UMI_min'], UMI_max = config['UMI_max'])
    puck = restrict_puck(puck, puck['counts'].columns)
    if class_df is None:
        class_df = pd.DataFrame(cell_type_info['info']['cell_type_names'], index = cell_type_info['info']['cell_type_names'], columns = ['class'])
    internal_vars = {'gene_list_reg': gene_list_reg, 'gene_list_bulk': gene_list_bulk, 'proportions': None, 'class_df': class_df, 'cell_types_assigned': False}
    DataPrep = {'spatialRNA': puck, 'originalSpatialRNA': puck_original, 'reference': reference, 'config': config, 'cell_type_info': cell_type_info, 'internal_vars': internal_vars}
    
    estimated_be = cal_be(DataPrep, loggings = loggings)
    
    return estimated_be


def cal_be(DataPrep, loggings = None):
    bulkData = prepareBulkData(DataPrep['cell_type_info']['info']['cell_type_means'], DataPrep['spatialRNA'], DataPrep['internal_vars']['gene_list_bulk'])
    loggings.info('fitBulk: decomposing bulk')
    decompose_results = decompose_full(bulkData['X'],DataPrep['spatialRNA']['nUMI'].sum().item(),
                                      bulkData['b'], verbose = False, constrain = False, MIN_CHANGE = DataPrep['config']['MIN_CHANGE_BULK'],
                                      n_iter = 100, bulk_mode = True, loggings = loggings)
    DataPrep['internal_vars']['proportions'] = decompose_results['weights']
    DataPrep['cell_type_info']['renorm'] = DataPrep['cell_type_info']['info'].copy()

    proportions = decompose_results['weights']
    bulk_vec = DataPrep['spatialRNA']['counts'].sum(1)
    weight_avg = (DataPrep['cell_type_info']['info']['cell_type_means'].loc[DataPrep['internal_vars']['gene_list_bulk'], :] * (proportions / proportions.sum()).values.squeeze()).sum(1)
    target_means = bulk_vec.loc[DataPrep['internal_vars']['gene_list_bulk']] / DataPrep['spatialRNA']['nUMI'].sum().item()
    return target_means / weight_avg