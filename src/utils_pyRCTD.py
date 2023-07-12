import sys
import scanpy as sc
import anndata
import pandas as pd
import numpy as np
import os
import time
import logging
import random
from numpy import array, dot
from qpsolvers import solve_qp
import qpsolvers
import itertools
# from qpsolvers.solvers.qpswift_ import qpswift_solve_qp
import psutil
import ray

data_type = 'float32'


import matplotlib as mpl
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')




def run_RCTD(RCTD, Q_mat_all, X_vals_loc, doublet_mode = 'full', loggings = None):
    print(doublet_mode)
    RCTD = fitBulk(RCTD, loggings = loggings)
    RCTD = choose_sigma_c(RCTD, Q_mat_all, X_vals_loc, loggings = loggings)
    RCTD = fitPixels(RCTD, doublet_mode = doublet_mode, loggings = loggings)
    return RCTD

 
    
    
def calc_Q_all(x, bead, likelihood_vars = None):
    X_vals = likelihood_vars['X_vals']
    Q_mat = likelihood_vars['Q_mat']
    epsilon = 1e-4
    X_max = max(X_vals)
    delta = 1e-5
    x = np.minimum(np.maximum(epsilon, x),X_max - epsilon)
    l = np.floor((x / delta) ** (2 / 3))
    l = np.minimum(l, 900) + np.floor(np.maximum(l - 900,0) / 30)
    l = l - 1 #change index to python
    l = l.astype(int)
    prop = (X_vals[l + 1] - x) / (X_vals[l+1] - X_vals[l])
    v1 = Q_mat[np.floor(bead).astype(int),l + 1]
    k = Q_mat[np.floor(bead).astype(int), l] - v1
    r1 = k * prop + v1
    v1 = Q_mat[np.floor(bead).astype(int) + 1, l+1]
    k = Q_mat[np.floor(bead).astype(int) + 1, l] - v1
    r2 = k * prop + v1
    v1 = Q_mat[np.floor(bead).astype(int) + 2, l+1]
    k = Q_mat[np.floor(bead).astype(int) + 2, l] - v1
    r3 = k * prop + v1
    return {'r1': r1,'r2': r2,'r3': r3}

def get_d1_d2(B, prediction, likelihood_vars = None):
    X_vals = likelihood_vars['X_vals']
    Q_mat = likelihood_vars['Q_mat']
    K_val = likelihood_vars['K_val']
    bead = B
    epsilon = 1e-4
    X_max = max(X_vals)
    x = np.minimum(np.maximum(epsilon, prediction), X_max - epsilon)
    Q_cur = calc_Q_all(x, bead, likelihood_vars = likelihood_vars)
    bead[bead > K_val] = K_val
    Q_k = Q_cur['r1']
    Q_k1 = Q_cur['r2']
    Q_k2 = Q_cur['r3']
    Q_d1 = 1/x * (-(bead+1)*Q_k1 + bead*Q_k)
    Q_d2 = 1/(x**2)*((bead+1)*(bead+2)*Q_k2 - bead*(2*(bead+1)*Q_k1 - (bead-1)*Q_k))
    d1_vec = Q_d1 / Q_k
    d2_vec = -Q_d1**2/(Q_k**2) + Q_d2/Q_k
    return {'d1_vec': d1_vec, 'd2_vec': d2_vec}


def calc_Q_k(x, bead, likelihood_vars = None):
    X_vals = likelihood_vars['X_vals']
    Q_mat = likelihood_vars['Q_mat']
    K_val = likelihood_vars['K_val']
    bead = np.copy(bead)
    bead[bead > K_val] = K_val
    epsilon = 1e-4
    X_max = max(X_vals)
    delta = 1e-5
    x = np.minimum(np.maximum(epsilon, x), X_max - epsilon)
    l = np.floor((x / delta) ** (2 / 3))
    l = np.minimum(l, 900) + np.floor(np.maximum(l - 900,0) / 30)
    l = l - 1 #change index to python
    l = l.astype(int)
    prop = (X_vals[l + 1] - x) / (X_vals[l+1] - X_vals[l])
    v1 = Q_mat[np.floor(bead).astype(int),l + 1]
    k = Q_mat[np.floor(bead).astype(int), l] - v1
    r1 = k * prop + v1
    return r1


def calc_log_l_vec(lamb, Y, return_vec = False, likelihood_vars = None):
    log_l_vec = -np.log(calc_Q_k(lamb,Y, likelihood_vars = likelihood_vars))
    if return_vec:
        return log_l_vec
    else:
        return log_l_vec.sum()
    
def chooseSigma(prediction, counts, Q_mat_all, X_vals, sigma):
    X = prediction.values.T.reshape(-1)
    X = np.maximum(X, 1e-4)
    Y = counts.T.reshape(-1)
    num_sample = min(1000000, X.shape[0])
    use_ind = np.random.choice(np.arange(X.shape[0]), num_sample, replace = False)
    X = X[use_ind]; Y = Y[use_ind]
    mult_fac_vec = np.arange(8,13) / 10
    sigma_ind = np.concatenate((np.arange(10,71), np.arange(72, 202,2)))
    si = np.where(sigma_ind == np.around(sigma))[0].item() + 1
    sigma_ind = sigma_ind[(max(1,si - 8) - 1):(min(si+8,sigma_ind.shape[0]))]
    score_vec = np.zeros(sigma_ind.shape[0])
    for i in range(sigma_ind.shape[0]):
        sigma = sigma_ind[i]
#         set_likelihood_vars(Q_mat_all[str(sigma)],X_vals)
        likelihood_vars_sigma = {'Q_mat': Q_mat_all[str(sigma)], 'X_vals': X_vals, 'N_X': Q_mat_all[str(sigma)].shape[1], 'K_val': Q_mat_all[str(sigma)].shape[0] - 3}
        best_val = calc_log_l_vec(X*mult_fac_vec[0], Y, likelihood_vars = likelihood_vars_sigma)
        for mult_fac in mult_fac_vec[1:]:
            best_val = min(best_val, calc_log_l_vec(X*mult_fac, Y, likelihood_vars = likelihood_vars_sigma))
        score_vec[i] = best_val
    sigma = sigma_ind[np.argmin(score_vec)]
    return sigma


def get_der_fast(S, B, S_mat, gene_list, prediction, bulk_mode = False, likelihood_vars = None):
    if bulk_mode:
        d1_vec = (np.log(prediction) - np.log(B)) / prediction * (-2)
        d2_vec = ((1 - np.log(prediction) + np.log(B))/prediction**2) * (-2)
    else:
        d1_d2 = get_d1_d2(B, prediction, likelihood_vars = likelihood_vars)
        d1_vec = d1_d2['d1_vec']
        d2_vec = d1_d2['d2_vec']
    
    grad = -d1_vec @ S
    hess = (-d2_vec[:,None,None] * S_mat).sum(0)
    return {'grad': grad, 'hess': hess}

def psd(H):
    eig = np.linalg.eig(H)
    epsilon = 1e-3
    if H.shape[0] == 1:
        P = eig[1] @ np.maximum(eig[0], epsilon) @ eig[1].T
#         P = eig[1] @ np.clip(eig[0], a_min = epsilon, a_max = eig[0].max() + 10) @ eig[1].T
    else:
        P = eig[1] @ np.diag(np.maximum(eig[0], epsilon)) @ eig[1].T
#         P = eig[1] @ np.diag(np.clip(eig[0], a_min = epsilon, a_max = eig[0].max() + 10)) @ eig[1].T
    return P

def solveWLS(S,B,S_mat,initialSol, nUMI, bulk_mode = False, constrain = False, likelihood_vars = None, solver = 'osqp'):
    solution = initialSol.copy()
    solution[solution < 0] = 0
    prediction = np.absolute(S @ solution)
    threshold = max(1e-4, nUMI * 1e-7)
    prediction[prediction < threshold] = threshold
    gene_list = np.array(S.index)
    derivatives = get_der_fast(S, B, S_mat, gene_list, prediction, bulk_mode = bulk_mode, likelihood_vars = likelihood_vars)
    d_vec = -derivatives['grad']
    D_mat = psd(derivatives['hess'])
    norm_factor = np.linalg.norm(D_mat, ord = 2)
    D_mat = D_mat / norm_factor
    d_vec = d_vec / norm_factor
    epsilon = 1e-7
    D_mat = D_mat + epsilon * np.identity(d_vec.shape[0])
    A = np.identity(S.shape[1])
    bzero = -solution
    alpha = 0.3
    if constrain:
        solution = solution + alpha*solve_qp(np.array(D_mat),-np.array(d_vec),-np.array(A),-np.array(bzero), np.ones(solution.shape[0]), 1 - solution.sum(), solver=solver)
    else:
        solution = solution + alpha*solve_qp(np.array(D_mat),-np.array(d_vec),-np.array(A),-np.array(bzero), solver=solver)
    return solution

def solveIRWLS_weights(S,B,nUMI, OLS=False, constrain = True, verbose = False,
                              n_iter = 50, MIN_CHANGE = .001, bulk_mode = False, solution = None, loggings = None, likelihood_vars = None, solver = 'osqp'):
    if not bulk_mode:
        K_val = likelihood_vars['K_val']
        B = np.copy(B)
        B[B > K_val] = K_val
    solution = np.ones(S.shape[1]) / S.shape[1]
    S_mat = np.einsum('ij, ik -> ijk', S, S)
    iterations = 0
    change = 1
    changes = []
    while (change > MIN_CHANGE) & (iterations < n_iter):
        new_solution = solveWLS(S,B,S_mat,solution, nUMI,constrain=constrain, bulk_mode = bulk_mode, likelihood_vars = likelihood_vars, solver = solver)
        change = np.linalg.norm(new_solution-solution, 1)
        if verbose:
            loggings.info('Change: {}'.format(change))
            loggings.info(solution)
        solution = new_solution
        iterations += 1
    return {'weights': pd.DataFrame(solution, index = S.columns), 'converged': (change <= MIN_CHANGE)}

@ray.remote
def decompose_full_ray(args):
    cell_type_profiles, nUMI, bead, constrain, OLS, MIN_CHANGE, likelihood_vars, loggings = args
    bulk_mode = False
    verbose = False
    n_iter = 50
    try:
        results = solveIRWLS_weights(cell_type_profiles,bead,nUMI,OLS = OLS, constrain = constrain,
                                   verbose = verbose, n_iter = n_iter, MIN_CHANGE = MIN_CHANGE, bulk_mode = bulk_mode, loggings = loggings, likelihood_vars = likelihood_vars, solver = 'osqp')
    except:
        results = solveIRWLS_weights(cell_type_profiles,bead,nUMI,OLS = OLS, constrain = constrain,
                                   verbose = verbose, n_iter = n_iter, MIN_CHANGE = MIN_CHANGE, bulk_mode = bulk_mode, loggings = loggings, likelihood_vars = likelihood_vars, solver = 'cvxopt')
    return results


def decompose_full(cell_type_profiles, nUMI, bead, constrain = True, OLS = False, MIN_CHANGE = 0.001, likelihood_vars = None, loggings = None, bulk_mode = False, verbose = False, n_iter = 50):
    results = solveIRWLS_weights(cell_type_profiles,bead,nUMI,OLS = OLS, constrain = constrain,
                               verbose = verbose, n_iter = n_iter, MIN_CHANGE = MIN_CHANGE, bulk_mode = bulk_mode, loggings = loggings, likelihood_vars = likelihood_vars)
    return results


def decompose_batch(nUMI, cell_type_means, beads, gene_list, constrain = True, OLS = False, max_cores = 22, MIN_CHANGE = 0.001, likelihood_vars = None, loggings = None):
    inp_args = []
    weights = []
    for i in range(beads.shape[0]):
        K_val = likelihood_vars['K_val']
        bead = beads[i,:]
        bead[bead > K_val] = K_val
        inp_args.append((cell_type_means.loc[gene_list,]*nUMI[i], nUMI[i], bead, constrain, OLS, MIN_CHANGE, likelihood_vars, loggings))
    weights = ray.get([decompose_full_ray.remote(arg) for arg in inp_args])
    return [_['weights'].values for _ in weights]


def get_norm_ref(puck, cell_type_means, gene_list, proportions):
    bulk_vec = puck['counts'].sum(1)
    weight_avg = (cell_type_means.loc[gene_list, :] * (proportions / proportions.sum()).values.squeeze()).sum(1)
    target_means = bulk_vec.loc[gene_list] / puck['nUMI'].sum().item()
    cell_type_means_renorm = cell_type_means.loc[gene_list,:] / (weight_avg / target_means).values.squeeze()[:, None]
    return cell_type_means_renorm

def fitBulk(RCTD, loggings = None):
    bulkData = prepareBulkData(RCTD['cell_type_info']['info']['cell_type_means'], RCTD['spatialRNA'], RCTD['internal_vars']['gene_list_bulk'])
    loggings.info('fitBulk: decomposing bulk')
    decompose_results = decompose_full(bulkData['X'],RCTD['spatialRNA']['nUMI'].sum().item(),
                                      np.array(bulkData['b']), verbose = False, constrain = False, MIN_CHANGE = RCTD['config']['MIN_CHANGE_BULK'],
                                      n_iter = 100, bulk_mode = True, loggings = loggings)
    RCTD['internal_vars']['proportions'] = decompose_results['weights']
    RCTD['cell_type_info']['renorm'] = RCTD['cell_type_info']['info'].copy()
    RCTD['cell_type_info']['renorm']['cell_type_means'] = get_norm_ref(RCTD['spatialRNA'], RCTD['cell_type_info']['info']['cell_type_means'], RCTD['internal_vars']['gene_list_bulk'], decompose_results['weights'])
    return RCTD


def SpatialRNA(coords, counts, nUMI = None):
    barcodes = list(set(coords.index) & set(counts.columns) & set(nUMI.index))
    if len(barcodes) == 0:
        raise ValueError('SpatialRNA: coords, counts, and nUMI do not share any barcode names. Please ensure that rownames(coords) matches colnames(counts) and names(nUMI)')
    if len(barcodes) < max(coords.shape[0], counts.shape[1], nUMI.shape[0]):
        raise ValueError('SpatialRNA: some barcodes in nUMI, coords, or counts were not mutually shared. Such barcodes were removed.')
    spatialrna_dict = {}
    spatialrna_dict['coords'] = coords
    spatialrna_dict['counts'] = counts
    spatialrna_dict['nUMI'] = nUMI
    return spatialrna_dict

def Reference(counts, cell_types, nUMI = None, n_max_cells = 10000, loggings=None):
    reference_dict = {}
    reference_dict['cell_types'] = cell_types
    reference_dict['counts'] = counts
    reference_dict['nUMI'] = nUMI
    cur_count = reference_dict['cell_types'].value_counts().max()
    if cur_count > n_max_cells:
        loggings.info('Reference: number of cells per cell type is {}, larger than maximum allowable of {}. Downsampling number of cells to: {}.'.format(cur_count, n_max_cells, n_max_cells))
    reference = create_downsampled_data(reference_dict, n_samples = n_max_cells)
    return reference
            
    
    
def create_downsampled_data(reference, cell_types_keep = None, n_samples = 10000):
    cell_types_list = np.array(reference['cell_types'].iloc[:,0])
    index_keep = []
    if cell_types_keep is None:
        cell_types_keep = np.unique(cell_types_list).tolist()
    for i in range(len(cell_types_keep)):
        new_index = cell_types_list == cell_types_keep[i]
        new_index = np.where(new_index == True)[0]
        new_samples = min(n_samples, new_index.shape[0])
        choose_index = np.random.choice(new_index, new_samples, replace = False)
#         choose_index = np.arange(new_index.shape[0])
#         random.shuffle(choose_index)
#         index_keep.append(new_index[choose_index[:new_samples]])
        index_keep.append(choose_index)
    index_keep = np.concatenate(index_keep, axis = -1)
    reference['counts'] = reference['counts'].iloc[:,index_keep]
    reference['cell_types'] = reference['cell_types'].iloc[index_keep,:]
    reference['nUMI'] = reference['nUMI'].iloc[index_keep,:]
    return reference
                   
    
def get_cell_type_info(counts, cell_types, nUMI, cell_type_names = None):
    if cell_type_names is None:
        cell_type_names = np.unique(cell_types.iloc[:,0])
    n_cell_types = cell_type_names.shape[0]
    
    def get_cell_mean(cell_type):
        index = np.array(cell_types.iloc[:,0]) == cell_type
        normData = counts.loc[:, index].values / nUMI[index].values.squeeze()[None,:]
        return normData.mean(1).squeeze()
    
    cell_type_means = pd.DataFrame()
    for cell_type in cell_type_names:
        cell_type_means[cell_type] = get_cell_mean(cell_type)
    cell_type_means.index = counts.index
    cell_type_means.columns = cell_type_names
    ret = {'cell_type_means': cell_type_means, 'cell_type_names': cell_type_names, 'n_cell_types': n_cell_types}
    return ret


def process_cell_type_info(reference, cell_type_names, CELL_MIN = 25, loggings = None):
    loggings.info("Begin: process_cell_type_info")
    loggings.info("process_cell_type_info: number of cells in reference: {}".format(reference['counts'].shape[1]))
    loggings.info("process_cell_type_info: number of genes in reference: {}".format(reference['counts'].shape[0]))
    cell_counts = reference['cell_types'].value_counts()
    loggings.info(cell_counts)
    if reference['cell_types'].value_counts().min() < CELL_MIN:
        loggings.info("process_cell_type_info error: need a minimum of {} cells for each cell type in the reference".format(CELL_MIN))
    cell_type_info = get_cell_type_info(reference['counts'], reference['cell_types'], reference['nUMI'], cell_type_names = cell_type_names)
    loggings.info("End: process_cell_type_info")
    return cell_type_info

def restrict_counts(puck, gene_list, UMI_thresh = 1, UMI_max = 20000):
    keep_loc = (puck['nUMI'] >= UMI_thresh) & (puck['nUMI'] <= UMI_max)
    puck['counts'] = puck['counts'].loc[gene_list, np.array(keep_loc)]
    puck['nUMI'] = puck['nUMI'][np.array(keep_loc)]
    return puck


def get_de_genes(cell_type_info, puck, fc_thresh = 1.25, expr_thresh = .00015, MIN_OBS = 3, loggings = None):
    total_gene_list = []
    epsilon = 1e-9
#     bulk_vec = pd.DataFrame(puck['counts'].sum(1))
    bulk_vec = puck['counts'].sum(1)
    gene_list = np.array(cell_type_info['cell_type_means'].index)
    index = np.array([gene.startswith('mt-') for gene in gene_list])
    if gene_list[index].shape[0] > 0:
        gene_list = gene_list[~index]
    gene_list = np.array(list(set(gene_list) & set(np.array(bulk_vec.index))))
    if gene_list.shape[0] == 0:
        raise ValueError("get_de_genes: Error: 0 common genes between SpatialRNA and Reference objects. Please check for gene list nonempty intersection.")
#     gene_list = gene_list[np.array(bulk_vec.loc[gene_list] >= MIN_OBS).squeeze()]
    gene_list = gene_list[bulk_vec[gene_list] >= MIN_OBS]
    for cell_type in cell_type_info['cell_type_names']:
        other_mean = cell_type_info['cell_type_means'].loc[gene_list, cell_type_info['cell_type_names'] != cell_type].mean(1)
        logFC = np.log(cell_type_info['cell_type_means'].loc[gene_list,cell_type] + epsilon) - np.log(other_mean + epsilon)
        type_gene_list = np.where(((logFC > fc_thresh) & (cell_type_info['cell_type_means'].loc[gene_list,cell_type] > expr_thresh)) == True)[0]
        loggings.info("get_de_genes: {} found DE genes: {}".format(cell_type, type_gene_list.shape[0]))
        total_gene_list.append(type_gene_list)
    total_gene_list = np.concatenate(total_gene_list, axis = -1)
    gene_list = np.unique(gene_list[total_gene_list])
    loggings.info("get_de_genes: total DE genes: {}".format(gene_list.shape[0]))
    return gene_list

def restrict_puck(puck, barcodes):
    puck['counts'] =  puck['counts'].loc[:, barcodes]
    puck['nUMI'] =  puck['nUMI'].loc[barcodes]
    puck['coords'] =  puck['coords'].loc[barcodes,:]
    return puck

def create_RCTD(spatialRNA, reference, max_cores = 4, test_mode = False, gene_cutoff = 0.000125, fc_cutoff = 0.5, gene_cutoff_reg = 0.0002, fc_cutoff_reg = 0.75, UMI_min = 100, UMI_max = 20000000, UMI_min_sigma = 100, MIN_OBS = 3,
                         class_df = None, CELL_MIN_INSTANCE = 25, cell_type_names = None, MAX_MULTI_TYPES = 4, keep_reference = False, cell_type_info = None, loggings = None):
    config = {'gene_cutoff': gene_cutoff, 'fc_cutoff': fc_cutoff, 'gene_cutoff_reg': gene_cutoff_reg, 'fc_cutoff_reg': fc_cutoff_reg, 'UMI_min': UMI_min, 'UMI_min_sigma': UMI_min_sigma, 'max_cores': max_cores,
                 'N_epoch': 8, 'N_X': 50000, 'K_val': 100, 'N_fit': 1000, 'N_epoch_bulk' :30, 'MIN_CHANGE_BULK': 0.0001, 'MIN_CHANGE_REG': 0.001, 'UMI_max': UMI_max, 'MIN_OBS': MIN_OBS, 'MAX_MULTI_TYPES': MAX_MULTI_TYPES}
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
    loggings.info('create.RCTD: getting regression differentially expressed genes: ')
    gene_list_reg = get_de_genes(cell_type_info['info'], puck_original, fc_thresh = config['fc_cutoff_reg'], expr_thresh = config['gene_cutoff_reg'], MIN_OBS = config['MIN_OBS'], loggings = loggings)
    if gene_list_reg.shape[0] == 0:
        raise ValueError("create.RCTD: Error: 0 regression differentially expressed genes found")
    loggings.info('create.RCTD: getting platform effect normalization differentially expressed genes: ')
    gene_list_bulk = get_de_genes(cell_type_info['info'], puck_original, fc_thresh = config['fc_cutoff'], expr_thresh = config['gene_cutoff'], MIN_OBS = config['MIN_OBS'], loggings = loggings)
    if gene_list_bulk.shape[0] == 0:
        raise ValueError("create.RCTD: Error: 0 bulk differentially expressed genes found")
    puck = restrict_counts(puck_original, gene_list_bulk, UMI_thresh = config['UMI_min'], UMI_max = config['UMI_max'])
    puck = restrict_puck(puck, puck['counts'].columns)
    if class_df is None:
        class_df = pd.DataFrame(cell_type_info['info']['cell_type_names'], index = cell_type_info['info']['cell_type_names'], columns = ['class'])
    internal_vars = {'gene_list_reg': gene_list_reg, 'gene_list_bulk': gene_list_bulk, 'proportions': None, 'class_df': class_df, 'cell_types_assigned': False}
    RCTD = {'spatialRNA': puck, 'originalSpatialRNA': puck_original, 'reference': reference, 'config': config, 'cell_type_info': cell_type_info, 'internal_vars': internal_vars}
    return RCTD




def prepareBulkData(cell_type_means, puck, gene_list, MIN_OBS = 10):
    bulk_vec = puck['counts'].sum(1)
    gene_list = np.array(list(set(bulk_vec.index[bulk_vec >= MIN_OBS]) & set(gene_list)))
    nUMI = puck['nUMI'].sum().item()
    X = cell_type_means.loc[gene_list,:] * nUMI
    b = bulk_vec[gene_list]
    ret = {'X': X, 'b': b}
    return ret


def choose_sigma_c(RCTD, Q_mat_all, X_vals_loc,loggings = None):
    puck = RCTD['spatialRNA']
    MIN_UMI = RCTD['config']['UMI_min_sigma']
    sigma = 100
    sigma_vals = Q_mat_all.keys()
    N_fit = min(RCTD['config']['N_fit'],(puck['nUMI'] > MIN_UMI).sum().item())
    if N_fit == 0:
        raise ValueError('choose_sigma_c determined a N_fit of 0! This is probably due to unusually low UMI counts per bead in your dataset. Try decreasing the parameter UMI_min_sigma. It currently is {} but none of the beads had counts larger than that.'.format(MIN_UMI))
    fit_ind = np.random.choice(puck['nUMI'][puck['nUMI'] > MIN_UMI].index, N_fit, replace = False)
    beads = puck['counts'].loc[RCTD['internal_vars']['gene_list_reg'],fit_ind].values.T
    loggings.info('chooseSigma: using initial Q_mat with sigma = {}'.format(sigma/100))
    likelihood_vars = {'Q_mat': Q_mat_all[str(sigma)], 'X_vals': X_vals_loc, 'N_X': Q_mat_all[str(sigma)].shape[1], 'K_val': Q_mat_all[str(sigma)].shape[0] - 3}
    for _ in np.arange(RCTD['config']['N_epoch']):
#         set_likelihood_vars(Q_mat_all[str(sigma)], X_vals_loc, )
        likelihood_vars['Q_mat'] = Q_mat_all[str(sigma)]
        results = decompose_batch(np.array(puck['nUMI'].loc[fit_ind]).squeeze(), RCTD['cell_type_info']['renorm']['cell_type_means'], beads, RCTD['internal_vars']['gene_list_reg'], constrain = False, max_cores = RCTD['config']['max_cores'], loggings = loggings,likelihood_vars = likelihood_vars)
        weights = np.zeros((N_fit, RCTD['cell_type_info']['renorm']['n_cell_types']))
        for i in range(N_fit):
            weights[i] = results[i].squeeze()

        prediction = RCTD['cell_type_info']['renorm']['cell_type_means'].loc[RCTD['internal_vars']['gene_list_reg'],:] @ weights.T * (puck['nUMI'].loc[fit_ind]).values.squeeze()[None,:]
        print('Likelihood value: {}'.format(calc_log_l_vec(prediction.values.T.reshape(-1), beads.reshape(-1),likelihood_vars = likelihood_vars)))
        sigma_prev = sigma
        sigma = chooseSigma(prediction, beads.T, Q_mat_all, likelihood_vars['X_vals'], sigma)
        loggings.info('Sigma value: {}'.format(sigma/100))
        if sigma == sigma_prev:
            break
            
            
    RCTD['internal_vars']['sigma'] = sigma/100
    RCTD['internal_vars']['Q_mat'] = Q_mat_all[str(sigma)]
    RCTD['internal_vars']['X_vals'] = likelihood_vars['X_vals']
    return(RCTD) 

# def fitPixels(RCTD, loggings = None):
#     RCTD['internal_vars']['cell_types_assigned'] = True
#     likelihood_vars = {'Q_mat': RCTD['internal_vars']['Q_mat'], 'X_vals': RCTD['internal_vars']['X_vals'], 'N_X': RCTD['internal_vars']['Q_mat'].shape[1], 'K_val': RCTD['internal_vars']['Q_mat'].shape[0] - 3}
#     cell_type_info = RCTD['cell_type_info']['renorm']
#     beads = RCTD['spatialRNA']['counts'].loc[RCTD['internal_vars']['gene_list_reg'],:].values.T
#     results = decompose_batch(np.array(RCTD['spatialRNA']['nUMI']).squeeze(), cell_type_info['cell_type_means'], beads, RCTD['internal_vars']['gene_list_reg'], constrain = False,
#                                   max_cores = RCTD['config']['max_cores'], MIN_CHANGE = RCTD['config']['MIN_CHANGE_REG'], loggings = loggings, likelihood_vars = likelihood_vars)
#     weights = np.zeros((len(results), RCTD['cell_type_info']['renorm']['n_cell_types']))
#     for i in range(weights.shape[0]):
#         weights[i] = results[i].squeeze()

#     RCTD['results'] = pd.DataFrame(weights, index = RCTD['spatialRNA']['counts'].columns, columns = RCTD['cell_type_info']['renorm']['cell_type_names'])
#     return RCTD


def fitPixels(RCTD, doublet_mode = 'full', loggings = None):
    RCTD['internal_vars']['cell_types_assigned'] = True
    likelihood_vars = {'Q_mat': RCTD['internal_vars']['Q_mat'], 'X_vals': RCTD['internal_vars']['X_vals'], 'N_X': RCTD['internal_vars']['Q_mat'].shape[1], 'K_val': RCTD['internal_vars']['Q_mat'].shape[0] - 3}
    cell_type_info = RCTD['cell_type_info']['renorm']
    
    if doublet_mode == 'doublet':
        results = process_beads_batch(cell_type_info, RCTD['internal_vars']['gene_list_reg'], RCTD['spatialRNA'], class_df = RCTD['internal_vars']['class_df'],
                                  constrain = False, MAX_CORES = RCTD['config']['max_cores'], MIN_CHANGE = RCTD['config']['MIN_CHANGE_REG'], loggings = loggings, likelihood_vars = likelihood_vars)
        return(gather_results(RCTD, results, loggings = loggings))
    elif doublet_mode == 'full':
        beads = RCTD['spatialRNA']['counts'].loc[RCTD['internal_vars']['gene_list_reg'],:].values.T
        results = decompose_batch(np.array(RCTD['spatialRNA']['nUMI']).squeeze(), cell_type_info['cell_type_means'], beads, RCTD['internal_vars']['gene_list_reg'], constrain = False,
                                      max_cores = RCTD['config']['max_cores'], MIN_CHANGE = RCTD['config']['MIN_CHANGE_REG'], loggings = loggings, likelihood_vars = likelihood_vars)
        weights = np.zeros((len(results), RCTD['cell_type_info']['renorm']['n_cell_types']))
        for i in range(weights.shape[0]):
            weights[i] = results[i].squeeze()

        RCTD['results'] = pd.DataFrame(weights, index = RCTD['spatialRNA']['counts'].columns, columns = RCTD['cell_type_info']['renorm']['cell_type_names'])
        return RCTD
    
    
    
    
def gather_results(RCTD, results, loggings = None):
    cell_type_names = RCTD['cell_type_info']['renorm']['cell_type_names']
    barcodes = np.array(RCTD['spatialRNA']['counts'].columns)
    N = len(results)
    weights = np.zeros((N, cell_type_names.shape[0]))
    weights_doublet = np.zeros((N,2))
    spot_class = []
    first_type = []
    second_type = []
    first_class = []
    second_class = []
    min_score = []
    singlet_score = []
    conv_all = []
    conv_doublet = []
    score_mat = []
    for i in range(N):
        if ((i + 1) % 1000) == 0:
            loggings.info("gather_results: finished: {}".format(i))
        weights_doublet[i] = np.array(results[i]['doublet_weights']).squeeze()
        weights[i] = np.array(results[i]['all_weights']).squeeze()
        spot_class.append(results[i]['spot_class'])
        first_type.append(results[i]['first_type'])
        second_type.append(results[i]['second_type'])
        first_class.append(results[i]['first_class'])
        second_class.append(results[i]['second_class'])
        min_score.append(results[i]['min_score'])
        singlet_score.append(results[i]['singlet_score'])
        conv_all.append(results[i]['conv_all'])
        conv_doublet.append(results[i]['conv_doublet'])
        score_mat.append(results[i]['score_mat'])
    weights = pd.DataFrame(weights, index = barcodes, columns = cell_type_names)
    weights_doublet = pd.DataFrame(weights_doublet, index = barcodes, columns = np.array(['first_type', 'second_type']))
    results_df = pd.DataFrame()
    results_df["spot_class"] = spot_class
    results_df["first_type"] = first_type
    results_df["second_type"] = second_type
    results_df["first_class"] = first_class
    results_df["second_class"] = second_class
    results_df["min_score"] = min_score
    results_df["singlet_score"] = singlet_score
    results_df["conv_all"] = conv_all
    results_df["conv_doublet"] = conv_doublet
    results_df.index = barcodes
    RCTD['results'] = {'results_df': results_df, 'weights': weights, 'weights_doublet': weights_doublet, 'score_mat': score_mat}
    return RCTD

# def process_beads_batch(cell_type_info, gene_list, puck, class_df = None, constrain = T, MAX_CORES = 8, MIN_CHANGE = 0.001, likelihood_vars = None, loggings = None):
#     beads = puck['counts'].loc[gene_list,:].T
#     results = []
#     for i in range(beads.shape[0]):
#         results.append(process_bead_doublet(cell_type_info, gene_list, puck['nUMI'].iloc[i].item(), beads.iloc[i,:], class_df = class_df, constrain = constrain, MIN_CHANGE = MIN_CHANGE, likelihood_vars = likelihood_vars, loggings = loggings))
        
#     return results

def process_beads_batch(cell_type_info, gene_list, puck, class_df = None, constrain = True, MAX_CORES = 8, MIN_CHANGE = 0.001, likelihood_vars = None, loggings = None):
    beads = puck['counts'].loc[gene_list,:].T
    
    inp_args = []
    
    for i in range(beads.shape[0]):
        inp_args.append((cell_type_info, gene_list, puck['nUMI'].iloc[i].item(), beads.iloc[i,:], class_df, constrain, False, MIN_CHANGE, loggings, likelihood_vars, i))
    
    results = ray.get([process_bead_doublet.remote(arg) for arg in inp_args])
    return results 
    
    
@ray.remote
def process_bead_doublet(args):
    cell_type_info, gene_list, UMI_tot, bead, class_df, constrain, verbose, MIN_CHANGE, loggings, likelihood_vars, i = args
    loggings.info("proceed bead {}".format(i))
    cell_type_profiles = cell_type_info['cell_type_means'].loc[gene_list,:]
    cell_type_profiles = cell_type_profiles * UMI_tot
    # cell_type_profiles = data.matrix(cell_type_profiles)
    
    QL_score_cutoff = 10
    doublet_like_cutoff = 25
    results_all = decompose_full(cell_type_profiles, UMI_tot, bead, constrain = constrain, verbose = verbose, MIN_CHANGE = MIN_CHANGE, likelihood_vars = likelihood_vars, loggings = loggings)
    all_weights = results_all['weights']
    conv_all = results_all['converged']
    initial_weight_thresh = 0.01
    cell_type_names = cell_type_info['cell_type_names']
    candidates = np.array(all_weights[np.array(all_weights > initial_weight_thresh)].index)

    if candidates.shape[0] == 0:
        candidates = cell_type_info['cell_type_names'][1:min(3,cell_type_info['n_cell_types'])]
    elif candidates.shape[0] == 1:
        if candidates[0] == cell_type_info['cell_type_names'][0]:
            candidates = np.concatenate((candidates, np.array([cell_type_info['cell_type_names'][1]])))
        else:
            candidates = np.concatenate((candidates, np.array([cell_type_info['cell_type_names'][0]])))
            
    score_mat = np.zeros((candidates.shape[0], candidates.shape[0]))
    min_score = 0
    first_type = None
    second_type = None
    first_class = False
    second_class = False
    for i in range(candidates.shape[0] - 1):
        type1 = candidates[i]
        for j in np.arange(i + 1, candidates.shape[0]):
            type2 = candidates[j]
            score = decompose_sparse(cell_type_profiles, UMI_tot, bead, type1, type2, score_mode = True, constrain = constrain, verbose = verbose, MIN_CHANGE = MIN_CHANGE, loggings = loggings, likelihood_vars = likelihood_vars)
            score_mat[i,j] = score
            score_mat[j,i] = score
            if (second_type == None) or (score < min_score):
                first_type = type1
                second_type = type2
                min_score = score
    score_mat = pd.DataFrame(score_mat, index = candidates, columns = candidates)
    
    type1_pres = check_pairs_type(cell_type_profiles, bead, UMI_tot, score_mat, min_score, first_type, class_df, QL_score_cutoff, constrain, MIN_CHANGE = MIN_CHANGE, loggings = loggings, likelihood_vars = likelihood_vars)
    type2_pres = check_pairs_type(cell_type_profiles, bead, UMI_tot, score_mat, min_score, second_type, class_df, QL_score_cutoff, constrain, MIN_CHANGE = MIN_CHANGE, loggings = loggings, likelihood_vars = likelihood_vars)
    
    if not type1_pres['all_pairs_class'] and not type2_pres['all_pairs_class']:
        spot_class = "reject"
        singlet_score = min_score + 2 * doublet_like_cutoff #arbitrary
    elif type1_pres['all_pairs_class'] and not type2_pres['all_pairs_class']:
        first_class = not type1_pres['all_pairs']
        singlet_score = type1_pres['singlet_score']
        spot_class = "doublet_uncertain"
    elif not type1_pres['all_pairs_class'] and type2_pres['all_pairs_class']:
        first_class = not type2_pres['all_pairs']
        singlet_score = type2_pres['singlet_score']
        temp = first_type
        first_type = second_type
        second_type = temp
        spot_class = "doublet_uncertain"
    else:
        spot_class = "doublet_certain"
        singlet_score = min(type1_pres['singlet_score'], type2_pres['singlet_score'])
        first_class = not type1_pres['all_pairs']
        second_class = not type2_pres['all_pairs']
        if type2_pres['singlet_score'] < type1_pres['singlet_score']:
            temp = first_type
            first_type = second_type
            second_type = temp
            first_class = not type2_pres['all_pairs']
            second_class = not type1_pres['all_pairs']
    if ((singlet_score - min_score) < doublet_like_cutoff):
        spot_class = "singlet"
    doublet_results = decompose_sparse(cell_type_profiles, UMI_tot, bead, first_type, second_type, constrain = constrain, MIN_CHANGE = MIN_CHANGE, loggings = loggings, likelihood_vars = likelihood_vars)
    doublet_weights = doublet_results['weights']
    conv_doublet = doublet_results['converged']
    # spot_class = factor(spot_class, c("reject", "singlet", "doublet_certain", "doublet_uncertain"))

    return {'all_weights': all_weights, 'spot_class': spot_class, 'first_type': first_type, 'second_type': second_type,
              'doublet_weights': doublet_weights, 'min_score': min_score, 'singlet_score': singlet_score,
              'conv_all': conv_all, 'conv_doublet': conv_doublet, 'score_mat': score_mat,
              'first_class': first_class, 'second_class': second_class}

def decompose_sparse(cell_type_profiles, nUMI, bead, type1=None, type2=None, score_mode = False, plot = False, custom_list = None, verbose=False, constrain = True, MIN_CHANGE = 0.001, loggings = None, likelihood_vars = None):
    if custom_list is None:
        cell_types = np.array([type1, type2])
    else:
        cell_types = custom_list
    if type2 == None:
        cell_types = np.array([type1])
    reg_data = cell_type_profiles.loc[:,cell_types]
    if score_mode:
        n_iter = 25
    else:
        n_iter = 50
    results = solveIRWLS_weights(reg_data,bead,nUMI,OLS = False, constrain = constrain, verbose = verbose, n_iter = n_iter, MIN_CHANGE = MIN_CHANGE, loggings = loggings, likelihood_vars = likelihood_vars)
    if not score_mode:
        results['weights'] = results['weights'] / results['weights'].sum()
        return results
    else:
        prediction = (reg_data @ results['weights']).values.squeeze()
        total_score = calc_log_l_vec(prediction, np.array(bead).squeeze(),likelihood_vars = likelihood_vars)
        return total_score
    
    
def get_singlet_score(cell_type_profiles, bead, UMI_tot, ty, constrain, MIN_CHANGE = 0.001, return_vec = False, loggings = None, likelihood_vars = None):
    if not constrain:
        return decompose_sparse(cell_type_profiles, UMI_tot, bead, type1=ty, score_mode = True, constrain = constrain, MIN_CHANGE = MIN_CHANGE, loggings = loggings, likelihood_vars = likelihood_vars)
    reg_data = cell_type_profiles.loc[:,ty]
    prediction = reg_data.values
    log_l = calc_log_l_vec(prediction, np.array(bead).squeeze(), return_vec = return_vec, likelihood_vars = likelihood_vars)
    return log_l


def check_pairs_type(cell_type_profiles, bead, UMI_tot, score_mat, min_score, my_type, class_df, QL_score_cutoff, constrain, MIN_CHANGE = 0.001, loggings = None, likelihood_vars = None):
    candidates = np.array(score_mat.index)
    singlet_score = get_singlet_score(cell_type_profiles, bead, UMI_tot, my_type, constrain, MIN_CHANGE = MIN_CHANGE, loggings = loggings, likelihood_vars = likelihood_vars)
    all_pairs = True
    all_pairs_class = class_df is not None
    other_class = [my_type] #other types present from this class
    
    for i in range(candidates.shape[0] - 1):
        type1 = candidates[i]
        for j in np.arange(i + 1, candidates.shape[0]):
            type2 = candidates[j]
            if (score_mat.values[i,j] < (min_score + QL_score_cutoff)):
                if (type1 != my_type) and (type2 != my_type):
                    all_pairs = False
                if class_df is not None:
                    first_class = class_df.loc[my_type,"class"] == class_df.loc[type1,"class"]
                    second_class = class_df.loc[my_type,"class"] == class_df.loc[type2,"class"]
                    if not first_class and not second_class:
                        all_pairs_class = False
                    if first_class and not type1 in other_class:
                        other_class = other_class + [type1]
                    if second_class and not type2 in other_class:
                        other_class = other_class + [type2]
        if class_df is None:
            all_pairs_class = all_class
        if all_pairs_class and not all_pairs and (len(other_class) > 1):
            for ty in other_class[2:len(other_class)]:
                singlet_score = min(singlet_score, get_singlet_score(cell_type_profiles, bead, UMI_tot, ty, constrain, MIN_CHANGE = MIN_CHANGE, loggings = loggings, likelihood_vars = likelihood_vars))
    return {'all_pairs': all_pairs, 'all_pairs_class': all_pairs_class, 'singlet_score': singlet_score}
