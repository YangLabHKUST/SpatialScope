import scanpy as sc
import anndata
import pandas as pd
import numpy as np
import os
import time
import logging
#from multiprocessing import Pool
import itertools
import json
import pickle
import gzip
# from qpsolvers.solvers.qpswift_ import qpswift_solve_qp

data_type = 'float32'


import matplotlib as mpl
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils import *
from utils_CTI import *
num_cpus = psutil.cpu_count(logical=False) 
os.environ["PYTHONPATH"] = os.getcwd()+ ":" + os.environ.get("PYTHONPATH", "")
ray.shutdown()
ray.init(num_cpus=num_cpus,_temp_dir='/tmp/ray')


import copy
import argparse

##
# Singlet/Doublet classification was applied firstly for high resolution ST data such as Slideseq
##

class SpatialScopeSDC:
    def __init__(self,tissue,out_dir, ST_Data, SC_Data, cell_class_column = 'cell_type'):
        self.tissue = tissue
        self.out_dir = out_dir 
        self.ST_Data = ST_Data
        self.SC_Data = SC_Data
        self.cell_class_column = cell_class_column
        
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        if not os.path.exists(os.path.join(out_dir,tissue)):
            os.mkdir(os.path.join(out_dir,tissue))
        
        self.out_dir = os.path.join(out_dir,tissue)
        loggings = configure_logging(os.path.join(self.out_dir,'logs'))
        self.loggings = loggings 
        self.LoadData(self.ST_Data, self.SC_Data, cell_class_column = self.cell_class_column)
        self.InitProp = None
        
        
    def LoadData(self, ST_Data, SC_Data, cell_class_column = 'cell_type'):
        sp_adata = anndata.read_h5ad(ST_Data)
        sc_adata = anndata.read_h5ad(SC_Data)
        sp_adata.obs_names_make_unique()
        sp_adata.var_names_make_unique()
        sc_adata.obs_names_make_unique()
        sc_adata.var_names_make_unique()

        if 'Marker' in sc_adata.var.keys():
            sel_genes = sc_adata.var.index[sc_adata.var['Marker']]
            sp_adata = sp_adata[:,sp_adata.var.index.isin(sel_genes)]

        if sp_adata.X.max()<30:
            try:
                sp_adata.X = np.exp(sp_adata.X) - 1
            except:
                sp_adata.X = np.exp(sp_adata.X.toarray()) - 1
        
        if sc_adata.X.max()<30:
            try:
                sc_adata.X = np.exp(sc_adata.X) - 1
            except:
                sc_adata.X = np.exp(sc_adata.X.toarray()) - 1
            sc.pp.normalize_total(sc_adata, inplace=True)
            
        self.sp_adata = sp_adata
        self.sc_adata = sc_adata
        self.cell_class_column = cell_class_column
        
        
    def LoadLikelihoodTable(self):
        with gzip.open(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/extdata/Q_mat_1_1.txt.gz','rt') as f:
            lines = f.readlines()
        with gzip.open(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/extdata/Q_mat_1_2.txt.gz','rt') as f:
            lines += f.readlines()

        Q1 = {}
        for i in range(len(lines)):
            if i == 61:
                Q1[str(72)] = np.reshape(np.array(lines[i].split(' ')).astype(np.float64), (2536, 103)).T
            elif i == 62:
                Q1[str(74)] = np.reshape(np.array(lines[i].split(' ')).astype(np.float64), (2536, 103)).T
            else:
                Q1[str(i + 10)] = np.reshape(np.array(lines[i].split(' ')).astype(np.float64), (2536, 103)).T

        with gzip.open(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/extdata/Q_mat_2_1.txt.gz','rt') as f:
            lines2 = f.readlines()
        with gzip.open(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/extdata/Q_mat_2_2.txt.gz','rt') as f:
            lines2 += f.readlines()

        Q2 = {}
        for i in range(len(lines2)):
            Q2[str(int(i * 2 + 76))] = np.reshape(np.array(lines2[i].split(' ')).astype(np.float64), (2536, 103)).T

        Q_mat_all = Q1 | Q2

        with open(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/extdata/X_vals.txt') as f:
            lines_X = f.readlines()

        X_vals_loc = np.array([float(lines_X_item.strip()) for lines_X_item in lines_X])
        
        self.Q_mat_all = Q_mat_all
        self.X_vals_loc = X_vals_loc
        
    def SingletDoubletClassification(self):
        
        self.LoadLikelihoodTable()
        counts = self.sp_adata.to_df().T
        coords = self.sp_adata.obs[['x', 'y']]

        nUMI = pd.DataFrame(np.array(self.sp_adata.X.sum(-1)), index = self.sp_adata.obs.index)
        puck = SpatialRNA(coords, counts, nUMI)
        counts = self.sc_adata.to_df().T
        cell_types = pd.DataFrame(self.sc_adata.obs[self.cell_class_column])
        nUMI = pd.DataFrame(self.sc_adata.to_df().T.sum(0))
        reference = Reference(counts, cell_types, nUMI)
        myRCTD = create_RCTD(puck, reference, max_cores = 22, UMI_min = 20, loggings = self.loggings)
        myRCTD = run_RCTD(myRCTD, self.Q_mat_all, self.X_vals_loc, loggings = self.loggings, doublet_mode = 'doublet')
        self.InitProp = myRCTD
        import pickle
        with open(os.path.join(self.out_dir, 'InitProp.pickle'), 'wb') as handle:
            pickle.dump(self.InitProp, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        ray.shutdown()

        p2_res = self.InitProp['results']
        p2_res['weights_doublet'].columns = ['first_type_weight', 'second_type_weight']
        p2_res = p2_res['results_df'].merge(p2_res['weights_doublet'],left_index=True,right_index=True)
        p2_res = p2_res.drop(['first_class','second_class','conv_all','conv_doublet'],axis=1)
        p2_res = p2_res.merge(self.sp_adata.obs[['x','y']],left_index=True,right_index=True)
        p2_res_single = SlideseqP2Parser(p2_res)
        self.sp_adata.uns['cell_locations'] = p2_res_single
        
        self.sp_adata.write_h5ad(os.path.join(self.out_dir, 'sp_adata_sdc.h5ad'))


if __name__ == "__main__":
    HEADER = """
    <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    <> 
    <> SpatialScope Singlet/Doublet classification 
    <> Version: %s
    <> MIT License
    <>
    <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    <> Software-related correspondence: %s or %s
    <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    <> Visium data example:
        python <install path>/src/Singlet_Doublet_Classification.py \\
            --cell_class_column cell_type \\
            --tissue cere \\
            --out_dir ./output \\
            --ST_Data ./demo_data/slideseq-cerebellum.h5ad \\
            --SC_Data ./Ckpts_scRefs/Cerebellum/Ref_snRNA_cerebellum_qc.h5ad 
    <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>  
    """ 
    
    parser = argparse.ArgumentParser(description='simulation sour_sep')
    parser.add_argument('--tissue', type=str, help='tissue name', default=None)
    parser.add_argument('--out_dir', type=str, help='output path', default=None)
    parser.add_argument('--ST_Data', type=str, help='ST data path', default=None)
    parser.add_argument('--SC_Data', type=str, help='single cell reference data path', default=None)
    parser.add_argument('--cell_class_column', type=str, help='input cell class label column in scRef file', default = 'cell_type')    

    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    if not os.path.exists(os.path.join(args.out_dir,args.tissue)):
        os.mkdir(os.path.join(args.out_dir,args.tissue))


    SDC = SpatialScopeSDC(args.tissue,args.out_dir, args.ST_Data, args.SC_Data, cell_class_column = args.cell_class_column)
    SDC.SingletDoubletClassification()