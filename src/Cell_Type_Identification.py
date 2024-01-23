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
ray.init(num_cpus=num_cpus-2,_temp_dir='/tmp/ray')


import copy
import argparse


class SpatialScopeCTI:
    def __init__(self,tissue,out_dir, ST_Data, SC_Data, cell_class_column = 'cell_type', InitProp = None):
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
        self.InitProp = InitProp
        
        
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

        # cell_class_column = args.cell_class_column

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
        
        
    def WarmStart(self,hs_ST,UMI_min_sigma = 300):
        self.LoadLikelihoodTable()
        
        counts = self.sp_adata.to_df().T
        if hs_ST:
            UMI_min = 20
            if 'z' in self.sp_adata.obs.columns:
                coords = self.sp_adata.obs[['x', 'y', 'z']]
            else:
                coords = self.sp_adata.obs[['x', 'y']]
        else:
            UMI_min =100
            if self.sp_adata.obsm['spatial'].shape[1] == 2:
                coords = pd.DataFrame(self.sp_adata.obsm['spatial'], index = counts.columns, columns = ['x', 'y'])
                self.loggings.info('A single ST data with spatial location shape: {}'.format(self.sp_adata.obsm['spatial'].shape))
            elif self.sp_adata.obsm['spatial'].shape[1] == 3:
                coords = pd.DataFrame(self.sp_adata.obsm['spatial'], index = counts.columns, columns = ['x', 'y', 'z'])
                self.loggings.info('3D aligned ST data with spatial location shape: {}'.format(self.sp_adata.obsm['spatial'].shape))
            else:
                self.loggings.error('Wrong spatial location shape: {}'.format(self.sp_adata.obsm['spatial'].shape))
                sys.exit()
        UMI_min = min(UMI_min,UMI_min_sigma)
        nUMI = pd.DataFrame(np.array(self.sp_adata.X.sum(-1)), index = self.sp_adata.obs.index)
        puck = SpatialRNA(coords, counts, nUMI)
        counts = self.sc_adata.to_df().T
        cell_types = pd.DataFrame(self.sc_adata.obs[self.cell_class_column])
        nUMI = pd.DataFrame(self.sc_adata.to_df().T.sum(0))
        reference = Reference(counts, cell_types, nUMI, loggings=self.loggings)
        myRCTD = create_RCTD(puck, reference, max_cores = 22, UMI_min=UMI_min, UMI_min_sigma = UMI_min_sigma, loggings = self.loggings)
        myRCTD = run_RCTD(myRCTD, self.Q_mat_all, self.X_vals_loc, loggings = self.loggings)
        self.InitProp = myRCTD
        import pickle
        with open(os.path.join(self.out_dir, 'InitProp.pickle'), 'wb') as handle:
            pickle.dump(self.InitProp, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return self.InitProp
        
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
        
    def CellTypeIdentification(self, nu = 10, n_neighbo = 10, hs_ST = False, UMI_min_sigma = 300):
        cell_locations = self.sp_adata.uns['cell_locations'].copy()
        if self.InitProp is not None:
            self.WarmStart(hs_ST=hs_ST, UMI_min_sigma = UMI_min_sigma)
        elif not os.path.exists(os.path.join(self.out_dir, 'InitProp.pickle')):
            self.WarmStart(hs_ST=hs_ST, UMI_min_sigma = UMI_min_sigma)
        elif os.path.exists(os.path.join(self.out_dir, 'InitProp.pickle')):
            self.loggings.info('Loading existing InitProp, no need to warmstart')
            with open(os.path.join(self.out_dir, 'InitProp.pickle'), 'rb') as handle:
                self.InitProp = pickle.load(handle)
                self.LoadLikelihoodTable()
        cell_locations = cell_locations.loc[cell_locations['spot_index'].isin(self.InitProp['results'].index.values)]
        CellTypeLabel = SingleCellTypeIdentification(self.InitProp, cell_locations, 'spot_index', self.Q_mat_all, self.X_vals_loc, nu = nu, n_epoch = 8, n_neighbo = n_neighbo, loggings = self.loggings, hs_ST = hs_ST)
        label2ct = CellTypeLabel['label2ct']
        discrete_label = CellTypeLabel['discrete_label'].copy()
        discrete_label['discrete_label_ct'] = label2ct.iloc[discrete_label['discrete_label']].values.squeeze()
        discrete_label.to_csv(os.path.join(self.out_dir, 'CellTypeLabel_nu' + str(nu) + '.csv'))
        self.sp_adata.uns['cell_locations'] = discrete_label

        if hs_ST:
            fig, ax = plt.subplots(figsize=(10,8.5),dpi=100)
            sns.scatterplot(data=discrete_label, x="x",y="y",s=10,hue='discrete_label_ct',palette='tab20',legend=True)
            plt.axis('off')
            plt.legend(bbox_to_anchor=(0.97, .98),framealpha=0)
            plt.savefig(os.path.join(self.out_dir, 'estemated_ct_label.png'))
            plt.close()
        
        else:
            if self.sp_adata.obsm['spatial'].shape[1] == 2:
                fig, ax = plt.subplots(1,1,figsize=(14, 8),dpi=200)
                PlotVisiumCells(self.sp_adata,"discrete_label_ct",size=0.4,alpha_img=0.4,lw=0.4,palette='tab20',ax=ax)
                plt.savefig(os.path.join(self.out_dir, 'estemated_ct_label.png'))
                plt.close()
                
        # estimate platform batch effect
        add_genes = np.array(self.sp_adata.var.index[self.sp_adata.var.index.isin(self.sc_adata.var.index)])
        self.sp_adata_be = self.sp_adata[:,add_genes].copy()

        counts = self.sp_adata_be.to_df().T
        
        if hs_ST:
            if 'z' in self.sp_adata.obs.columns:
                coords = self.sp_adata.obs[['x', 'y', 'z']]
            else:
                coords = self.sp_adata.obs[['x', 'y']]
        else:
            if self.sp_adata.obsm['spatial'].shape[1] == 2:
                coords = pd.DataFrame(self.sp_adata.obsm['spatial'], index = counts.columns, columns = ['x', 'y'])
            elif self.sp_adata.obsm['spatial'].shape[1] == 3:
                coords = pd.DataFrame(self.sp_adata.obsm['spatial'], index = counts.columns, columns = ['x', 'y', 'z'])
            else:
                self.loggings.error('Wrong spatial location shape: {}'.format(self.sp_adata.obsm['spatial'].shape))
                sys.exit()
        nUMI = pd.DataFrame(np.array(self.sp_adata_be.X.sum(-1)), index = self.sp_adata_be.obs.index)
        puck = SpatialRNA(coords, counts, nUMI)
        counts = self.sc_adata.to_df().T
        cell_types = pd.DataFrame(self.sc_adata.obs[self.cell_class_column])
        nUMI = pd.DataFrame(self.sc_adata.to_df().T.sum(0))
        reference = Reference(counts, cell_types, nUMI, loggings=self.loggings)
        estimated_be = calculate_batch_effect(puck, reference, add_genes, max_cores = 22, loggings = self.loggings)
            
        sp_data = self.sp_adata_be.copy()
        sc_data = self.sc_adata.copy()
        
        ## visualize estimated_be
        genes = sp_data.var.index
        sp_data_be = sp_data.copy()[:, genes]
        sc_data_be = sc_data.copy()[:,genes]


        log_gamma = np.array(np.log2(sp_data_be.X.mean(0) / sc_data_be.X.mean(0))).squeeze()
        exponential_gamma = estimated_be[genes]

        index_nan = np.isnan(exponential_gamma)
        index_0 = exponential_gamma == 0
        index_inf = np.isinf(exponential_gamma)
        gamma_plot = pd.DataFrame()
        gamma_plot['reference batch effect'] = log_gamma[~(index_nan | index_0 | index_inf)]
        gamma_plot['estimated'] = np.array(np.log(exponential_gamma[~(index_nan | index_0 | index_inf)]/ sc_data_be.X.sum(-1).mean() * sp_data_be.X.sum(-1).mean()))

        with mpl.rc_context({'figure.figsize': (10, 10)}):
            sns.scatterplot(data = gamma_plot, x = 'reference batch effect', y = 'estimated')
            plt.plot(np.arange(-10,10,1), np.arange(-10,10,1))
            plt.xlim([-10, 10])
            plt.ylim([-10, 10])
            plt.title('estimated batch effect')
            plt.savefig(os.path.join(self.out_dir, 'estimated_batch_effect.png'))
        #     plt.show()
            plt.close()

            
        # save data
        self.sp_adata_be.var['estimated_be'] = exponential_gamma / sc_data_be.X.sum(-1).mean() * sp_data_be.X.sum(-1).mean()
        self.sp_adata_be.write_h5ad(os.path.join(self.out_dir, 'sp_adata.h5ad'))
                                    
        ray.shutdown()
            


if __name__ == "__main__":
    HEADER = """
    <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    <> 
    <> SpatialScope Cell Type Identification
    <> Version: %s
    <> MIT License
    <>
    <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    <> Software-related correspondence: %s or %s
    <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    <> Visium data example:
        python <install path>/src/Cell_Type_Identification.py \\
            --cell_class_column cell_type \\
            --tissue heart \\
            --out_dir ./output \\
            --ST_Data ./output/heart/sp_adata_ns.h5ad \\
            --SC_Data ./Ckpts_scRefs/Heart_D2/Ref_Heart_sanger_D2.h5ad 
    <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>  
    """ 
  
    parser = argparse.ArgumentParser(description='simulation sour_sep')
    parser.add_argument('--tissue', type=str, help='tissue name', default=None)
    parser.add_argument('--out_dir', type=str, help='output path', default=None)
    parser.add_argument('--nu', type=float, help='spatial prior parameter, higher nu means stronger spatial prior', default=10)
    parser.add_argument('--n_neighbo', type=int, help='spatial prior parameter, the range of neighbor cells', default=10)
    parser.add_argument('--ST_Data', type=str, help='ST data path', default=None)
    parser.add_argument('--SC_Data', type=str, help='single cell reference data path', default=None)
    parser.add_argument('--cell_class_column', type=str, help='input cell class label column in scRef file', default = 'cell_type')    
    parser.add_argument('--hs_ST', action="store_true", help='high resolution ST data such as Slideseq, DBiT-seq, and HDST, MERFISH etc.')
    parser.add_argument('--UMI_min_sigma', type=int, help='WarmStart parameter', default=300)
    parser.add_argument('--InitProp', type=str, help='whether to run warmstart', default = None)   
    args = parser.parse_args()
    
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    if not os.path.exists(os.path.join(args.out_dir,args.tissue)):
        os.mkdir(os.path.join(args.out_dir,args.tissue))

    if (args.nu - int(args.nu)) == 0:
        args.nu = int(args.nu)
    
    CTI = SpatialScopeCTI(args.tissue,args.out_dir, args.ST_Data, args.SC_Data, cell_class_column = args.cell_class_column, InitProp = args.InitProp)
    CTI.CellTypeIdentification(nu = args.nu, n_neighbo = args.n_neighbo, hs_ST = args.hs_ST, UMI_min_sigma = args.UMI_min_sigma)