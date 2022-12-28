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

# from WaveGrad.utils import configure_logging
from utils import *
from utils_CTI import *
num_cpus = psutil.cpu_count(logical=False) 
os.environ["PYTHONPATH"] = os.getcwd()+ ":" + os.environ.get("PYTHONPATH", "")
ray.shutdown()
ray.init(num_cpus=num_cpus,_temp_dir='/tmp/ray')


import copy
import argparse


class SpatialScopeCTI:
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

        # cell_class_column = args.cell_class_column


        if sc_adata.X.max()<30:
            try:
                sc_adata.X = np.exp(sc_adata.X) - 1
            #     sc.pp.normalize_total(sc_adata, inplace=True)
            except:
                sc_adata.X = np.exp(sc_adata.X.toarray()) - 1
            #     sc.pp.normalize_total(sc_adata, inplace=True)
            
        self.sp_adata = sp_adata
        self.sc_adata = sc_adata
        self.cell_class_column = cell_class_column
        
        
    def WarmStart(self, loc = 'spatial', X = None, Y = None):
        self.LoadLikelihoodTable()
        
        
        counts = self.sp_adata.to_df().T
        if loc == 'spatial':
            coords = pd.DataFrame(self.sp_adata.obsm['spatial'], index = counts.columns, columns = ['x', 'y'])
        elif loc == 'obs':
            coords = self.sp_adata.obs.loc[:, [X,Y]]
        nUMI = pd.DataFrame(np.array(self.sp_adata.X.sum(-1)), index = self.sp_adata.obs.index)
        puck = SpatialRNA(coords, counts, nUMI)
        counts = self.sc_adata.to_df().T
        cell_types = pd.DataFrame(self.sc_adata.obs[self.cell_class_column])
        nUMI = pd.DataFrame(self.sc_adata.to_df().T.sum(0))
        reference = Reference(counts, cell_types, nUMI)
        myRCTD = create_RCTD(puck, reference, max_cores = 22, loggings = self.loggings)
        myRCTD = run_RCTD(myRCTD, self.Q_mat_all, self.X_vals_loc, loggings = self.loggings)
        self.InitProp = myRCTD
        import pickle
        with open(os.path.join(self.out_dir, 'InitProp.pickle'), 'wb') as handle:
            pickle.dump(self.InitProp, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return self.InitProp
        
    def LoadLikelihoodTable(self):
        with gzip.open(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/extdata/Q_mat_1.txt.gz','rt') as f:
            lines = f.readlines()

        Q1 = {}
        for i in range(len(lines)):
            if i == 61:
                Q1[str(72)] = np.reshape(np.array(lines[i].split(' ')).astype(np.float), (2536, 103)).T
            elif i == 62:
                Q1[str(74)] = np.reshape(np.array(lines[i].split(' ')).astype(np.float), (2536, 103)).T
            else:
                Q1[str(i + 10)] = np.reshape(np.array(lines[i].split(' ')).astype(np.float), (2536, 103)).T

        with gzip.open(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/extdata/Q_mat_2.txt.gz','rt') as f:
            lines2 = f.readlines()

        Q2 = {}
        for i in range(len(lines2)):
            Q2[str(int(i * 2 + 76))] = np.reshape(np.array(lines2[i].split(' ')).astype(np.float), (2536, 103)).T

        Q_mat_all = Q1 | Q2

        with open(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/extdata/X_vals.txt') as f:
            lines_X = f.readlines()

        X_vals_loc = np.array([float(lines_X_item.strip()) for lines_X_item in lines_X])
        
        self.Q_mat_all = Q_mat_all
        self.X_vals_loc = X_vals_loc
        
    def CellTypeIdentification(self, nu = 10, loc = 'spatial', X = None, Y = None):
        cell_locations = self.sp_adata.uns['cell_locations'].copy()
        if self.InitProp is None and not os.path.exists(os.path.join(self.out_dir, 'InitProp.pickle')):
            self.WarmStart(loc = loc, X = X, Y = Y)
        elif self.InitProp is None and os.path.exists(os.path.join(self.out_dir, 'InitProp.pickle')):
            with open(os.path.join(self.out_dir, 'InitProp.pickle'), 'rb') as handle:
                self.InitProp = pickle.load(handle)
                self.LoadLikelihoodTable()
            
        if not os.path.exists(os.path.join(self.out_dir, 'CellTypeLabel_nu_' + str(nu) + '.csv')):
            CellTypeLabel = SingleCellTypeIdentification(self.InitProp, cell_locations, 'spot_index', self.Q_mat_all, self.X_vals_loc, nu = nu, n_epoch = 8, loggings = self.loggings)
            label2ct = CellTypeLabel['label2ct']
            discrete_label = CellTypeLabel['discrete_label'].copy()
            discrete_label['discrete_label_ct'] = label2ct.iloc[discrete_label['discrete_label']].values.squeeze()
            discrete_label.to_csv(os.path.join(self.out_dir, 'CellTypeLabel_nu_' + str(nu) + '.csv'))

            # with mpl.rc_context({'figure.figsize': (10, 10)}):
            #     ax = sns.scatterplot(data=discrete_label, x="x", y="y", hue="discrete_label_ct", s = 30, legend = False)
            #     plt.savefig(os.path.join(self.out_dir, 'estemated_ct_label1.png'))
            #     plt.close()
                
            self.sp_adata.uns['cell_locations'] = discrete_label
            fig, ax = plt.subplots(1,1,figsize=(14, 8),dpi=200)
            PlotVisiumCells(self.sp_adata,"discrete_label_ct",size=0.4,alpha_img=0.4,lw=0.4,palette='tab20',ax=ax)
            plt.savefig(os.path.join(self.out_dir, 'estemated_ct_label.png'))
            plt.close()
                
        # estimate platform batch effect
        add_genes = np.array(self.sp_adata.var.index[self.sp_adata.var.index.isin(self.sc_adata.var.index)])
        self.sp_adata_be = self.sp_adata[:,add_genes].copy()

        counts = self.sp_adata_be.to_df().T
        if loc == 'spatial':
            coords = pd.DataFrame(self.sp_adata.obsm['spatial'], index = counts.columns, columns = ['x', 'y'])
        elif loc == 'obs':
            coords = self.sp_adata.obs.loc[:, [X,Y]]
        nUMI = pd.DataFrame(np.array(self.sp_adata_be.X.sum(-1)), index = self.sp_adata_be.obs.index)
        puck = SpatialRNA(coords, counts, nUMI)
        counts = self.sc_adata.to_df().T
        cell_types = pd.DataFrame(self.sc_adata.obs[self.cell_class_column])
        nUMI = pd.DataFrame(self.sc_adata.to_df().T.sum(0))
        reference = Reference(counts, cell_types, nUMI)
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
    <> example:
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
    parser.add_argument('--ST_Data', type=str, help='ST data path', default=None)
    parser.add_argument('--SC_Data', type=str, help='single cell reference data path', default=None)
    parser.add_argument('--cell_class_column', type=str, help='input cell class label column in scRef file', default = 'cell_type')
    parser.add_argument('--location', type=str, help='spatial data location, e.g., .obsm[\'spatial\'] for Visum data', default='spatial')
    parser.add_argument('--X', type=str, help='spatial data location X', default='x')
    parser.add_argument('--Y', type=str, help='spatial data location Y', default='y')

    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    if not os.path.exists(os.path.join(args.out_dir,args.tissue)):
        os.mkdir(os.path.join(args.out_dir,args.tissue))


    CTI = SpatialScopeCTI(args.tissue,args.out_dir, args.ST_Data, args.SC_Data, cell_class_column = args.cell_class_column)
    CTI.CellTypeIdentification(nu = args.nu, loc = args.location, X = args.X, Y = args.Y)