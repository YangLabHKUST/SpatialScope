import anndata
import numpy as np
import pandas as pd
import sys
import pickle
import os
import copy
import argparse
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns

from SDRCTD_utils import *




class SDRCTD:
    def __init__(self,tissue,out_dir,RCTD_results_dir,RCTD_results_name, ST_Data, SC_Data, cell_class_column = 'cell_type', hs_ST = True):
        self.tissue = tissue
        self.out_dir = out_dir 
        self.RCTD_results_dir = RCTD_results_dir 
        self.RCTD_results_name = RCTD_results_name 
        self.ST_Data = ST_Data
        self.SC_Data = SC_Data
        self.cell_class_column = cell_class_column
        self.hs_ST = hs_ST
        
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        if not os.path.exists(os.path.join(out_dir,tissue)):
            os.mkdir(os.path.join(out_dir,tissue))
        
        self.out_dir = os.path.join(out_dir,tissue)
        loggings = configure_logging(os.path.join(self.out_dir,'logs'))
        self.loggings = loggings 
        
        self.LoadRCTDresults()
        if SC_Data is not None:
            self.LoadSCData()

        
    def LoadRCTDresults(self):
        with open(os.path.join(self.RCTD_results_dir, self.RCTD_results_name + '.pickle'), 'rb') as handle:
            RCTD_results = pickle.load(handle)
            
        if self.hs_ST:
            try:
                weights = RCTD_results['results']['weights']
            except:
                weights = RCTD_results['results']
        else:
            weights = RCTD_results['results']

        
        self.weights = (weights / np.array(weights.sum(1))[:, None])
        
    def single_cell_type_assignment(self, cell_num_column = 'cell_count', VisiumCellsPlot = True):
        seged_sp_adata = sc.read(self.ST_Data) #ST_Data already complete nuclei segmentation with StarDist. 'cell_locations' already in uns

        mat = self.weights.values
        cell_nums = np.array(seged_sp_adata.obs[cell_num_column])

        cell_counts = distribute_cells(mat, cell_nums)
        cell_types = self.weights.columns
        cell_type_list = assign_cell_type(cell_counts, cell_types)
        seged_sp_adata.uns['cell_locations']['SDRCTD_cell_type'] = cell_type_list

        self.cell_type_list = cell_type_list
        self.seged_sp_adata = seged_sp_adata
        
        seged_sp_adata.uns['RCTD_weights'] = self.weights
        seged_sp_adata.write(os.path.join(self.out_dir, 'single_cell_type_label_bySDRCTD.h5ad'))
        
        # plot results
        if self.hs_ST or not VisiumCellsPlot:
            fig, ax = plt.subplots(figsize=(10,8.5),dpi=100)
            sns.scatterplot(data=seged_sp_adata.uns['cell_locations'], x="x",y="y",s=10,hue='SDRCTD_cell_type',palette='tab20',legend=True)
            plt.axis('off')
            plt.legend(bbox_to_anchor=(0.97, .98),framealpha=0)
            plt.savefig(os.path.join(self.out_dir, 'SDRCTD_estemated_ct_label.png'))
            plt.close()
        
        elif VisiumCellsPlot:
            if seged_sp_adata.obsm['spatial'].shape[1] == 2:
                fig, ax = plt.subplots(1,1,figsize=(14, 8),dpi=200)
                PlotVisiumCells(seged_sp_adata,"SDRCTD_cell_type",size=0.4,alpha_img=0.4,lw=0.4,palette='tab20',ax=ax)
                plt.savefig(os.path.join(self.out_dir, 'SDRCTD_estemated_ct_label.png'))
                plt.close()
                
    def cell_type_mean_assignment(self):
        # cell type mean as decomposed cell gene expression
        ref_df = pd.DataFrame([[ct, i]for i, ct in enumerate(self.sc_data_process_marker.obs[self.cell_class_column].astype('category').cat.categories)], columns = ['cell_type', 'cell_type_code'])
        ref_df.index = ref_df.cell_type
        ref_df = ref_df.iloc[:,1:]

        x_decom = self.mu[ref_df.loc[np.array(self.cell_type_list)].cell_type_code.tolist()]
        x_decom_adata = anndata.AnnData(X = x_decom.copy(), obs = self.seged_sp_adata.uns['cell_locations'].copy(), var = self.sc_data_process_marker.var)
        x_decom_adata.write(os.path.join(self.out_dir, 'cell_type_mean_bySDRCTD.h5ad'))

        
    def LoadSCData(self):
        # load sc data
        sc_data_process = anndata.read_h5ad(self.SC_Data)
        if 'Marker' in sc_data_process.var.columns:
            sc_data_process_marker = sc_data_process[:,sc_data_process.var['Marker']]
        else:
            sc_data_process_marker = sc_data_process

        if sc_data_process_marker.X.max() <= 30:
            self.loggings.info(f'Maximum value: {sc_data_process_marker.X.max()}, need to run exp')
            try:
                sc_data_process_marker.X = np.exp(sc_data_process_marker.X) - 1
            except:
                sc_data_process_marker.X = np.exp(sc_data_process_marker.X.toarray()) - 1


        cell_type_array = np.array(sc_data_process_marker.obs[self.cell_class_column])
        cell_type_class = np.unique(cell_type_array)
        df_category = sc_data_process_marker.obs[[self.cell_class_column]].astype('category').apply(lambda x: x.cat.codes)

        # parameters: mean and cell type index
        cell_type_array_code = np.array(df_category[self.cell_class_column]) 
        try:
            data = sc_data_process_marker.X.toarray()
        except:
            data = sc_data_process_marker.X

        n, d = data.shape
        q = cell_type_class.shape[0]
        self.loggings.info(f'scRNA-seq data shape: {data.shape}')
        self.loggings.info(f'scRNA-seq cell class number: {q}')

        mu = np.zeros((q, d))
        for k in range(q):
            mu[k] = data[cell_type_array_code == k].mean(0).squeeze()
        self.mu = mu
        self.sc_data_process_marker = sc_data_process_marker
    
    
        
    
            


if __name__ == "__main__":
    HEADER = """
    <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    <> 
    <> StarDist + RCTD
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
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
    
    parser = argparse.ArgumentParser(description='simulation sour_sep')
    parser.add_argument('--out_dir', type=str, help='output path', default=None)
    parser.add_argument('--RCTD_results_dir', type=str, help='RCTD results path', default=None)
    parser.add_argument('--RCTD_results_name', type=str, help='RCTD results file\'s name', default='InitProp')
    parser.add_argument('--ST_Data', type=str, help='ST data path', default=None)
    parser.add_argument('--SC_Data', type=str, help='single cell reference data path', default=None)
    parser.add_argument('--cell_class_column', type=str, help='input cell class label column in scRef file', default = 'cell_type')
    parser.add_argument('--cell_num_column', type=str, help='cell number column in spatial file', default = 'cell_count')
    parser.add_argument('--hs_ST', action="store_true", help='high resolution ST data such as Slideseq, DBiT-seq, and HDST, MERFISH etc.')
    parser.add_argument("--VisiumCellsPlot", type=str2bool, const=True, default=True, nargs="?", help="whether to plot in VisiumCells mode or just scatter plot")
    args = parser.parse_args()
    
    args.tissue = 'SDRCTD_results'
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    if not os.path.exists(os.path.join(args.out_dir,args.tissue)):
        os.mkdir(os.path.join(args.out_dir,args.tissue))
    if args.RCTD_results_dir is None:
        args.RCTD_results_dir = args.out_dir


    sdr = SDRCTD(args.tissue,args.out_dir, args.RCTD_results_dir, args.RCTD_results_name, args.ST_Data, args.SC_Data, cell_class_column = args.cell_class_column, hs_ST = args.hs_ST)
    sdr.single_cell_type_assignment(cell_num_column = args.cell_num_column, VisiumCellsPlot = args.VisiumCellsPlot)
    if args.SC_Data is not None:
        sdr.cell_type_mean_assignment()