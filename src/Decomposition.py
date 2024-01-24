import scanpy as sc
import anndata
import pandas as pd
import numpy as np
import torch
from torch import nn
import os
import tqdm
import logging
import scipy
import json
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from SCGrad import SCGradNN, EMAHelper

import warnings
warnings.filterwarnings('ignore')

import copy
import argparse
from utils import *
from utils_CTI import *


class GeneExpDecomposition:
    def __init__(self, config, loggings = None):
        self.config = config
        self.out_dir = os.path.join(self.config.data.out_dir, self.config.data.tissue)
        if loggings is None:
            loggings = configure_logging(os.path.join(self.out_dir,'logs'))
            self.loggings = loggings
        else:
            self.loggings = loggings
        
    def LoadScData(self):
        self.loggings.info(f'load scRNA-seq reference: {self.config.data.SC_Data}')
        sc_data_process = anndata.read_h5ad(self.config.data.SC_Data)

        if 'Marker' in sc_data_process.var.columns:
            self.sc_data_process_marker = sc_data_process[:,sc_data_process.var['Marker']]
        else:
            self.sc_data_process_marker = sc_data_process

        if self.sc_data_process_marker.X.max()>30:
            self.loggings.info(f'scRNA-seq reference Maximum value: {self.sc_data_process_marker.X.max()}, need to run log1p')
            sc.pp.log1p(self.sc_data_process_marker)

        cell_type_array = np.array(self.sc_data_process_marker.obs[self.config.data.cell_class_column])
        cell_type_class = np.unique(cell_type_array)
        df_category = self.sc_data_process_marker.obs[[self.config.data.cell_class_column]].astype('category').apply(lambda x: x.cat.codes)


        # parameters: mean and cell type index
        cell_type_array_code = np.array(df_category[self.config.data.cell_class_column]) 
        try:
            self.data = self.sc_data_process_marker.X.toarray()
        except:
            self.data = self.sc_data_process_marker.X
        self.loggings.info(f'scRNA-seq reference data shape: {self.data.shape}')
        n, d = self.data.shape
        q = cell_type_class.shape[0]
        self.loggings.info(f'scRNA-seq reference cell class number: {q}')
        self.mu = np.zeros((q, d))
        for k in range(q):
            self.mu[k] = self.data[cell_type_array_code == k].mean(0).squeeze() 
        u, indices = np.unique(np.array(df_category[self.config.data.cell_class_column]), return_counts = True)
        ####################################################################################################################################
        # make cell_type2mu coding df
        code2cell_type = []
        for i in range(q):
            index = cell_type_array_code == i
            index = np.array(index)
            code2cell_type.append(np.array(self.sc_data_process_marker.obs[self.config.data.cell_class_column])[index][0])

        code2cell_type = np.array(code2cell_type)
        self.cell_type2mu = pd.DataFrame(np.arange(code2cell_type.shape[0]), index = code2cell_type)
        
    def LoadModel(self):
        self.loggings.info(f'load checkpoint: {self.config.ckpt_path}')
        self.loggings.info(f'is GPU availabel: {torch.cuda.is_available()}')
        self.loggings.info(f'availabel GPUs: {torch.cuda.device_count()}')
        states = torch.load(self.config.ckpt_path, map_location=lambda storage, loc: storage)
        if self.data.shape[1]!=states['Genes'].shape[0]:
            self.loggings.error('Wrong checkpoints/scRef file, checkpoint gene number: {}, scRef marker genes number: {}'.format(states['Genes'].shape[0],self.data.shape[1]))
            sys.exit()
        if self.config.data.cell_class_column!=states['cell_type_column']:
            self.loggings.error('Wrong checkpoints/scRef file, checkpoint cell_class_column: {}, scRef cell_class_column: {}'.format(states['cell_class_column'],self.config.data.cell_class_column))
            sys.exit()
        if 'factors' in states.keys():
            model_single = SCGradNN(input_dim1 = self.data.shape[1], down_block_dim=states['down_block_dim'], factors = states['factors']).to(self.config.device)
        else:
            model_single = SCGradNN(input_dim1 = self.data.shape[1], down_block_dim=states['down_block_dim']).to(self.config.device)
        model = nn.DataParallel(model_single, device_ids = list(range(torch.cuda.device_count())))

        ema_helper = EMAHelper()
        ema_helper.register(model)

        
        ema_helper.load_state_dict(states['ckpt'])
        ema_helper.check_device(model, self.config.device)

        ema_helper.ema(model)
        model.eval()
        self.model = model
        
        self.config.sigma=ConfigWrapper(**states["sigma"])
        self.sigmas = torch.tensor(
                    np.exp(np.linspace(np.log(self.config.sigma.sigma_begin), np.log(self.config.sigma.sigma_end),
                                       self.config.sigma.num_classes))).float()
        
    def LoadSTdata(self):
        self.loggings.info('load ST data: {}'.format(os.path.join(self.out_dir, 'sp_adata.h5ad')))
        if not os.path.exists(os.path.join(self.out_dir, 'sp_adata.h5ad')):
            self.loggings.error('{} does not exist, shold run step II, Cell_Type_Identification.py, first'.format(os.path.join(self.out_dir, 'sp_adata.h5ad')))
            sys.exit()
            
        self.config.data.ST_DataDir = os.path.join(self.out_dir, 'sp_adata.h5ad')
        spatial_adata = anndata.read_h5ad(os.path.join(self.out_dir, 'sp_adata.h5ad'))
        self.loggings.info(f'ST data shape: {spatial_adata.shape}')

        if isinstance(spatial_adata.X, scipy.sparse.csr_matrix):
            spatial_adata.X = spatial_adata.X.toarray()

        if spatial_adata.X.max()<30:
            sp_adata.X = np.exp(sp_adata.X) - 1
                
        if self.config.decomposition.test_genes is not None:
            test_gene = np.array(self.config.decomposition.test_genes)
        
        # correct platform batche effect
        spatial_adata_expgamma = np.array((spatial_adata.var['estimated_be'])).squeeze()
        index = np.isnan(spatial_adata_expgamma)
        spatial_adata_expgamma[index] = 1
        index = spatial_adata_expgamma == 0
        spatial_adata_expgamma[index] = 1
        index = np.isinf(spatial_adata_expgamma)
        spatial_adata_expgamma[index] = 1
        spatial_adata.raw = spatial_adata
        spatial_adata.X = spatial_adata.raw.X.copy()
        spatial_adata.X = spatial_adata.X / spatial_adata_expgamma[None,:]

        
        # meke gene mask
        if self.config.decomposition.leave_out_test:
            self.loggings.info('Leave testing genes out: {}'.format(','.join(test_gene)))
            gene_index1 = self.sc_data_process_marker.var.index.isin(spatial_adata.var.index)
            gene_index2 = self.sc_data_process_marker.var.index.isin(test_gene)
            self.gene_mask = (gene_index1) & (~gene_index2)
        else:
            self.gene_mask = self.sc_data_process_marker.var.index.isin(spatial_adata.var.index)

        self.spatial_adata_reorder = spatial_adata[:,self.sc_data_process_marker.var.index[self.gene_mask]]
        self.spatial_adata_reorder.X = self.spatial_adata_reorder.X / self.spatial_adata_reorder.X.sum(-1)[:,None] * (np.exp(self.data[:, self.gene_mask]) - 1).sum(-1).mean()

        
    def single_decomposition(self, rep_No = 0):
        self.CellTypeLabel = self.spatial_adata_reorder.uns['cell_locations']
        sp_index = np.array(KeepOrderUnique(self.CellTypeLabel['spot_index']))
        self.spatial_adata_reorder = self.spatial_adata_reorder[sp_index,:]

        # decomposition specific spot
        sp_index_table = pd.DataFrame(np.arange(sp_index.shape[0]),index = sp_index)

        spot_label = np.arange(sp_index_table.shape[0])
        if self.config.decomposition.spot_range is not None:
            self.spot_range = self.config.decomposition.spot_range
        else:
            self.spot_range = [0,sp_index.shape[0]]
        
        Index = (spot_label >= self.spot_range[0]) & (spot_label < self.spot_range[1])
        spot_ids = spot_label[Index]
        subspot_label = sp_index_table.loc[self.CellTypeLabel['spot_index'].values].values.squeeze()
        index = (subspot_label >= self.spot_range[0]) & (subspot_label < self.spot_range[1])

        for l in range(5):
            if l > 0:
                index = nan_list + self.spot_range[0]
                Index = np.unique(subspot_label[index])
                spot_ids = spot_label[Index]

            estimated_ct = np.array(self.CellTypeLabel['discrete_label_ct'])[index] 
            estimated = np.array(self.cell_type2mu.loc[estimated_ct]).squeeze(-1)
            subspot_ids = subspot_label[index]
            cell_num_total = np.array([np.where(self.CellTypeLabel['spot_index'].values == sp_name)[0].shape[0] for sp_name in sp_index])
            cell_num_total = cell_num_total[Index].squeeze()


            y = self.spatial_adata_reorder.X[Index]

            subspot_ids4decom = []
            mask = np.zeros((y.shape[0], estimated.shape[0])) # #spot * #cells
            for i, spot_id in enumerate(spot_ids):
                ind = subspot_ids == spot_id
                mask[i,ind] = 1
                subspot_ids4decom = subspot_ids4decom + np.where(ind == True)[0].shape[0] * [i]
            subspot_ids4decom = np.array(subspot_ids4decom)
            if self.config.decomposition.save_process:
                x_decom0, collected_samples = self.Langevin_Decom(estimated, self.mu, y, self.model, mask, cell_num_total, subspot_ids4decom, self.gene_mask)
            else:
                x_decom0 = self.Langevin_Decom(estimated, self.mu, y, self.model, mask, cell_num_total, subspot_ids4decom, self.gene_mask)

            x_decom0 = np.array(x_decom0)
            if l == 0:
                x_decom = x_decom0.copy()
            if l > 0:
                x_decom[nan_list] = x_decom0.squeeze()
                self.loggings.info('######################################################## iter:{} | nan list: {}'.format(l, nan_list))
            
            nan_list = np.where(np.isnan(x_decom)==True)[0]
            if nan_list.shape[0] == 0:
                break  

        spot_label = np.arange(sp_index_table.shape[0])
        Index = (spot_label >= self.spot_range[0]) & (spot_label < self.spot_range[1])
        spot_ids = spot_label[Index]
        subspot_label = sp_index_table.loc[self.CellTypeLabel['spot_index'].values].values.squeeze()
        index = (subspot_label >= self.spot_range[0]) & (subspot_label < self.spot_range[1])
        estimated_ct = np.array(self.CellTypeLabel['discrete_label_ct'])[index]
        estimated = np.array(self.cell_type2mu.loc[estimated_ct]).squeeze()

        # output results
        x_decom_obs = pd.DataFrame()
        x_decom_obs.index = np.array(self.CellTypeLabel['cell_index'][index]).squeeze()
        x_decom_obs['cell_type_label'] = estimated_ct
        x_decom_obs['spot_index'] = np.array(self.CellTypeLabel['spot_index'][index]).squeeze()
        try:
            x_decom_obs['cell_nums'] = np.array(self.CellTypeLabel['cell_nums'][index]).squeeze()
        except:
            x_decom_obs['cell_nums'] = np.array(self.CellTypeLabel['cell_num'][index]).squeeze()
        
        x_decom_var = pd.DataFrame()
        x_decom_var.index = self.sc_data_process_marker.var.index
        x_decom_var['exp_gamma'] = np.zeros(self.sc_data_process_marker.shape[-1]) - 1
        x_decom_var['exp_gamma'][self.gene_mask] = np.array(self.spatial_adata_reorder.var['estimated_be'])
        x_decom_adata = anndata.AnnData(X = np.array(x_decom).copy(), obs = x_decom_obs, var = x_decom_var)

        separator = '_'
        spot_range = separator.join(str(e) for e in self.spot_range)

        if not os.path.exists(os.path.join(self.out_dir, 'decomposition')):
            os.makedirs(os.path.join(self.out_dir, 'decomposition'))
        if not os.path.exists(os.path.join(self.out_dir, 'decomposition', 'rep' + str(rep_No))):
            os.makedirs(os.path.join(self.out_dir, 'decomposition', 'rep' + str(rep_No)))
        x_decom_adata.write(os.path.join(self.out_dir, 'decomposition', 'rep' + str(rep_No),  spot_range + '_Sig' + str(self.config.decomposition.power) + '_Lr' + str(self.config.sigma.step_lr_decom) + '.h5ad'))
        if self.config.decomposition.save_process:
            np.save(os.path.join(self.out_dir, 'decomposition', 'rep' + str(rep_No),  spot_range + '_Sig' + str(self.config.decomposition.power) + '_Lr' + str(self.config.sigma.step_lr_decom) + '_' + 'median_samples.npy'), np.array(collected_samples))

        
    def Langevin_Decom(self, estimated, mu, y, model, mask, cell_num_total, subspot_ids, gene_mask):
        clip1 = 1.
        clip2 = 2.
        d = mu.shape[-1]

        q = estimated.shape[0]
    #     x_decom = (torch.randn(q, d) * sigma_begin).to(self.config.device)
        x_decom = (torch.randn(q, d)).to(self.config.device)
        mask = torch.Tensor(mask).to(self.config.device)
        cell_num_total = torch.Tensor(cell_num_total).to(self.config.device)

        mu = torch.Tensor(mu.copy()[estimated]).to(self.config.device)
        cell_type = torch.tensor((estimated).copy().squeeze()).to(self.config.device)

        y = torch.Tensor(y).to(self.config.device)
        collected_sample = []
        with torch.no_grad():
            for c, sigma in tqdm.tqdm(enumerate(self.sigmas), total=len(self.sigmas), desc='annealed Langevin dynamics sampling'):
                eta = self.config.sigma.step_lr_decom * (sigma / self.sigmas[-1]) ** 2

                noise_level = torch.ones(x_decom.shape[0], device=x_decom.device) * sigma

            ####################################################################################
                lambda_recon = 1.0/(sigma**self.config.decomposition.power)
            ####################################################################################
                for i in range(self.config.sigma.time):    
                    noise = torch.randn_like(x_decom) * np.sqrt(eta * 2)

                    grad_x = model(x_decom, mu, noise_level) / sigma
                    grad_x = grad_x.detach()
                    diff = torch.zeros_like(x_decom).to(self.config.device)

                    grad_mix = torch.cat([(torch.exp(x_decom + mu) - 1)[mask[_] == 1].sum(0)[None,:] for _ in range(mask.shape[0])])[:,gene_mask] - y * cell_num_total[:,None]
                    grad_mix = torch.clamp(grad_mix,-clip1,clip1)

                    grad_mix =  torch.exp(x_decom + mu)[:,gene_mask] * grad_mix[(subspot_ids - subspot_ids[0])]
                    grad_mix = torch.clamp(grad_mix,-clip2,clip2)


                    diff[:,gene_mask] = grad_mix
                    x_decom += eta * (grad_x - lambda_recon * diff) + noise

                    if self.config.decomposition.verbose:
                        if (i + 1) % 5 == 0:
                            self.loggings.info("i:{} | sigma:{}".format(i, sigma))
                
                if self.config.decomposition.verbose:
                    self.loggings.info('Test, {},{} | {}, {}'.format(grad_x.mean(),grad_x.var(), (lambda_recon * grad_mix).mean(), (lambda_recon * grad_mix).var()))
                if self.config.decomposition.leave_out_test:
                    collected_single_sample = (x_decom + mu).clone()
                    collected_sample.append(np.array(collected_single_sample.to('cpu')))

            grad_x = model(x_decom, mu, noise_level) / sigma        

            grad_mix = torch.cat([(torch.exp(x_decom + mu) - 1)[mask[i] == 1].sum(0)[None,:] for i in range(mask.shape[0])])[:,gene_mask] - y * cell_num_total[:,None]
            grad_mix = torch.clamp(grad_mix,-clip1,clip1)

            grad_mix =  torch.exp(x_decom + mu)[:,gene_mask] * grad_mix[(subspot_ids - subspot_ids[0])]
            grad_mix = torch.clamp(grad_mix,-clip2,clip2)

            diff[:,gene_mask] = grad_mix
            x_decom += sigma**2 * (grad_x - lambda_recon * diff) 

            x_decom = x_decom + mu
            x_decom = x_decom.to('cpu')
            mu = mu.to('cpu')
            y = y.to('cpu')
        if self.config.decomposition.save_process:
            return x_decom, collected_sample
        else:
            return x_decom


        
    def posterior_mean(self):
        self.CellTypeLabel = self.spatial_adata_reorder.uns['cell_locations']
        sp_index = np.array(KeepOrderUnique(self.CellTypeLabel['spot_index']))
        # decomposition specific spot
        sp_index_table = pd.DataFrame(np.arange(sp_index.shape[0]),index = sp_index)

        spot_label = np.arange(sp_index_table.shape[0])
        subspot_label = sp_index_table.loc[self.CellTypeLabel['spot_index'].values].values.squeeze()
        index = (subspot_label >= self.spot_range[0]) & (subspot_label < self.spot_range[1])
        
        separator = '_'
        spot_range = separator.join(str(e) for e in self.spot_range)
        
        cell_files = []
        reorder_index_all = []
        cell_file = sc.read(os.path.join(self.out_dir, 'decomposition', 'rep0',  spot_range + '_Sig' + str(self.config.decomposition.power) + '_Lr' + str(self.config.sigma.step_lr_decom) + '.h5ad'))
        
        cell_files.append(cell_file)
        reorder_index_all.append(np.arange(cell_file.shape[0]).astype(int))
        for rep_No in np.arange(1,self.config.decomposition.replicates):
            cells = sc.read(os.path.join(self.out_dir, 'decomposition', f'rep{rep_No}',  spot_range + '_Sig' + str(self.config.decomposition.power) + '_Lr' + str(self.config.sigma.step_lr_decom) + '.h5ad'))
            cell_files.append(cells)
            reorder_index_all.append(reorder(cell_file, cells))

        mean_express = np.zeros(cell_file.shape)
        for i,cells in enumerate(cell_files):
            cells.X = np.exp(cells.X) - 1
            self.loggings.info(cells.X.max())
            try:
                mean_express = mean_express + np.array(cells.X[reorder_index_all[i]])
            except:
                mean_express = mean_express + np.array(cells.X.toarray()[reorder_index_all[i]])
        mean_express = mean_express / len(cell_files)
        mean_express[mean_express < 0] = 0 
        mean_express = pd.DataFrame(data = mean_express, index = cell_file.to_df().index, columns = cell_file.to_df().columns)

        x_decom_adata = sc.AnnData(mean_express.reset_index(drop=True), obs=self.CellTypeLabel.loc[index,:])
        x_decom_adata.write(os.path.join(self.out_dir,  'generated_cells_spot{}.h5ad'.format('_'.join(str(_) for _ in self.spot_range))))
        self.loggings.info('save generated single-cell resolution ST data at: {}'.format(os.path.join(self.out_dir, 'generated_cells_spot{}.h5ad'.format('_'.join(str(_) for _ in self.spot_range)))))
        
        
    def decomposition(self):
        self.LoadScData()
        self.LoadModel()
        self.LoadSTdata()  

        for i in range(self.config.decomposition.replicates):
            self.loggings.info(f'Gene expression decomposition round: {i}')
            self.single_decomposition(rep_No = i)
                
        with open(os.path.join(self.out_dir,  'config.json'), 'w') as f:
            self.config.device = str(self.config.device)
            json.dump(self.config.to_dict_type(), f)
        
            
if __name__ == "__main__":
    HEADER = """
    <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    <> 
    <> decomposition: SpatialScope Singel-cell Gene Expression decomposition
    <> Version: %s
    <> MIT License
    <>
    <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    <> Software-related correspondence: %s or %s
    <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    <> example:
        python <install path>/src/Decomposition.py \\
            --tissue heart \\
            --out_dir ./output \\
            --SC_Data ./Ckpts_scRefs/Heart_D2/Ref_Heart_sanger_D2.h5ad \\
            --ckpt_path ./Ckpts_scRefs/Heart_D2/model_600001.pt \\
            --spot_range 0,1000  --gpu 5,6,7,8 
    <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>  
    """ 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell_class_column', type=str, help='input cell class label column in scRef file', default = 'cell_type')
    parser.add_argument('--tissue', type=str, help='tissue name', default=None)
    parser.add_argument('--out_dir', type=str, help='output path, shuold be identical to --out_dir in Cell_Type_Identification.py', default=None)
    parser.add_argument('--SC_Data', type=str, help='single cell reference data path', default=None)
    parser.add_argument('--ckpt_path', type=str, help='checkpoint file', default=None) 
    parser.add_argument('--spot_range', type=str, help='limited by GPU memory, we can only handle at most about 1000 spots in 4 GPUs at a time. \
                        e.g., 0,1000 means 0 to 1000-th spot', default=None) #[0,10]
    parser.add_argument('--gpu', type=str, help='gpus', default='0,1,2,3')
    
    parser.add_argument('--leave_out_test', action="store_true", help='leave some genes out as test') 
    parser.add_argument('--test_genes', type=str, help='test genes names, seperated by coma', default=None) # 'MYH11,ACTA2,JAG1'
    parser.add_argument('--replicates', type=int, help='perform decomposition with multiple replicates for robustness', default=3)
    parser.add_argument('--power', type=float, help='power of sigma', default=1)
    parser.add_argument('--save_process', action="store_true", help='save snapshots of decomposition process') 
    parser.add_argument('--verbose', action="store_true", help='print out some values in the process of decomposition, used for tuning sigma power') 
    
    args = parser.parse_args()

    # with open(os.path.join('configs', args.config + '.json')) as f:
    #     config = ConfigWrapper(**json.load(f))
    config = ConfigWrapper()
        
    config.data = ConfigWrapper()
    config.data.cell_class_column = args.cell_class_column
    config.data.tissue = args.tissue
    config.data.out_dir = args.out_dir
    config.data.SC_Data = args.SC_Data
        
    config.decomposition = ConfigWrapper()
    config.decomposition.leave_out_test = args.leave_out_test
    config.decomposition.test_genes = args.test_genes.split(',') if args.test_genes is not None else args.test_genes
    config.decomposition.save_process = args.save_process
    config.decomposition.power = args.power
    config.decomposition.verbose = args.verbose
    config.decomposition.replicates = args.replicates
    
    config.ckpt_path = args.ckpt_path

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device
        
    if args.spot_range is not None:
        config.decomposition.spot_range = [int(x) for x in args.spot_range.split(',')]
    else:
        config.decomposition.spot_range = args.spot_range
        
    DECOM = GeneExpDecomposition(config)
    DECOM.decomposition()
    DECOM.posterior_mean()
