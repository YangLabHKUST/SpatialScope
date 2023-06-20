# basic imports
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from spaotsc import SpaOTsc
from scipy import stats
import tangram as tg
import sys
# Example
# python ~/bio/SpatialBenchmarking/Codes/Deconvolution/SpaOTsc_pipeline.py \
# /home/share/xiaojs/spatial/sour_sep/mouce_brain_VISp/Ref_scRNA_VISp_qc2.h5ad \
# /home/share/xiaojs/spatial/sour_sep/tangram/merfish/MERFISH_mop.h5ad \
# cell_subclass \
# /home/share/xwanaf/sour_sep/simulation/SpaOTsc_test


sc_file_path = sys.argv[1]
spatial_file_path = sys.argv[2]
celltype_key = sys.argv[3]
output_file_path = sys.argv[4]

ad_sc = sc.read(sc_file_path)
ad_sp = sc.read(spatial_file_path)
df_sc = ad_sc.to_df()
df_IS = ad_sp.to_df()

try:
    pts = ad_sp.obs[['X','Y']].values
except:
    pts = ad_sp.obs[['x','y']].values
is_dmat = distance_matrix(pts, pts)

df_is=df_IS

gene_is=df_is.columns.tolist()
gene_sc=df_sc.columns.tolist()
gene_overloap=list(set(gene_is).intersection(gene_sc))
a=df_is[gene_overloap]
b=df_sc[gene_overloap]


rho, pval = stats.spearmanr(a, b,axis=1)
rho[np.isnan(rho)]=0
mcc=rho[-(len(df_sc)):,0:len(df_is)]
C = np.exp(1-mcc)

issc = SpaOTsc.spatial_sc(sc_data=df_sc, is_data=df_is, is_dmat = is_dmat)

issc.transport_plan(C**2, alpha=0, rho=1.0, epsilon=1.0, cor_matrix=mcc, scaling=False)
gamma = issc.gamma_mapping
for j in range(gamma.shape[1]):
    gamma[:,j] = gamma[:,j]/np.sum(gamma[:,j])
ad_map = sc.AnnData(gamma,obs = ad_sc.obs, var=ad_sp.obs)
tg.project_cell_annotations(ad_map, ad_sp, annotation=celltype_key)
ad_sp.obsm['tangram_ct_pred'].to_csv(output_file_path + '/SpaOTsc_decon.csv')

