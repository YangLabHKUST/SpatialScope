import os
import scanpy as sc
# import squidpy as sq
import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import matplotlib as mpl
# import skimage
import seaborn as sns
from itertools import chain
# from stardist.models import StarDist2D
from csbdeep.utils import normalize
from anndata import AnnData
from scipy.spatial.distance import pdist
import logging
import sys
from sklearn.metrics.pairwise import cosine_similarity

def PlotVisiumCells(adata,annotation_list,size=0.8,alpha_img=0.3,lw=1,subset=None,palette='tab20',show_circle = True, legend = True, ax=None,**kwargs):
    merged_df = adata.uns['cell_locations'].copy()
    test = sc.AnnData(np.zeros(merged_df.shape), obs=merged_df)
    test.obsm['spatial'] = merged_df[["x", "y"]].to_numpy().copy()
    test.uns = adata.uns
    
    if subset is not None:
        #test = test[test.obs[annotation_list].isin(subset)]
        test.obs.loc[~test.obs[annotation_list].isin(subset),annotation_list] = None
        
    sc.pl.spatial(
        test,
        color=annotation_list,
        size=size,
        frameon=False,
        alpha_img=alpha_img,
        show=False,
        palette=palette,
        na_in_legend=False,
        ax=ax,title='',sort_order=True,**kwargs
    )
    if show_circle:
        sf = adata.uns['spatial'][list(adata.uns['spatial'].keys())[0]]['scalefactors']['tissue_hires_scalef']
        spot_radius = adata.uns['spatial'][list(adata.uns['spatial'].keys())[0]]['scalefactors']['spot_diameter_fullres']/2
        for sloc in adata.obsm['spatial']:
            rect = mpl.patches.Circle(
                (sloc[0] * sf, sloc[1] * sf),
                spot_radius * sf,
                ec="grey",
                lw=lw,
                fill=False
            )
            ax.add_patch(rect)
    ax.axes.xaxis.label.set_visible(False)
    ax.axes.yaxis.label.set_visible(False)
    
    if not legend:
        ax.get_legend().remove()
    
    # make frame visible
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        
        

def assign_cell_type(cell_counts, cell_types):
    cell_type_list = []
    for i in range(cell_counts.shape[0]):
        cell_count = cell_counts[i]
        idx = np.where(cell_count > 0)[0]
        cell_type_list_row = [[cell_types[idx][_]] * cell_count[idx[_]] for _ in range(idx.shape[0])]
        cell_type_list_row = np.array([item for sublist in cell_type_list_row for item in sublist])
        np.random.shuffle(cell_type_list_row)
        cell_type_list = cell_type_list + list(cell_type_list_row)
        
    return cell_type_list

def distribute_cells(mat, cell_nums): # mat:  spots * cell_type; cell_nums: spots * 1
    cell_nums = cell_nums.astype(int)
    cell_nums_original = cell_nums.copy()
    
    mat[np.absolute(mat) < 1e-3] = 0
    cell_counts = np.zeros(mat.shape).astype(int)
    assert not np.any(cell_nums < 0)
    assert not np.any(mat < 0)
    
    mat = mat * cell_nums[:, None]
    cell_num_dist = np.floor(mat).astype(int)
    cell_counts = cell_counts + cell_num_dist
    cell_nums_remain = cell_nums - cell_num_dist.sum(1)
    mat_remain = mat - cell_num_dist

    assert not np.any(cell_nums_remain < 0)
    assert not np.any(mat_remain < 0)

    mat = mat_remain
    cell_nums = cell_nums_remain
    
    while(np.any(cell_nums_remain > 0)):
        mat[mat.argsort()[:, ::-1].argsort() >= cell_nums[:, None]] = 0
        mat = np.divide(mat, mat.sum(1)[:,None], out=np.zeros_like(mat), where=mat.sum(1)[:,None]!=0)
        mat = mat * cell_nums[:, None]
        cell_num_dist = np.floor(mat).astype(int)
        cell_counts = cell_counts + cell_num_dist
        cell_nums_remain = cell_nums - cell_num_dist.sum(1)
        mat_remain = mat - cell_num_dist

        assert not np.any(cell_nums_remain < 0)
        assert not np.any(mat_remain < 0)

        mat = mat_remain
        cell_nums = cell_nums_remain
        
    assert np.array_equal(cell_counts.sum(1), cell_nums_original)
    
    return cell_counts

    


def configure_logging(logger_name):
    LOG_LEVEL = logging.DEBUG
    log_filename = logger_name+'.log'
    importer_logger = logging.getLogger('importer_logger')
    importer_logger.setLevel(LOG_LEVEL)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')

    fh = logging.FileHandler(filename=log_filename)
    fh.setLevel(LOG_LEVEL)
    fh.setFormatter(formatter)
    importer_logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(LOG_LEVEL)
    sh.setFormatter(formatter)
    importer_logger.addHandler(sh)
    return importer_logger



