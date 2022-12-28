import logging
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scanpy as sc
import os 
import matplotlib as mpl


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


class ConfigWrapper(object):
    """
    Wrapper dict class to avoid annoying key dict indexing like:
    `config.sample_rate` instead of `config["sample_rate"]`.
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = ConfigWrapper(**v)
            self[k] = v
      
    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def to_dict_type(self):
        return {
            key: (value if not isinstance(value, ConfigWrapper) else value.to_dict_type())
            for key, value in dict(**self).items()
        }

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()
    
    

def PlotVisiumCells(adata,annotation_list,size=0.8,alpha_img=0.3,lw=1,subset=None,palette='tab20',show_circle = True, legend = True, ax=None,**kwargs):
    merged_df = adata.uns['cell_locations'].copy()
    test = sc.AnnData(np.zeros(merged_df.shape), obs=merged_df)
    if 'x' in merged_df.columns:
        test.obsm['spatial'] = merged_df[["x", "y"]].to_numpy().copy()
    elif 'X' in merged_df.columns:
        test.obsm['spatial'] = merged_df[["X", "Y"]].to_numpy().copy()
    else:
        Error("No locations founded in cell_locations dataframe")
        
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
    

def PlotVisiumGene(generated_cells,gene,size=0.8,alpha_img=0.3,perc=0.00,palette='rocket_r', vis_index = None, vis_index_only = None, colorbar_loc='right',title=None,ax=None):
    test = generated_cells.copy()
    if not 'spatial' in test.obsm.keys():
        if 'x' in test.obs.columns:
            test.obsm['spatial'] = test.obs[["x", "y"]].to_numpy()
        elif 'X' in test.obs.columns:
            test.obsm['spatial'] = test.obs[["X", "Y"]].to_numpy()
        else:
            raise Error("No locations founded in obs")
        spot_size = 30
    else:
        spot_size = None
        
    try:
        tmp = test[:,test.var.index==gene].X.toarray().copy()
    except:
        tmp = test[:,test.var.index==gene].X.copy()
        
    tmp = np.clip(tmp,np.quantile(tmp,perc),np.quantile(tmp,1-perc))
    tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
    if vis_index is not None:
        tmp[~vis_index] = None
    if vis_index_only is not None:
        test = test[vis_index_only]
        tmp = tmp[vis_index_only]
    test.obs[gene+'_visual'] = tmp
    if title is None:
        title='${}$'.format(gene)
    
    sc.pl.spatial(
        test,
        color=gene+'_visual',
        size=size,
        spot_size=spot_size,
        frameon=False,
        alpha_img=alpha_img,
        show=False,
        na_color='#e3dede',
        cmap=palette,
        na_in_legend=False,
        colorbar_loc=colorbar_loc,
        ax=ax,title=title
    )
    
    ax.axes.xaxis.label.set_visible(False)
    ax.axes.yaxis.label.set_visible(False)
        

    
    
def ConcatCells(s=0,e=4000,inter=500,es=3813,file_path='/home/share/xwanaf/sour_sep/Visium/data',prefix='',suffix='_noMarker_heart_Sig1.6_1e-05.h5ad',obs=None):
    spot_id_range = np.concatenate((np.arange(s,e,inter), np.array([es])))
    slices = []
    for spot_id in np.concatenate((spot_id_range[:-1][None, :], spot_id_range[1:][None, :]), axis = 0).T:
        sep = '_'
        file_name = prefix+sep.join(str(e) for e in spot_id) + suffix
        slices.append(sc.read(os.path.join(file_path, file_name)))
        
    x_decon_adata = slices[0].concatenate(
        slices[1:],
        batch_key="_",
        uns_merge="unique",
        index_unique=None
        )
#     y_sep_adata.obs = y_sep_adata.obs.reset_index(drop=True)
#     y_sep_adata.obs = obs
#     y_sep_adata.obs.index = y_sep_adata.obs.index.astype(str)
    return x_decon_adata


# def reorder(file1, file2):
#     file1 = file1.copy()
#     file2 = file2.copy()
#     reorder_index = np.zeros(file1.shape[0])
#     for i, spot_id in enumerate(np.unique(file1.obs['spot_id'])):
#         spot_id_index = np.array(file1.obs['spot_id'] == spot_id)   
#         file1_spot_id = file1[spot_id_index].copy()
#         file2_spot_id = file2[spot_id_index].copy()
#         assert np.array_equal(np.array(file1_spot_id.obs['spot_id']), np.array(file2_spot_id.obs['spot_id']))
#         assert np.array_equal(np.array(file1_spot_id.obs['cell_type_label']), np.array(file2_spot_id.obs['cell_type_label']))

#         try:
#             file1_spot_id.X = file1_spot_id.X.toarray()
#             file2_spot_id.X = file2_spot_id.X.toarray()
#         except:
#             pass

#         spot_ct = np.array(file1_spot_id.obs['cell_type_label'])
#         vals, counts = np.unique(spot_ct, return_counts = True)
#         spot_reorder_index = np.zeros(spot_ct.shape[0], dtype = int)
#         for j, val in enumerate(vals):
#             ct_index = np.array(spot_ct == val)
#             d_mtx = cosine_similarity(file2_spot_id.X[ct_index], file1_spot_id.X[ct_index])
#             nrow, ncol = d_mtx.shape
#             row, col = np.unravel_index(np.argsort(d_mtx.ravel())[::-1], d_mtx.shape)
#             row_index = []
#             col_index = []
#             assert nrow == ncol
#             for k in range(min(nrow, ncol)):
#                 if k == 0:
#                     row_index.append(row[k])
#                     col_index.append(col[k])

#                 else:
#                     index1 = (~np.isin(row, row_index)) & (~np.isin(col, col_index))
#                     row_index.append(row[index1][0])
#                     col_index.append(col[index1][0])
#             c_index = np.array(row_index)[np.argsort(np.array(col_index))]
#             spot_reorder_index[ct_index] = np.arange(file1.shape[0])[spot_id_index][ct_index][c_index]

#         reorder_index[spot_id_index] = spot_reorder_index
#         reorder_index = reorder_index.astype(int)
#     return reorder_index

def reorder(file1, file2, spot_id_name = 'spot_index', cell_type_colname = 'cell_type_label'):
    file1 = file1.copy()
    file2 = file2.copy()
    
    reorder_index = np.zeros(file1.shape[0])
    for i, spot_id in enumerate(np.unique(file2.obs[spot_id_name])):
        spot_id_index = np.array(file2.obs[spot_id_name] == spot_id)   
        file1_spot_id = file1[spot_id_index].copy()
        file2_spot_id = file2[spot_id_index].copy()
#         assert np.array_equal(np.array(file1_spot_id.obs['spot_id']), np.array(file2_spot_id.obs['spot_id']))
#         assert np.array_equal(np.array(file1_spot_id.obs['cell_type_label']), np.array(file2_spot_id.obs['cell_type_label']))

        try:
            file1_spot_id.X = file1_spot_id.X.toarray()
            file2_spot_id.X = file2_spot_id.X.toarray()
        except:
            pass

        spot_ct = np.array(file2_spot_id.obs[cell_type_colname])
        vals, counts = np.unique(spot_ct, return_counts = True)
        spot_reorder_index = np.zeros(spot_ct.shape[0], dtype = int)
        for j, val in enumerate(vals):
            ct_index = np.array(spot_ct == val)
            d_mtx = cosine_similarity(file2_spot_id.X[ct_index], file1_spot_id.X[ct_index])
            nrow, ncol = d_mtx.shape
            row, col = np.unravel_index(np.argsort(d_mtx.ravel())[::-1], d_mtx.shape)
            row_index = []
            col_index = []
            assert nrow == ncol
            for k in range(min(nrow, ncol)):
                if k == 0:
                    row_index.append(row[k])
                    col_index.append(col[k])

                else:
                    index1 = (~np.isin(row, row_index)) & (~np.isin(col, col_index))
                    row_index.append(row[index1][0])
                    col_index.append(col[index1][0])
            c_index = np.array(row_index)[np.argsort(np.array(col_index))]
            spot_reorder_index[ct_index] = np.arange(file1.shape[0])[spot_id_index][ct_index][c_index]

        reorder_index[spot_id_index] = spot_reorder_index
        reorder_index = reorder_index.astype(int)
    return reorder_index


