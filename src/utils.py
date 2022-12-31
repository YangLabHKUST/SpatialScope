import logging
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scanpy as sc
import os 
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    

def PlotVisiumGene(generated_cells,gene,size=0.8,alpha_img=0.3,perc=0.00,palette='rocket_r', vis_index = None, vis_index_0=False, vis_index_only = None, colorbar_loc='right',title=None,ax=None):
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
        if vis_index_0:
            tmp[~vis_index] = 0
        else:
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


def VisualDE(ad_sc,gene,perc=0.02,Sparse=False,title=None):
    try:
        tmp = ad_sc[:,ad_sc.var.index==gene].X.toarray().copy()
    except:
        tmp = ad_sc[:,ad_sc.var.index==gene].X.copy()
    tmp = np.clip(tmp,np.quantile(tmp,perc),np.quantile(tmp,1-perc))
    tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
    if Sparse:
        tmp[tmp==0] = -1
    ad_sc.obs['gene_visual'] = tmp
    plt.subplots(figsize=(10,8.5))
    sns.scatterplot(data=ad_sc.obs, x="x", y="y",s=10,hue='gene_visual',palette='rocket_r',legend=False,alpha=0.9)#rocket_r
    #plt.gca().invert_yaxis()
    plt.axis('off')
    if title is None:
        plt.title('${}$'.format(gene),fontsize=30)
    else:
        plt.title(title,fontsize=30)
    plt.show()
    
    

def VisualscDE(ad_sc,gene):
    try:
        tmp = ad_sc[:,gene].X.A
    except:
        tmp = ad_sc[:,gene].X
    tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
    ad_sc.obs['visual_gene'] = tmp
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    sc.pl.umap(
        ad_sc, color="visual_gene", size=10, frameon=False, show=False, ax=axs, cmap='rocket_r',title=f'${gene}$'
    )
    plt.tight_layout()
    

# sns.set_context('paper',font_scale=2) 
# PlotLRGenes(generated_cells,'SLIT3','ROBO1','Smooth_muscle_cells','Fibroblast') 
def PlotLRGenes(generated_cells,gene_L,gene_R,celltype_L,celltype_R,perc=0.0,palette_L='rocket_r',palette_R='Greens',title='',s=50,ax=None,invertY=True):
    generated_cells_vis_L = generated_cells.copy()
    L_ct_index = generated_cells_vis_L.obs['discrete_label_ct'].isin(np.array([celltype_L]))

    generated_cells_vis_R = generated_cells.copy()
    R_ct_index = generated_cells_vis_R.obs['discrete_label_ct'].isin(np.array([celltype_R]))
    
    test = generated_cells.copy()
    if not 'spatial' in test.obsm.keys():
        test.obsm['spatial'] = test.obs[["x", "y"]].to_numpy()
    try:
        tmp = test[:,test.var.index==gene_L].X.toarray().copy()
    except:
        tmp = test[:,test.var.index==gene_L].X.copy()        
    tmp = np.clip(tmp,np.quantile(tmp,perc),np.quantile(tmp,1-perc))
    tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
    tmp[~np.array(L_ct_index)] = 0
    test.obs[gene_L+'_visual'] = tmp
    
    try:
        tmp = test[:,test.var.index==gene_R].X.A.copy()
    except:
        tmp = test[:,test.var.index==gene_R].X.copy()
    tmp = np.clip(tmp,np.quantile(tmp,perc),np.quantile(tmp,1-perc))
    tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min()) #+ 0.15
    tmp[~np.array(R_ct_index)] = 0
    test.obs[gene_R+'_visual'] = tmp
    test.obs['test'] = 'test'
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(11, 8),dpi=100,facecolor='#fafafa')
    sns.scatterplot(data=test.obs[L_ct_index],x='x',y='y',s=s,hue=gene_L+'_visual',ax=ax,palette=palette_L,alpha=0.8)
    sns.scatterplot(data=test.obs[R_ct_index],x='x',y='y',s=s*0.5,style='test',markers=['s'],hue=gene_R+'_visual',ax=ax,palette=palette_R,alpha=0.6)
    margin = (test.obs.x.max()-test.obs.x.min())*0.03
    plt.xlim([test.obs.x.min()-margin,test.obs.x.max()+margin])
    margin = (test.obs.y.max()-test.obs.y.min())*0.03
    plt.ylim([test.obs.y.min()-margin,test.obs.y.max()+margin])
    plt.axis('off')
    plt.title(title)
    ax.get_legend().remove()
    if invertY:
        plt.gca().invert_yaxis()
    plt.show()
    
    
def ScanpyDowngradeSave(f):
    ad_sp = sc.read(f)
    try:
        np.save(f+'.X.npy',ad_sp.X)
    except:
        np.save(f+'.X.npy',ad_sp.X.A)
    ad_sp.obs.to_csv(f+'.obs.csv')
    ad_sp.var.to_csv(f+'.var.csv')
    
def ScanpyDowngradeLoad(f):
    sc_rna = np.load(f + '.X.npy')
    sc_rna_obs = pd.read_csv(f + '.obs.csv', index_col = 0)
    sc_rna_var = pd.read_csv(f + '.var.csv', index_col = 0)
    sc_rna_adata = AnnData(X = sc_rna, obs = sc_rna_obs, var = sc_rna_var)
    return sc_rna_adata



def ComputeSigma(sc_data_process_marker,cell_class_column):
    if 'Marker' in sc_data_process_marker.var.columns:
        sc_data_process_marker = sc_data_process_marker[:,sc_data_process_marker.var['Marker']]
    df_category = sc_data_process_marker.obs[[cell_class_column]].astype('category').apply(lambda x: x.cat.codes)
    cell_type_array_code = np.array(df_category[cell_class_column]) 
    q = len(np.unique(cell_type_array_code))
    data = sc_data_process_marker.X.toarray()
    data_Ed = []
    for i in range(q):
        index = cell_type_array_code == i
        data_Ed.append(pdist(data[index]).max())
    return data_Ed


def PlotVisiumCellType(ad_sp,weight,cell_type,size=1.5,ax=None):
    test = ad_sp.copy()
    test.obs = test.obs.merge(weight,left_index=True,right_index=True,how='left')
    if not 'spatial' in test.obsm.keys():
        test.obsm['spatial'] = test.obs[["x", "y"]].to_numpy()
    
    test.obs.loc[pd.isnull(test.obs[cell_type]),cell_type] = 0
    test.obs[cell_type] = (test.obs[cell_type]-test.obs[cell_type].min())/(test.obs[cell_type].max()-test.obs[cell_type].min())
    test.obs[cell_type] = test.obs['cell_count']*test.obs[cell_type]
    sc.pl.spatial(
        test,
        color=cell_type,
        size=size,
        frameon=False,
        alpha_img=0.,
        show=False,
        cmap='rocket_r',
        ax=ax,colorbar_loc=None,title=''
    )
    
    ax.axes.xaxis.label.set_visible(False)
    ax.axes.yaxis.label.set_visible(False)
    
    
def GenerateCellTypeProportion(ss_res):
    ss_res = pd.pivot_table(ss_res[['spot_index','discrete_label_ct']],index=['spot_index'],columns=['discrete_label_ct'], aggfunc=len, fill_value=0).reset_index()
    ss_res = ss_res.set_index("spot_index")
    ss_res = pd.DataFrame(ss_res.values/ss_res.values.sum(1)[:,None],columns=ss_res.columns,index=ss_res.index)
    return ss_res


def PlotSampledData(gen_sample_adata,sc_data_process,cell_type_key,palette=None,scale=True):   
    gen_sample_adata = gen_sample_adata.copy()
    sc_data_process = sc_data_process.copy()
    sc_data_process_marker = sc_data_process[:,sc_data_process.var['Marker']]
    if sc_data_process_marker.X.max()>30:
        print(f'Maximum value: {sc_data_process_marker.X.max()}, need to run log1p')
        sc.pp.log1p(sc_data_process_marker)
    adata_snrna_raw_small_raw = sc_data_process_marker
    adata_snrna_raw_small_raw.obs_names_make_unique()
    adata_snrna_raw_small_raw.var_names_make_unique()
    if scale:
        sc.pp.scale(adata_snrna_raw_small_raw,max_value=10)
        sc.pp.scale(gen_sample_adata,max_value=10)
    sc.tl.pca(adata_snrna_raw_small_raw, n_comps=20,use_highly_variable=False) 
    
    pcs = adata_snrna_raw_small_raw.varm['PCs']
    gen_sample_adata.obsm['X_pca'] = (gen_sample_adata.X - adata_snrna_raw_small_raw.X.mean(0)) @ pcs
    gen_sample_adata.varm['PCs'] = pcs

    adata_all = adata_snrna_raw_small_raw.concatenate(gen_sample_adata)
    adata_all.varm['PCs'] = pcs
    adata_all.raw = adata_all

    sc.pp.neighbors(adata_all, n_neighbors=15, metric = 'cosine', n_pcs=20)
    sc.tl.umap(adata_all, min_dist = 0.5, spread = 1., maxiter = 80)

    df = pd.DataFrame(adata_all.obsm['X_umap'], columns = ['x', 'y'])
    color_mu = ['All cells from the single-cell reference data' for i in range(adata_all.shape[0]-gen_sample_adata.shape[0])]+\
                ['Generated cells by SpatialScope' for i in range(gen_sample_adata.shape[0])]
    df['Label'] = color_mu

    fig, ax = plt.subplots(figsize=(10,10),dpi=150)
    sns.scatterplot(data=df, x="x", y="y", hue="Label", s=2, palette = ['#a1c9f4','#c44e52'],ax=ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc="upper left",framealpha=0,markerscale=2,handletextpad=0.2)
    plt.show()        
    fig, ax = plt.subplots(figsize=(10,10),dpi=150)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    sc.pl.umap(adata_all, color=[cell_type_key], size=15,
               color_map = 'RdPu', ncols = 1, na_in_legend=False, legend_loc='on data',
               legend_fontsize=10,title='',palette=palette,ax=ax)
    plt.show()  
    return adata_all




def plot_genes_sc(
    genes, 
    adata_measured, 
    adata_predicted,
    x="x",
    y = "y",
    spot_size=None, 
    scale_factor=None, 
    cmap="inferno", 
    perc=0,
    return_figure=False
):
    adata_measured = adata_measured.copy()
    adata_predicted = adata_predicted.copy()
    # remove df_plot in obs 
    adata_measured.obs.drop(
        ["{} (measured)".format(gene) for gene in genes],
        inplace=True,
        errors="ignore",
        axis=1,
    )
    adata_predicted.obs.drop(
        ["{} (predicted)".format(gene) for gene in genes],
        inplace=True,
        errors="ignore",
        axis=1,
    )

    # prepare adatas
    convert_adata_array(adata_measured)

    adata_measured.var.index = [g.lower() for g in adata_measured.var.index]
    adata_predicted.var.index = [g.lower() for g in adata_predicted.var.index]

    adata_predicted.obsm = adata_measured.obsm
    adata_predicted.uns = adata_measured.uns

    # remove previous df_plot in obs
    adata_measured.obs.drop(
        ["{} (measured)".format(gene) for gene in genes],
        inplace=True,
        errors="ignore",
        axis=1,
    )
    adata_predicted.obs.drop(
        ["{} (predicted)".format(gene) for gene in genes],
        inplace=True,
        errors="ignore",
        axis=1,
    )

    # construct df_plot
    data = []
    for ix, gene in enumerate(genes):
        if gene not in adata_measured.var.index:
            data.append(np.zeros_like(np.array(adata_measured[:, 0].X).flatten()))
        else:
            data.append(np.array(adata_measured[:, gene].X).flatten())

    df = pd.DataFrame(
        data=np.array(data).T, columns=genes, index=adata_measured.obs.index,
    )
    construct_obs_plot(df, adata_measured, suffix="measured")

    df = pd.DataFrame(
        data=np.array(adata_predicted[:, genes].X),
        columns=genes,
        index=adata_predicted.obs.index,
    )
    construct_obs_plot(df, adata_predicted, perc=perc, suffix="predicted")

    fig = plt.figure(figsize=(7, len(genes) * 3.5))
    gs = GridSpec(len(genes), 2, figure=fig)
    
    #non visium data
    if 'spatial' not in adata_measured.obsm.keys():
        #add spatial coordinates to obsm of spatial data 
        coords = [[x,y] for x,y in zip(adata_measured.obs[x].values,adata_measured.obs[y].values)]
        adata_measured.obsm['spatial'] = np.array(coords)
        coords = [[x,y] for x,y in zip(adata_predicted.obs[x].values,adata_predicted.obs[y].values)]
        adata_predicted.obsm['spatial'] = np.array(coords)

    if ("spatial" not in adata_measured.uns.keys()) and (spot_size==None and scale_factor==None):
        raise ValueError("Spot Size and Scale Factor cannot be None when ad_sp.uns['spatial'] does not exist")
        
    for ix, gene in enumerate(genes):
        ax_m = fig.add_subplot(gs[ix, 0])
        sc.pl.spatial(
            adata_measured,
            spot_size=spot_size,
            scale_factor=scale_factor,
            color=["{} (measured)".format(gene)],
            frameon=False,
            ax=ax_m,
            show=False,
            cmap=cmap,
            title="{} (measured)".format(gene),colorbar_loc=None
        )
        ax_p = fig.add_subplot(gs[ix, 1])
        sc.pl.spatial(
            adata_predicted,
            spot_size=spot_size,
            scale_factor=scale_factor,
            color=["{} (predicted)".format(gene)],
            frameon=False,
            ax=ax_p,
            show=False,
            cmap=cmap,
            title='Predicted',colorbar_loc=None
        )
        
    #     sc.pl.spatial(adata_measured, color=['{} (measured)'.format(gene) for gene in genes], frameon=False)
    #     sc.pl.spatial(adata_predicted, color=['{} (predicted)'.format(gene) for gene in genes], frameon=False)
    fig.tight_layout()#(pad=.8)
    # remove df_plot in obs
    adata_measured.obs.drop(
        ["{} (measured)".format(gene) for gene in genes],
        inplace=True,
        errors="ignore",
        axis=1,
    )
    adata_predicted.obs.drop(
        ["{} (predicted)".format(gene) for gene in genes],
        inplace=True,
        errors="ignore",
        axis=1,
    )
    if return_figure==True:
        return fig

def convert_adata_array(adata):
    if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
        adata.X = adata.X.toarray()
        
def construct_obs_plot(df_plot, adata, perc=0, suffix=None):
    # clip
    df_plot = df_plot.clip(df_plot.quantile(perc), df_plot.quantile(1 - perc), axis=1)

    # normalize
    df_plot = (df_plot - df_plot.min()) / (df_plot.max() - df_plot.min())

    if suffix:
        df_plot = df_plot.add_suffix(" ({})".format(suffix))
    adata.obs = pd.concat([adata.obs, df_plot], axis=1)

from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from matplotlib.gridspec import GridSpec