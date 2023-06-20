# basic imports
import pandas as pd
import sys
import numpy as np
import pandas as pd
import scanpy as sc

# add `Tangram` to path
import tangram as tg


from anndata import AnnData
import pathlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import skimage
import seaborn as sns
import anndata
from ast import literal_eval
import re
import pickle5 as pickle

def create_segment_cell_df(adata_sp):
    """
    Produces a Pandas dataframe where each row is a segmentation object, columns reveals its position information.
    Args:
        adata_sp (AnnData): spot-by-gene AnnData structure. Must contain obsm.['image_features']
    Returns:
        None.
        Update spatial AnnData.uns['tangram_cell_segmentation'] with a dataframe: each row represents a segmentation object (single cell/nuclei). Columns are 'spot_idx' (voxel id), and 'y', 'x', 'centroids' to specify the position of the segmentation object.
        Update spatial AnnData.obsm['trangram_spot_centroids'] with a sequence
    """

    if "image_features" not in adata_sp.obsm.keys():
        raise ValueError(
            "Missing parameter for tangram deconvolution. Run `sqidpy.im.calculate_image_features`."
        )

    
    centroids = adata_sp.obsm["image_features"][["segmentation_centroid", 'segmentation_centroid_index']].copy()
    centroids["centroids_idx"] = [
        np.array([f"{k}_{j}" for j in np.arange(i)], dtype="object")
        for k, i in zip(
            adata_sp.obs.index.values,
            adata_sp.obsm["image_features"]["segmentation_label"],
        )
    ]
    centroids_idx = centroids.explode("centroids_idx")
    centroids_coords = centroids.explode("segmentation_centroid")
    centroids_given_idx = centroids.explode("segmentation_centroid_index")
    
    segmentation_df = pd.DataFrame(
        centroids_coords["segmentation_centroid"].to_list(),
        columns=["y", "x"],
        index=centroids_coords.index,
    )
    segmentation_df["centroids"] = centroids_idx["centroids_idx"].values
    segmentation_df["given_centroids_index"] = centroids_given_idx["segmentation_centroid_index"].values
    segmentation_df.index.set_names("spot_idx", inplace=True)
    segmentation_df.reset_index(
        drop=False, inplace=True,
    )

    adata_sp.uns["tangram_cell_segmentation"] = segmentation_df
    adata_sp.obsm["tangram_spot_centroids"] = centroids["centroids_idx"]
    
    
    

sc_file_path = sys.argv[1]
spatial_file_path = sys.argv[2]
celltype_key = sys.argv[3]
output_file_path = sys.argv[4]

def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

ad_sp = anndata.read_h5ad(spatial_file_path)
ad_sc = anndata.read_h5ad(sc_file_path)
cell_locations = ad_sp.uns['cell_locations'].copy()


if 'x' in ad_sp.obs.columns:
    ad_sp.obs['X'] = ad_sp.obs['x']
    ad_sp.obs['Y'] = ad_sp.obs['y']
    cell_locations['X'] = cell_locations['x']
    cell_locations['Y'] = cell_locations['y']
if 'cell_nums' in ad_sp.obs.columns:
    ad_sp.obs['cell_num'] = ad_sp.obs['cell_nums']
    cell_locations['cell_num'] = cell_locations['cell_nums']

ad_sp.obsm['spatial'] = ad_sp.obs[['X','Y']].values
ad_sp_obsm_image_features = pd.DataFrame()
ad_sp_obsm_image_features['segmentation_label'] = ad_sp.obs['cell_num'].values
# ad_sp_obsm_image_features['segmentation_centroid'] = [[[1,1] for i in range(int(ad_sp.obs['cell_num'].values[_]))] for _ in range(ad_sp.obs.shape[0])]
ad_sp_obsm_image_features['segmentation_centroid'] = [cell_locations.loc[:,np.array(['X', 'Y'])].values[cell_locations['spot_index_int'] == _].tolist() for _ in np.unique(cell_locations['spot_index_int'])]
ad_sp_obsm_image_features['segmentation_centroid_index'] = [cell_locations.index.values[cell_locations['spot_index_int'] == _].tolist() for _ in np.unique(cell_locations['spot_index_int'])]
# ad_sp_obsm_image_features.index=[str(_) for _ in range(ad_sp.obs.shape[0])]
ad_sp_obsm_image_features.index=ad_sp.obs.index
ad_sp.obsm['image_features'] = ad_sp_obsm_image_features


# ad_sp.uns['cell_locations'] = cell_locations
sp_index = np.array(f7(ad_sp.uns['cell_locations'].copy()['spot_index']))
ad_sp = ad_sp[sp_index,:]

# cell_locations = pd.read_csv('cell_locations.csv', index_col = 0)
unqiue_cell_locations = cell_locations.iloc[np.unique(cell_locations['spot_index'], return_index = True)[1]]

cell_num_df = pd.DataFrame(data = np.zeros(ad_sp.shape[0]), index = ad_sp.to_df().index, columns = ['X'])
for one_spot_index in unqiue_cell_locations['spot_index']:
    cell_num_df.loc[one_spot_index,'X'] = unqiue_cell_locations.loc[(unqiue_cell_locations['spot_index'] == one_spot_index),'cell_num'].values[0]

ad_sp.obs['cell_count'] = cell_num_df.values.squeeze()


sc.pp.normalize_total(ad_sc)
celltype_counts = ad_sc.obs[celltype_key].value_counts()
celltype_drop = celltype_counts.index[celltype_counts < 2]
print(f'Drop celltype {list(celltype_drop)} contain less 2 sample')
ad_sc = ad_sc[~ad_sc.obs[celltype_key].isin(celltype_drop),].copy()
sc.tl.rank_genes_groups(ad_sc, groupby=celltype_key, use_raw=False)
markers_df = pd.DataFrame(ad_sc.uns["rank_genes_groups"]["names"]).iloc[0:200, :]
print(markers_df)
genes_sc = np.unique(markers_df.melt().value.values)
print(genes_sc)
genes_st = ad_sp.var_names.values
genes = list(set(genes_sc).intersection(set(genes_st)))

tg.pp_adatas(ad_sc, ad_sp, genes=genes)

ad_map = tg.map_cells_to_space(
    ad_sc,
    ad_sp,
    mode="constrained",
    target_count=ad_sp.obs.cell_count.sum(),
    density_prior=np.array(ad_sp.obs.cell_count) / ad_sp.obs.cell_count.sum(),
    num_epochs=1000,
    device="cuda:0",
#     device='cpu',
)

tg.project_cell_annotations(ad_map, ad_sp, annotation=celltype_key)
annotation_list = list(pd.unique(ad_sc.obs[celltype_key]))
# tg.plot_cell_annotation_sc(ad_sp, annotation_list, perc=0.02)

create_segment_cell_df(ad_sp)
ad_sp.uns["tangram_cell_segmentation"].head()

tg.count_cell_annotations(
    ad_map,
    ad_sc,
    ad_sp,
    annotation=celltype_key,
)
ad_sp.obsm["tangram_ct_count"].head()

adata_segment = tg.deconvolve_cell_annotations(ad_sp)
adata_segment.obs['spot_index'] = np.array(['_'.join(_.split('_')[:2]) for _ in adata_segment.obs['centroids']])
# adata_segment.obs['align'] = [spot_index + '--' + cluster for (spot_index, cluster) in zip(np.array(adata_segment.obs['spot_index']), np.array(adata_segment.obs['cluster']))]
adata_segment.obs.head()


adata_map = ad_map.copy()
adata_sc = ad_sc.copy()
adata_sp = ad_sp.copy()
annotation=celltype_key
threshold=0.5

xs = adata_sp.obsm["spatial"][:, 1]
ys = adata_sp.obsm["spatial"][:, 0]
cell_count = adata_sp.obsm["image_features"]["segmentation_label"]

df_segmentation = adata_sp.uns["tangram_cell_segmentation"]
centroids = adata_sp.obsm["tangram_spot_centroids"]

# create a dataframe
df_vox_cells = df_vox_cells = pd.DataFrame(
    data={"x": xs, "y": ys, "cell_n": cell_count, "centroids": centroids},
    index=list(adata_sp.obs.index),
)
# get the most probable voxel for each cell
resulting_voxels = np.argmax(adata_map.X, axis=1)

# create a list with filtered cells and the voxels where they have been placed with the
# highest probability a cell i is filtered if F_i > threshold'
if "F_out" in adata_map.obs.keys():
    filtered_voxels_to_types = [
        (adata_sp.obs.index[j], adata_sc.obs[annotation][k], k)
        for i, j, k in zip(
            adata_map.obs["F_out"], resulting_voxels, range(len(adata_sc))
        )
        if i > threshold
    ]

    vox_ct = filtered_voxels_to_types

else:
    vox_ct = [(resulting_voxels, adata_sc.obs[annotation])]


map_sc_to_sp_df = pd.DataFrame(filtered_voxels_to_types)
map_sc_to_sp_df['spot_index'] = map_sc_to_sp_df[0]
map_sc_to_sp_df['cluster'] = map_sc_to_sp_df[1]
map_sc_to_sp_df['sc_index'] = map_sc_to_sp_df[2]
map_sc_to_sp_df = map_sc_to_sp_df.iloc[:,3:]
# map_sc_to_sp_df['align'] = [spot_index + '--' + cluster for (spot_index, cluster) in zip(np.array(map_sc_to_sp_df['spot_index']), np.array(map_sc_to_sp_df['cluster']))]
map_sc_to_sp_df_list = map_sc_to_sp_df.groupby(['spot_index','cluster']).aggregate(lambda tdf: tdf.unique().tolist())
map_sc_to_sp_df_list.head()
adata_segment_df = adata_segment.obs.merge(map_sc_to_sp_df_list,left_on=['spot_index','cluster'],right_index=True)
adata_segment_df['sc_index_idx'] = adata_segment_df.apply(lambda x:x['sc_index'][int(x['centroids'].split('_')[-1])%len(x['sc_index'])],axis=1)
adata_segment_df = adata_segment_df.drop(['sc_index'], axis=1)
adata_segment_df.head(10)

tangram_maped_sc_adata = sc.AnnData(ad_sc.X[np.array(adata_segment_df['sc_index_idx'])], obs=adata_segment_df, var = ad_sc.var)

tangram_maped_sc_adata.write_h5ad(output_file_path + '/tangram_maped_cells.h5ad')