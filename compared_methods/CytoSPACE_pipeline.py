# basic imports
import pandas as pd
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import tempfile
import os

# Example
# python ~/bio/SpatialBenchmarking/Codes/Deconvolution/CytoSPACE_pipeline.py \
# /home/share/xiaojs/spatial/sour_sep/mouce_brain_VISp/Ref_scRNA_VISp_qc2.h5ad \
# /home/share/xiaojs/spatial/sour_sep/tangram/merfish/MERFISH_mop.h5ad \
# cell_subclass \
# /home/share/xwanaf/sour_sep/simulation/cytospace_test


sc_file_path = sys.argv[1]
spatial_file_path = sys.argv[2]
celltype_key = sys.argv[3]
output_file_path = sys.argv[4]


adata_sc = sc.read(sc_file_path)
adata_sc.var.index = [_.lower() for _ in adata_sc.var.index]
adata_sc.var_names_make_unique()
adata_sc.obs_names_make_unique()

cell_types = np.array(adata_sc.obs[celltype_key].unique())

adata_st = sc.read(spatial_file_path)
adata_st.obs['cell_count'] = 1
adata_st.var.index = [_.lower() for _ in adata_st.var.index]
adata_st.var_names_make_unique()
adata_st.obs_names_make_unique()

if adata_st.X.max()<30:
    try:
        adata_st.X = np.exp(adata_st.X) - 1
    except:
        adata_st.X = np.exp(adata_st.X.toarray()) - 1
        
if adata_sc.X.max()<30:
    try:
        adata_sc.X = np.exp(adata_sc.X) - 1
    except:
        adata_sc.X = np.exp(adata_sc.X.toarray()) - 1

sel_genes = list(set(adata_st.var.index.values)&set(adata_sc.var.index.values))

adata_sc = adata_sc[:,sel_genes]
adata_st = adata_st[:,sel_genes]

print(adata_st.shape, adata_sc.shape)
print(adata_st.X.max(),adata_sc.X.max())


scRef_express_df_file = tempfile.NamedTemporaryFile(suffix='.txt')
scRef_express_df_file_name = scRef_express_df_file.name
scRef_express_df = adata_sc.to_df().astype(int).T
scRef_express_df.index.name = 'GENES'
scRef_express_df.to_csv(scRef_express_df_file_name,sep='\t')

scRef_celltype_file = tempfile.NamedTemporaryFile(suffix='.txt')
scRef_celltype_file_name = scRef_celltype_file.name
scRef_celltype = adata_sc.obs[[celltype_key]]
scRef_celltype.columns = ['CellType']
scRef_celltype.index.name = 'Cell IDs'
scRef_celltype.to_csv(scRef_celltype_file_name,sep='\t')


sp_express_df_file = tempfile.NamedTemporaryFile(suffix='.txt')
sp_express_df_file_name = sp_express_df_file.name
sp_express_df = adata_st.to_df().astype(int).T
sp_express_df.index.name = 'GENES'
sp_express_df.to_csv(sp_express_df_file_name,sep='\t')


try:
    adata_st.obs['row'] = adata_st.obs['X']
    adata_st.obs['col'] = adata_st.obs['Y']
except:
    adata_st.obs['row'] = adata_st.obs['x']
    adata_st.obs['col'] = adata_st.obs['y']
sp_coord_df = adata_st.obs[['row','col']]
sp_coord_df.index.name = 'SpotID'
sp_coord_df_file = tempfile.NamedTemporaryFile(suffix='.txt')
sp_coord_df_file_name = sp_coord_df_file.name
sp_coord_df.to_csv(sp_coord_df_file_name,sep='\t')

sp_frac_file = tempfile.NamedTemporaryFile(suffix='.txt')
sp_frac_file_name = sp_frac_file.name

Rcom = f'Rscript /home/share/xiaojs/software/cytospace/get_cellfracs_seuratv3.R \
{scRef_express_df_file_name} {scRef_celltype_file_name} {sp_express_df_file_name} {sp_frac_file_name}'
print(Rcom)
os.system(Rcom)

Pcom = f'cytospace -sp {scRef_express_df_file_name} -ctp {scRef_celltype_file_name} -stp {sp_express_df_file_name} \
-cp {sp_coord_df_file_name} -ctfep {sp_frac_file_name} -o {output_file_path}'
print(Pcom)
os.system(Pcom)


print('Organizing results')

df = adata_st.uns['cell_locations']
alignres_cytospace = pd.read_csv(output_file_path+'/assigned_locations.csv')
alignres_cytospace['CellType:CellIndex'] = alignres_cytospace.apply(lambda x:x['CellType']+':'+x['OriginalCID'], axis=1)
alignres_cytospace = alignres_cytospace.groupby('SpotID')['CellType:CellIndex'].agg(list)

df = df.merge(alignres_cytospace,left_on='spot_index',right_index=True)
df['cytospace_result'] = df.apply(lambda x:x['CellType:CellIndex'][int(x['cell_index'].split('_')[-1])] \
                                     if int(x['cell_index'].split('_')[-1]) < len(x['CellType:CellIndex']) else None, axis=1)
df = df.loc[~df['cytospace_result'].isnull()]
df['cell_type_cytospace'] = df['cytospace_result'].apply(lambda x:x.split(':')[0])
df['cell_index_cytospace'] = df['cytospace_result'].apply(lambda x:x.split(':')[1])

#df.to_csv('/'.join(output_file_path.split('/')[:-1]) + '/CytoSPACE_SinglecellMapping_result.txt')
df.to_csv(output_file_path + '/CytoSPACE_SinglecellMapping_result.txt')

ss_res = df.copy()
ss_res = pd.pivot_table(ss_res[['spot_index','cell_type_cytospace']],index=['spot_index'],columns=['cell_type_cytospace'], aggfunc=len, fill_value=0).reset_index()
ss_res = ss_res.set_index("spot_index")
ss_res = pd.DataFrame(ss_res.values/ss_res.values.sum(1)[:,None],columns=ss_res.columns,index=ss_res.index)
ss_res = adata_st.obs[[]].merge(ss_res,left_index=True,right_index=True,how='left')
ss_res = ss_res.fillna(0)
#ss_res.to_csv('/'.join(output_file_path.split('/')[:-1]) + '/CytoSPACE_result.txt')
ss_res.to_csv(output_file_path + '/CytoSPACE_result.txt')