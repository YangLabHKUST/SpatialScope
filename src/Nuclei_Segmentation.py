import scanpy as sc
import squidpy as sq
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
import argparse
import anndata
import pandas as pd
from PIL import Image 
Image.MAX_IMAGE_PIXELS = 1000000000 
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils import *


    
class SpatialScopeNS:
    def __init__(self,tissue,out_dir,ST_Data,Img_Data,prob_thresh,max_cell_number,min_counts):
        self.tissue = tissue
        self.out_dir = out_dir 
        self.ST_Data = ST_Data
        self.Img_Data = Img_Data
        self.prob_thresh = prob_thresh
        self.max_cell_number = max_cell_number
        self.min_counts = min_counts
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        if not os.path.exists(os.path.join(out_dir,tissue)):
            os.mkdir(os.path.join(out_dir,tissue))
        
        self.out_dir = os.path.join(out_dir,tissue)
        loggings = configure_logging(os.path.join(self.out_dir,'logs'))
        self.loggings = loggings 
        self.LoadData(self.ST_Data, self.Img_Data, self.min_counts)


    def LoadData(self, ST_Data, Img_Data, min_counts):
        self.loggings.info(f'Reading spatial data: {ST_Data}')
        sp_adata = anndata.read_h5ad(ST_Data)
        sp_adata.obs_names_make_unique()
        sp_adata.var_names_make_unique()
        self.loggings.info(f'Spatial data shape: {sp_adata.shape}')
        sc.pp.filter_cells(sp_adata, min_counts=min_counts)
        sc.pp.filter_cells(sp_adata, max_counts=20000)
        self.loggings.info(f'Spatial data shape after QC: {sp_adata.shape}')
        self.sp_adata = sp_adata
        
        self.loggings.info(f'Reading image data: {ST_Data}')
        image = plt.imread(Img_Data)
        img = sq.im.ImageContainer(image)
        crop = img.crop_corner(0, 0)
        self.loggings.info(f'Image shape: {crop.shape}')
        
        self.image = crop
        
        
    @staticmethod
    def stardist_2D_versatile_he(img, nms_thresh=None, prob_thresh=None):
        #axis_norm = (0,1)   # normalize channels independently
        axis_norm = (0,1,2) # normalize channels jointly
        # Make sure to normalize the input image beforehand or supply a normalizer to the prediction function.
        # this is the default normalizer noted in StarDist examples.
        img = normalize(img, 1, 99.8, axis=axis_norm)
        model = StarDist2D.from_pretrained('2D_versatile_he')
        labels, _ = model.predict_instances(img, nms_thresh=nms_thresh, prob_thresh=prob_thresh)
        return labels

    
    @staticmethod
    def DissectSegRes(df):
        tmps = []
        for row in df.iterrows():
            if row[1]['segmentation_label'] == 0:
                continue
            for idx,i in enumerate(row[1]['segmentation_centroid']):
                tmps.append(list(i)+[row[0],row[0]+'_{}'.format(idx),row[1]['segmentation_label']])
        return pd.DataFrame(tmps,columns=['x','y','spot_index','cell_index','cell_nums'])  

        
        
    def NucleiSegmentation(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
        StarDist2D.from_pretrained('2D_versatile_he')
        sq.im.segment(
            img=self.image,
            layer="image",
            channel=None,
            method=self.stardist_2D_versatile_he,
            layer_added='segmented_stardist_default',
            prob_thresh=self.prob_thresh)
        self.loggings.info(f"Number of segments: {len(np.unique(self.image['segmented_stardist_default']))}")
        
        # define image layer to use for segmentation
        features_kwargs = {
            "segmentation": {
                "label_layer": "segmented_stardist_default",
                "props": ["label", "centroid"],
                "channels": [1, 2],
            }
        }
        
        # calculate segmentation features
        sq.im.calculate_image_features(
            self.sp_adata,
            self.image,
            layer="image",
            key_added="image_features",
            features_kwargs=features_kwargs,
            features="segmentation",
            mask_circle=True,
        )
        
        df_cells = self.sp_adata.obsm['image_features'].copy().astype(object)
        for row in df_cells.iterrows():
            if row[1]['segmentation_label']>self.max_cell_number:
                df_cells.loc[row[0],'segmentation_label'] = self.max_cell_number
                df_cells.loc[row[0],'segmentation_centroid'] = row[1]['segmentation_centroid'][:self.max_cell_number]
        self.sp_adata.obsm['image_features'] = df_cells
        
        self.sp_adata.obs["cell_count"] = self.sp_adata.obsm["image_features"]["segmentation_label"].astype(int)
        
        fig, axes = plt.subplots(1, 3,figsize=(30,9),dpi=250)
        self.image.show("image", ax=axes[0])
        _ = axes[0].set_title("H&E")
        self.image.show("segmented_stardist_default", cmap="jet", interpolation="none", ax=axes[1])
        _ = axes[1].set_title("Nuclei Segmentation")
        sc.pl.spatial(self.sp_adata, color=["cell_count"], frameon=False, ax=axes[2],title='')
        _ = axes[2].set_title("Cell Count")
        plt.savefig(os.path.join(self.out_dir, 'nuclei_segmentation.png'))
        plt.close()
        
        
        self.sp_adata.uns['cell_locations'] = self.DissectSegRes(self.sp_adata.obsm['image_features'])
        self.sp_adata.obsm['image_features']['segmentation_label'] = self.sp_adata.obsm['image_features']['segmentation_label'].astype(int)
        self.sp_adata.obsm['image_features']['segmentation_centroid'] = self.sp_adata.obsm['image_features']['segmentation_centroid'].astype(str)
        
        self.sp_adata.write_h5ad(os.path.join(self.out_dir, 'sp_adata_ns.h5ad'))



if __name__ == "__main__":
    HEADER = """
    <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    <> 
    <> Nuclei_Segmentation: SpatialScope Nuclei Segmentation
    <> Version: %s
    <> MIT License
    <>
    <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    <> Software-related correspondence: %s or %s
    <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    <> example:
        python <install path>/src/NucleiSegmentation.py \\
            --tissue heart \\
            --out_dir ./output \\
            --ST_Data ./test_data/V1_Human_Heart_spatial.h5ad \\
            --Image_DataDir ./test_data/V1_Human_Heart_image.tif 

    <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>  
    """ 
    
    parser = argparse.ArgumentParser(description='simulation sour_sep')
    parser.add_argument('--tissue', type=str, help='tissue name', default=None)
    parser.add_argument('--out_dir', type=str, help='output path', default=None)
    parser.add_argument('--ST_Data', type=str, help='ST data path', default=None)
    parser.add_argument('--Img_Data', type=str, help='H&E stained image data path', default=None)
    parser.add_argument('--prob_thresh', type=float, help='object probability threshol, decrease this parameter if too many nucleus are missing', default=0.5)
    parser.add_argument('--max_cell_number', type=int, help='maximum cell number per spot', default=20)
    parser.add_argument('--min_counts', type=int, help='minimum UMI count per spot', default=500)
    args = parser.parse_args()
    
    NS = SpatialScopeNS(args.tissue, args.out_dir, args.ST_Data, args.Img_Data, args.prob_thresh, args.max_cell_number, args.min_counts)
    NS.NucleiSegmentation()
    




