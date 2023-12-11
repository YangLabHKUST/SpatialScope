# SpatialScope
A unified approach for integrating spatial and single-cell transcriptomics data by leveraging deep generative models

![SpatialScope](https://github.com/YangLabHKUST/SpatialScope/blob/master/mainfig-flowchat.jpg)

Visit our [documentation](https://spatialscope-tutorial.readthedocs.io/en/latest/) for installation, tutorials, examples and more.

## Installation
``` shell
$ git clone https://github.com/YangLabHKUST/SpatialScope.git
$ cd SpatialScope
$ conda env create -f environment.yml
$ conda activate SpatialScope
$ python setup.py develop
# fix bug of squidpy
$ rsync ./src/_feature_mixin.py ~/.conda/envs/SpatialScope/lib/python3.9/site-packages/squidpy/im/_feature_mixin.py
```
check the installation status
```shell
$ python ./src/Cell_Type_Identification.py -h
# or directly if `python setup.py develop` has been run
$ Cell_Type_Identification.py -h
```


## Reproducibility

We provide source codes for reproducing the SpatialScope analysis in the main text in the `demos` directory.

All relevent materials involved in the reproducing codes are availabel from [here](https://drive.google.com/drive/folders/1PXv_brtr-tXshBVEd_HSPIagjX9oF7Kg?usp=sharing)

+ [Benchmarking Dataset 1](https://github.com/YangLabHKUST/SpatialScope/blob/master/demos/Benchmarking-Dataset_1.ipynb)
+ [Benchmarking Dataset 2](https://github.com/YangLabHKUST/SpatialScope/blob/master/demos/Benchmarking-Dataset_2.ipynb)
+ [Benchmarking Dataset 3](https://github.com/YangLabHKUST/SpatialScope/blob/master/demos/Benchmarking-Dataset_3.ipynb)
+ [Benchmarking Dataset 4](https://github.com/YangLabHKUST/SpatialScope/blob/master/demos/Benchmarking-Dataset_4.ipynb)
+ [Benchmarking Dataset 5](https://github.com/YangLabHKUST/SpatialScope/blob/master/demos/Benchmarking-Dataset_5.ipynb)
+ [Benchmarking Dataset 6](https://github.com/YangLabHKUST/SpatialScope/blob/master/demos/Benchmarking-Dataset_6.ipynb)
+ [Human Heart (Visium, a single slice)](https://github.com/YangLabHKUST/SpatialScope/blob/master/demos/Human-Heart.ipynb)
+ [Mouse Brain (Visium, 3D alignment of multiple slices)](https://github.com/YangLabHKUST/SpatialScope/blob/master/demos/Mouse-Brain.ipynb)
+ [Mouse Cerebellum (Slideseq-V2)](https://github.com/YangLabHKUST/SpatialScope/blob/master/demos/Mouse-Cerebellum-Slideseq.ipynb)
+ [Mouse MOp (MERFISH)](https://github.com/YangLabHKUST/SpatialScope/blob/master/demos/Mouse-MOp-MERFISH.ipynb)


## Quick start for Visium data

We illustrate the usage of SpatialScope using a single slice of 10x Visium human heart data:
- Spatial data: ./demo_data/V1_Human_Heart_spatial.h5ad
- Image data: ./demo_data/V1_Human_Heart_image.tif
- scRNA reference data: ./Ckpts_scRefs/Heart_D2/Ref_Heart_sanger_D2.h5ad
- Pretrained model checkpoint: ./Ckpts_scRefs/Heart_D2/model_5000.pt (**only required for Step3**)

All relevent materials involved in the following example are availabel from [here](https://drive.google.com/drive/folders/1PXv_brtr-tXshBVEd_HSPIagjX9oF7Kg?usp=sharing)

### Step1: Nuclei segmentation

```
python ./src/Nuclei_Segmentation.py --tissue heart --out_dir  ./output  --ST_Data ./demo_data/V1_Human_Heart_spatial.h5ad --Img_Data  ./demo_data/V1_Human_Heart_image.tif
```

Input:

- --out_dir: output directory
- --tissue: output sub-directory
- --ST_Data: ST data file path
- --Img_Data: H&E stained image data file path (require **raw** H&E image with high resolution, about **10000x10000** resolution, **500M** file size)

This step will make `./output/heart` directory, and generate two files:

- Visualization of nuclei segmentation results: nuclei_segmentation.png
- Preprocessed ST data for cell type identification: sp_adata_ns.h5ad (**cell_locations** that contains spatial locations of segmented cells will be added to .uns)

### Step2: Cell type identification

```
python ./src/Cell_Type_Identification.py --tissue heart --out_dir  ./output  --ST_Data ./output/heart/sp_adata_ns.h5ad --SC_Data ./Ckpts_scRefs/Heart_D2/Ref_Heart_sanger_D2.h5ad --cell_class_column cell_type
```

Input:

- --out_dir: output directory
- --tissue: output sub-directory
- --ST_Data: ST data file path (generated in Step 1)
- --SC_Data: single-cell reference data file path (When using your own scRef file, we recommend adding a **Marker** column to the .var to pre-select several thousand marker or highly variable genes as in "./Ckpts_scRefs/Heart_D2/Ref_Heart_sanger_D2.h5ad")
- --cell_class_column: cell class label column in scRef file
  
This step will generate three files:

- Visualization of cell type identification results: estemated_ct_label.png
- Cell type identification results: CellTypeLabel_nu10.csv
- Preprocessed ST data for gene expression decomposition: sp_adata.h5ad

Now we can use the sp_adata.h5ad to visualize the single-cell resolution spatial distribution of different cell types. 


### Step3: Gene expression decomposition

```
python ./src/Decomposition.py --tissue heart --out_dir  ./output --SC_Data ./Ckpts_scRefs/Heart_D2/Ref_Heart_sanger_D2.h5ad --cell_class_column cell_type  --ckpt_path ./Ckpts_scRefs/Heart_D2/model_5000.pt --spot_range 0,100 --gpu 0,1,2,3
```

Input:

- --out_dir: output directory
- --tissue: output sub-directory
- --SC_Data: single-cell reference data file path 
- --cell_class_column: cell class label column in scRef file
- --ckpt_path: model checkpoint file path (As the model checkpoint was trained on scRef file, the checkpoint and scRef file much be matched)
- --spot_range: limited by GPU memory, we can only handle at most about 1000 spots in 4 GPUs at a time. e.g., 0,1000 means 0 to 1000-th spot
- --gpu: Visible GPUs

This step will generate one file:

- Single-cell resolution ST data generated by SpatialScope for spot 0-100: generated_cells_spot0-100.h5ad

## Learning the gene expression distribution of scRNA-seq reference using score-based model

We use four GPUs to train scRNA-seq reference in parallel.
```
python ./src/Train_scRef.py \
--ckpt_path ./Ckpts_scRefs/VISp \
--scRef ./Ckpts_scRefs/VISp/Ref_scRNA_VISp_qc2_2Kgenes.h5ad \
--cell_class_column cell_subclass \
--gpus 0,1,2,3 
```
The checkpoints and sampled psuedo-cells will be saved in ./Ckpts_scRefs/VISp, e.g, model_5000.pt, model_5000.h5ad

Conveniently, we provided the pre-trained checkpoint (Ckpts_scRefs/VISp/model_5000.pt) in here, so you can skip this part


## Contact information

Please contact Xiaomeng Wan (xwanaf@connect.ust.hk), Jiashun Xiao (jxiaoae@connect.ust.hk) or Prof. Can Yang (macyang@ust.hk) if any enquiry.
