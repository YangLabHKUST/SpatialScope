# SpatialScope
A unified approach for integrating spatial and single-cell transcriptomics data by leveraging deep generative models

## Installation
``` shell
$ git clone https://github.com/YangLabHKUST/SpatialScope.git
$ cd SpatialScope
$ conda env create -f environment.yml
$ conda activate SpatialScope
```
check the installation status
```shell
$ python ./src/Cell_Type_Identification.py -h
```

## Quick start

We illustrate the usage of SpatialScope using 10X Visium human heart data:
- Spatial data: ./demo_data/V1_Human_Heart_spatial.h5ad
- Image data: ./demo_data/V1_Human_Heart_image.tif
- scRNA reference data: ./Ckpts_scRefs/Heart_D2/Ref_Heart_sanger_D2.h5ad
- Checkpoint: ./Ckpts_scRefs/Heart_D2/model_600001.pt

### Step1: Nuclei segmentation

```
python ./src/Nuclei_Segmentation.py --tissue heart --out_dir  ./output  --ST_Data ./demo_data/V1_Human_Heart_spatial.h5ad --Img_Data  ./demo_data/V1_Human_Heart_image.tif
```

### Step2: Cell type identification

```
python ./src/Cell_Type_Identification.py --tissue heart --out_dir  ./output  --ST_Data ./output/heart/sp_adata_ns.h5ad --SC_Data ./Ckpts_scRefs/Heart_D2/Ref_Heart_sanger_D2.h5ad --cell_class_column cell_type
```

### Step3: Gene expression decomposition

```
python ./src/Decomposition.py --tissue heart --out_dir  ./output --SC_Data ./Ckpts_scRefs/Heart_D2/Ref_Heart_sanger_D2.h5ad --cell_class_column cell_type  --ckpt_path ./Ckpts_scRefs/Heart_D2/model_600001.pt --spot_range 0,100 --gpu 0,1,2,3
```


## Contact information

Please contact Xiaomeng Wan (xwanaf@connect.ust.hk), Jiashun Xiao (jxiaoae@connect.ust.hk) or Prof. Can Yang (macyang@ust.hk) if any enquiry.
