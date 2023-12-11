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

Now we can use the `sp_adata.h5ad` to visualize the single-cell resolution spatial distribution of different cell types:

```python
ad_sp = sc.read('./output/heart/sp_adata.h5ad')
fig, ax = plt.subplots(1,1,figsize=(12, 8),dpi=100)
PlotVisiumCells(ad_sp,"discrete_label_ct",size=0.3,alpha_img=0.3,lw=0.8,ax=ax)
```
![SpatialScope](https://github.com/YangLabHKUST/SpatialScope/blob/master/human-heart-crop.png)
more details are available in jupyter notebook [Human Heart (Visium, a single slice)](https://github.com/YangLabHKUST/SpatialScope/blob/master/demos/Human-Heart.ipynb).


### Step3: Gene expression decomposition

In Step3, by conditioning on the inferred cell type labels from Step2, SpatialScope performs gene expression decomposition, transforming the spot-level gene expression proﬁle into single-cell resolution. To do this, we ﬁrst learn a [score-based generative model](#Learning-the-gene-expression-distribution-of-scRNA-seq-reference-using-score-based-model) to approximate the expression distribution of different cell types from the single-cell reference data. Then we use the learned model to decompose gene expression from the spot level to the single-cell level, while accounting for the batch effect between single-cell reference and ST data.

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

The scRNA-seq reference `./Ckpts_scRefs/Heart_D2/Ref_Heart_sanger_D2.h5ad` was preprocessed following the standard precedures, more details available in jupyter notebook [Human Heart (Visium, a single slice)](https://github.com/YangLabHKUST/SpatialScope/blob/master/demos/Human-Heart.ipynb). In order to make the distribution learning process more **efficient**, we only learned the gene expression distributions of 2,000 selected highly variable genes.  Besides, we subsampled the number of cells per cell type, up to a maximum of 3,000. 


We use four RTX 2080 Ti GPUs to train scRNA-seq reference in parallel.

```
python ./src/Train_scRef.py \
--ckpt_path ./Ckpts_scRefs/Heart_D2 \
--scRef ./Ckpts_scRefs/Heart_D2/Ref_Heart_sanger_D2.h5ad \
--cell_class_column cell_type \
--gpus 0,1,2,3 \
--sigma_begin 50 --sigma_end 0.002 --step_lr 3e-7 
```
The checkpoints and sampled psuedo-cells will be saved in `./Ckpts_scRefs/Heart_D2`, e.g, model_5000.pt, model_5000.h5ad. The pre-trained checkpoint can be used for any spatial data from the same tissue.

Due to the low sequencing depth (~2000 UMIs per cell) of this Human Heart scRNA-seq reference, we changed the default parameters of sigma_begin, sigma_end and step_lr.

As the sampling process of diffusion/score-based models requires hundreds to thousands of network evaluations to emulate a continuous process, the entire training process takes approximately 40 hours on four RTX 2080 Ti GPUs. Therefore, we are trying to accelarate the training process with some new technologies in the field of diffusion mode, such as stable diffusion and one-step difussion etc. 

Conveniently, we provided the pre-trained checkpoint (Ckpts_scRefs/Heart_D2/model_5000.pt) in [here](https://drive.google.com/drive/folders/1PXv_brtr-tXshBVEd_HSPIagjX9oF7Kg?usp=sharing), so you can skip this part.



## Frequently Asked Questions

1. I have access to a 3090 alternatively 2x V100-SXM2. Will that work for imputing onto a 200,000 cell MERFISH dataset?

   Answer: The minimum GPU requirement for SpatialScope is 2080 Ti. However, limited by GPU memory, we recommend impute 1000 cells at a time, more details are availabel in demo notebook [Mouse MOp (MERFISH)](https://github.com/YangLabHKUST/SpatialScope/blob/master/demos/Mouse-MOp-MERFISH.ipynb).


## Contact information

Please contact Xiaomeng Wan (xwanaf@connect.ust.hk), Jiashun Xiao (jxiaoae@connect.ust.hk) or Prof. Can Yang (macyang@ust.hk) if any enquiry.