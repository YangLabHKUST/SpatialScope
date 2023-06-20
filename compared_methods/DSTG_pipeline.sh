set -ex

scrna_path=$1
spatial_path=$2
celltype_key=$3
output_path=$4


echo $scrna_path
echo $spatial_path
echo $celltype_key

cd /home/jxiaoae/spatial/revision/SpatialBenchmarking/Codes/Deconvolution/DSTG/DSTG

Rscript convert_data_h5ad.r $scrna_path $celltype_key $spatial_path

Rscript convert_data.R scRNAseq_data.RDS  spatial_data.RDS  scRNAseq_label.RDS 

python  train.py

cp ./DSTG_Result/predict_output.csv $output_path/DSTG_output.csv

# ./DSTG_pipeline.sh \
# /home/share/xwanaf/sour_sep/benchmarking/Decon/MERFISH/MOp3_16x58_x1.0/sc_rna.h5seurat \
# /home/share/xwanaf/sour_sep/benchmarking/Decon/MERFISH/MOp3_16x58_x1.0/spatial.h5seurat \
# cell_type

# ~/spatial/SpatialScope/compared_methods/DSTG/DSTG/DSTG_Result/predict_output.csv
