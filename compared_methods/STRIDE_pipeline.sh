set -ex

scrna_path=$1
spatial_path=$2
celltype_key=$3
output_path=$4

prefix='STRIDE'


python  /home/jxiaoae/spatial/revision/SpatialBenchmarking/Codes/Deconvolution/STRIDE/covert_data.py  $scrna_path $celltype_key $spatial_path

topics=`awk '{print $2}' /home/jxiaoae/spatial/revision/SpatialBenchmarking/Codes/Deconvolution/STRIDE/scRNAseq_label.csv | sort | uniq  | wc -l`

echo $topics

STRIDE deconvolve --sc-count /home/jxiaoae/spatial/revision/SpatialBenchmarking/Codes/Deconvolution/STRIDE/scRNAseq_data.csv \
--sc-celltype /home/jxiaoae/spatial/revision/SpatialBenchmarking/Codes/Deconvolution/STRIDE/scRNAseq_label.csv \
--st-count /home/jxiaoae/spatial/revision/SpatialBenchmarking/Codes/Deconvolution/STRIDE/spatial_data.csv \
--outdir $output_path --outprefix $prefix --normalize --gene-use All --st-scale-factor 300 --sc-scale-factor 300 --ntopics $topics

# ./STRIDE_pipeline.sh \
# /home/share/xwanaf/sour_sep/benchmarking/Decon/MERFISH/MOp3_16x58_x1.0/sc_rna.h5ad \
# /home/share/xwanaf/sour_sep/benchmarking/Decon/MERFISH/MOp3_16x58_x1.0/spatial.h5ad \
# cell_type \
# ./test
# /home/jxiaoae/spatial/revision/SpatialBenchmarking/Codes/Deconvolution/STRIDE_pipeline.sh \
# /home/share/xwanaf/sour_sep/benchmarking/Decon/MERFISH/MOp3_16x58_x1.0/sc_rna.h5ad \
# /home/share/xwanaf/sour_sep/revision/data/RCTDnv_MOp/260UMI/spatial.h5ad cell_type \
# /home/share/xiaojs/spatial/benchmarking/MOp_260UMI