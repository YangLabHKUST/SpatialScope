library(CARD)
library(Matrix)
library(SingleCellExperiment)
library(Matrix)
library(SeuratDisk)

args<-commandArgs(T)
snrna_path = args[1]
spatial_path = args[2]
celltype_final = args[3]
output_path = args[4]
loc_path = args[5]

sc <- LoadH5Seurat(snrna_path)
st <- LoadH5Seurat(spatial_path)
loc = read.table(loc_path, sep = ',', row.name = 1, header = T)

spatial_count = st@assays$RNA@counts
spatial_location = loc
sc_count = sc@assays$RNA@counts
sc_meta = sc@meta.data
sc_meta$sampleInfo = "sample1"

# create card object
CARD_obj = createCARDObject(
	sc_count = sc_count,
	sc_meta = sc_meta,
	spatial_count = spatial_count,
	spatial_location = spatial_location,
	ct.varname = celltype_final,
	ct.select = unique(sc_meta[,celltype_final]),
	sample.varname = 'sampleInfo',
	minCountGene = 100,
	minCountSpot = 0) 


# Decon
CARD_obj = CARD_deconvolution(CARD_object = CARD_obj)

norm_weights = sweep(CARD_obj@Proportion_CARD, 1, rowSums(CARD_obj@Proportion_CARD), '/') 
write.csv(norm_weights, paste0(output_path, '/CARD_result.txt'))