from utils import *
import torch
import json
import warnings
warnings.filterwarnings('ignore')

import copy
import argparse
from utils_CTI import *
from Cell_Type_Identification import SpatialScopeCTI
from Decomposition import GeneExpDecomposition


if __name__ == "__main__":
    HEADER = """
    <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    <> 
    <> SpatialScope
    <> Version: %s
    <> MIT License
    <>
    <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    <> Software-related correspondence: %s or %s
    <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    <> example:
        python <install path>/src/Cell_Type_Identification.py \\
            --tissue heart \\
            --out_dir path_to_save_results \\
            --ST_DataDir path_to_STdata \\
            --SC_DataDir path_to_single_cell_reference_data 

    <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>  
    """ 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, type=str, help='configuration file')
    parser.add_argument('--out_dir', type=str, help='output path', default='./output')
    parser.add_argument('--nu', type=float, help='xval 10fold NO.', default=10)
    parser.add_argument('--spot_range', type=str, help='spot range', default=None) #[0,10]
    parser.add_argument("--posterior_mean", action="store_true", help="whether calculate posterior mean")
    parser.add_argument('--gpu', type=str, help='gpu', default='0,1,2,3')
    
    args = parser.parse_args()

    with open(os.path.join('configs', args.config + '.json')) as f:
        config = ConfigWrapper(**json.load(f))
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device
        
    if args.spot_range is not None:
        config.decomposition.spot_range = [int(x) for x in args.spot_range.split(',')]
    config.data.out_dir = args.out_dir
    config.CTI.nu = args.nu
    config.decomposition.posterior_mean = args.posterior_mean
        
        
        
    # cell type identification
    CTI = SpatialScopeCTI(config.data.tissue,config.data.out_dir, config.data.ST_DataDir, config.data.SC_DataDir, cell_class_column = config.data.cell_class_column)
    CTI.CellTypeIdentification(nu = config.CTI.nu, loc = config.CTI.location, X = config.CTI.X, Y = config.CTI.Y)
    
    # single-cell gene expression decomposition
    DECOM = GeneExpDecomposition(config, loggings = CTI.loggings)
    DECOM.decomposition()
    if config.decomposition.posterior_mean:
        DECOM.posterior_mean(e = DECOM.spot_range[1], inter = DECOM.spot_range[1], es = DECOM.spot_range[1], suffix='_Sig' + str(DECOM.config.decomposition.power) + '_' + str(DECOM.config.sigma.step_lr_decom) + '.h5ad')
        
        



    
