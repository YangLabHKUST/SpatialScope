import os
import argparse
import json
import numpy as np
import anndata
import pandas as pd
import scanpy as sc
import shutil
import glob as gb

import torch
from torch import nn
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from SCGrad import SCGradNN, EMAHelper
from utils import configure_logging,ConfigWrapper
import tqdm

      
class MouseBrainRefDataset(torch.utils.data.Dataset):
    def __init__(self, data, cell_type_num):
        super(MouseBrainRefDataset, self).__init__()
        
        self.data = data
        self.cell_type_num = cell_type_num

    def __getitem__(self, index):
        return self.data[index], self.cell_type_num[index]
    
    def __len__(self):
        return self.data.shape[0]

    
def sample_data(dataset, batch_size):
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=4
            )
            loader = iter(loader)
            yield next(loader)
            
            
def anneal_dsm_score_estimation_ddpm_uncond(scorenet, samples, mels, labels, sigmas, anneal_power=2., loss_type = 'L1'):
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
    target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)
    #scores = scorenet(mels, perturbed_samples, used_sigmas.squeeze())
    scores = scorenet(perturbed_samples, mels, used_sigmas.squeeze()) / used_sigmas
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    
    if loss_type == 'L2':
        anneal_power = 2.
        target = target * used_sigmas 
        scores = scores * used_sigmas 
        loss = torch.nn.MSELoss()(target, scores)

    elif loss_type == 'L1':
        anneal_power = 1.
        target = target * used_sigmas 
        scores = scores * used_sigmas 
        loss = torch.nn.L1Loss()(target, scores)

    return loss


def anneal_Langevin_dynamics_ddpm_uncond(x_mod, mels, scorenet, sigmas, n_steps_each=100, step_lr=0.00002):
    images = []
    
    with torch.no_grad():
        for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc='annealed Langevin dynamics sampling'):
            
            noise_level = torch.ones(x_mod.shape[0], device=x_mod.device) * sigma
            
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for s in range(n_steps_each):
                images.append(x_mod.to('cpu'))
                noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                grad = scorenet(x_mod, mels, noise_level) / sigma # noise_level
                x_mod = x_mod + step_size * grad + noise
                
                # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                #                                                          grad.abs().max()))
        grad = scorenet(x_mod, mels, noise_level) / sigma # noise_level
        x_mod = x_mod + sigma ** 2 * grad
        images.append(x_mod.to('cpu'))
        return images[-1]
    
    
    
def train(config, args, loggings):
    # load data
    sc_data_process = anndata.read_h5ad(args.scRef)
    if 'Marker' in sc_data_process.var.columns:
        sc_data_process_marker = sc_data_process[:,sc_data_process.var['Marker']]
    else:
        sc_data_process_marker = sc_data_process

    if sc_data_process_marker.X.max()>30:
        loggings.info(f'Maximum value: {sc_data_process_marker.X.max()}, need to run log1p')
        sc.pp.log1p(sc_data_process_marker)

    cell_type_array = np.array(sc_data_process_marker.obs[args.cell_class_column])
    cell_type_class = np.unique(cell_type_array)
    df_category = sc_data_process_marker.obs[[args.cell_class_column]].astype('category').apply(lambda x: x.cat.codes)


    # parameters: mean and cell type index
    cell_type_array_code = np.array(df_category[args.cell_class_column]) 
    try:
        data = sc_data_process_marker.X.toarray()
    except:
        data = sc_data_process_marker.X

    n, d = data.shape
    q = cell_type_class.shape[0]
    loggings.info(f'scRNA-seq data shape: {data.shape}')
    loggings.info(f'scRNA-seq cell class number: {q}')
        
    mu = np.zeros((q, d))
    std = np.zeros(q)
    for k in range(q):
        mu[k] = data[cell_type_array_code == k].mean(0).squeeze() 
        std[k] = np.std(data[cell_type_array_code == k] - mu[k][None,:])
    u, indices = np.unique(np.array(df_category[args.cell_class_column]), return_counts = True)

    # centering data
    loggings.info('data mean and var before centering: {}, {}'.format(data.mean(),np.std(data)))
    data = data - mu[cell_type_array_code]
    loggings.info('data mean and var after centering: {}, {}'.format(data.mean(),np.std(data)))   


    # load model
    model = SCGradNN(input_dim1 = data.shape[1], factors = args.factors, down_block_dim=args.down_block_dim).to(args.device)
    model_test = SCGradNN(input_dim1 = data.shape[1], factors = args.factors, down_block_dim=args.down_block_dim).to(args.device)
    model = nn.DataParallel(model, device_ids = list(range(torch.cuda.device_count())))
    model_test = nn.DataParallel(model_test, device_ids = list(range(torch.cuda.device_count())))
    ema_helper = EMAHelper()
    ema_helper.register(model)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.training_config.scheduler_step_size,
        gamma=config.training_config.scheduler_gamma
    )

    iteration = 1
    train_dataset = MouseBrainRefDataset(data = data, cell_type_num = cell_type_array_code)
    batch_size = args.bs
    dataset = iter(sample_data(train_dataset,batch_size))

    sigma_begin = args.sigma_begin
    sigma_end = args.sigma_end
    num_classes = args.num_classes
    step_lr = args.step_lr
    LT = args.LT

    sigmas = torch.tensor(
                np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end),
                                   num_classes))).float().to(args.device)
    
    loggings.info('maximum iteration: {}'.format(int(args.epoch * n / batch_size)))   
    for iteration in range(1, int(args.epoch * n / batch_size)):
        # Training step
        model.train()

        batch, cell_type = next(dataset)

        model.zero_grad()

        batch = batch.to(args.device, dtype=torch.float)
        mels = mu.copy()[cell_type]
        mels = torch.Tensor(mels)
        mels = mels.to(args.device)

        labels = torch.randint(0, len(sigmas), (batch.shape[0],), device=args.device)

        loss = anneal_dsm_score_estimation_ddpm_uncond(model, batch, mels, labels, sigmas, anneal_power=1., loss_type = 'L1')

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(),
            max_norm=config.training_config.grad_clip_threshold
        )

        optimizer.step()
        ema_helper.update(model)
        epoch_current = iteration*batch_size // n + 1

        if iteration % 100 == 0:
            ema_helper.ema(model_test)
            model_test.eval()
            with torch.no_grad():
                loss_test = anneal_dsm_score_estimation_ddpm_uncond(model_test, batch, mels, labels, sigmas, anneal_power=1., loss_type = 'L1')
                loss_stats = {
                    'total_loss': loss.item(),
                    'ema_loss': loss_test.item(),
                    'grad_norm': grad_norm.item()
                }

                loggings.info('Epoch:{} | iterate:{} | loss:{} | ema_loss:{} | grad_norm:{} | lr:{}'.format(epoch_current, iteration, loss_stats['total_loss'], loss_stats['ema_loss'], loss_stats['grad_norm'], scheduler.optimizer.param_groups[0]['lr']))


        if epoch_current % 500 == 0 and not os.path.exists(os.path.join(args.ckpt_path, f'model_{str(epoch_current).zfill(5)}.pt')):
            
            states = {'ckpt':ema_helper.state_dict()}
            states['Name'] = args.ckpt_path.split('/')[-1]
            states['cell_type_column'] = args.cell_class_column
            states['scRef_file'] = args.scRef
            states['Genes'] = sc_data_process_marker.var.index.values
            states['down_block_dim'] = args.down_block_dim
            states['factors'] = args.factors
            if args.sigma_begin>70:
                step_lr_decom = 1e-05
            else:
                step_lr_decom = 1e-06
            states["sigma"] = {"sigma_begin": args.sigma_begin, "sigma_end": args.sigma_end, "num_classes": args.num_classes, "step_lr_sample": args.step_lr, "step_lr_decom": step_lr_decom, "time": args.LT}
            torch.save(
                states,os.path.join(args.ckpt_path, f'model_{str(epoch_current).zfill(5)}.pt')
            )
            loggings.info('save checkpoint at: '+args.ckpt_path+f'/model_{str(epoch_current).zfill(5)}.pt')

            ema_helper.ema(model_test)
            model_test.eval()
            with torch.no_grad():
                sigmas_sample = torch.tensor(
                    np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end),
                                       num_classes))).float()

                loggings.info(f'sampling {args.g_ss} psuedo-cells')

                sample_size = args.g_ss

                index = np.random.multinomial(1, indices/indices.sum(), size = sample_size)
                mels = torch.Tensor(index @ mu.copy())
                mels = mels.to(args.device)

                samples = torch.randn(sample_size, mels.shape[-1], device=args.device)

                all_samples = anneal_Langevin_dynamics_ddpm_uncond(samples, mels, model_test, sigmas_sample, LT, step_lr)
                all_samples = np.array(all_samples) + np.array(mels.to('cpu'))

                nan_list = list(np.unique(np.where(np.isnan(all_samples) == 1)[0]))
                inf_list = list(np.unique(np.where(np.isinf(all_samples) == 1)[0]))
                all_samples = np.delete(np.array(all_samples),nan_list + inf_list,axis = 0)
                gen_sample_adata = anndata.AnnData(X = np.array(all_samples), var = sc_data_process_marker.var, obs=pd.DataFrame({'cell type':np.delete(index.argmax(1),nan_list + inf_list)})) 
                torch.cuda.empty_cache() 
                loggings.info('save sampled psuedo-cells at: '+args.ckpt_path+f'/model_{str(epoch_current).zfill(5)}.h5ad')
                gen_sample_adata.write(args.ckpt_path+f'/model_{str(epoch_current).zfill(5)}.h5ad')


def main():
    parser = argparse.ArgumentParser(description='Learning the gene expression distribution of scRNA ref using score-based model')
    parser.add_argument('--scRef', type=str, help='input sc/sn reference file')
    parser.add_argument('--cell_class_column', type=str, help='input cell class label column of reference file')
    parser.add_argument('--ckpt_path', type=str, help='output checkpoint path')
    parser.add_argument('--epoch', type=int, help='number of training epochs, at least 3000', default=10000)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
    parser.add_argument('--bs', type=int, help='batch size per iteration', default=256)
    parser.add_argument('--g_ss', type=int, help='size of sampled pseudo-cells', default=2000)
    parser.add_argument('--ckpt_path_overwrite', action="store_true")
    parser.add_argument("--sigma_begin", type=int, help="sigma_begin", default=100)
    parser.add_argument("--sigma_end", type=float, help="sigma_end", default=0.01)
    parser.add_argument("--num_classes", type=int, help="num_classes", default=232)
    parser.add_argument("--step_lr", type=float, help="step_lr", default=6.6e-6)
    parser.add_argument("--LT", type=int, help="LT", default=5)
    parser.add_argument('--factors', type=str, help='factor of Unet', default = '3,4,5,1')
    parser.add_argument('--down_block_dim', type=str, help='down_block_dim of Unet', default = '32,128,256,512')
    parser.add_argument('--gpus', type=str, help='used GPUs', default = '0,1,2,3')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.__setattr__('device', device)    

    args.factors = [int(i) for i in args.factors.split(',')]
    args.down_block_dim = [int(i) for i in args.down_block_dim.split(',')]

    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)
    else:
        if args.ckpt_path_overwrite:
            shutil.rmtree(args.ckpt_path)
            os.makedirs(args.ckpt_path)

    loggings = configure_logging(args.ckpt_path+'/train')
    
    loggings.info(f'is GPU availabel: {torch.cuda.is_available()}')
    loggings.info(f'availabel GPUs: {torch.cuda.device_count()}')
    loggings.info(f'sc Ref: {args.scRef}')
    loggings.info(f'cell_class_column: {args.cell_class_column}')
    loggings.info(f'output log/checkpoint name: {args.ckpt_path}')
    loggings.info(f'learning rate: {args.lr}')
    loggings.info(f'max epoch: {args.epoch}')
    loggings.info(f'batch size: {args.bs}')
    loggings.info(f'Sigma begin: {args.sigma_begin}')
    loggings.info(f'Sigma end: {args.sigma_end}')
    loggings.info(f'num_classes: {args.num_classes}')
    loggings.info(f'step_lr: {args.step_lr}')
    loggings.info(f'LT: {args.LT}')
    loggings.info(f'factors: {args.factors}')
    loggings.info(f'down_block_dim: {args.down_block_dim}')
    loggings.info(f'gpus: {args.gpus}')

    
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default.json')) as f:
        config = ConfigWrapper(**json.load(f))
        
    train(config, args, loggings)


if __name__ == '__main__':
    main()

