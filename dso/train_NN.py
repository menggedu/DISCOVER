from dso import Denoise_NN


# Libraries
import numpy as np
import torch
import argparse
import logging
import random

def set_seed(seed):
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def check_args(args):
    if args.normalize_type == 'min_max':
        args.loss_type = 'gls_thresh'
        args.activation = 'Softplus'
    else:
        assert args.normalize_type == 'standard' or args.normalize_type == 'None'
        args.loss_type = 'mse'
        args.activation = 'sin'
        
    if '2D' in args.data_name:
        args.input_dim=3
    return args

def test(model, device):
    save_path = model.model_name+'/best.ckpt'
    epoch, best_val = model.load_best_model()
    model.nn.to(device)
    model.to_tensor(device)
    model.save_model(save_path, epoch, best_val)
    model.evaluate()
    model.save_data()  
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train FNN as denoising model')
    parser.add_argument('--maxit', default=500000, type=int, help='number of iterations')
    parser.add_argument('--hidden-num', default=1024, type=int, help='hidden')
    parser.add_argument('--num-layer', default=1, type=int, help='total length')
    parser.add_argument('--input-dim', default=2, type=int, help='Nx') 
    parser.add_argument('--out-dim', default=1, type=int, help='Ny')
    parser.add_argument('--activation', default='sin', type=str, help='initial condition') # ['circle', 'dumbbell', 'star', 'separation', 'torus', 'maze']
    parser.add_argument('--batch-num', default=None, type=int, help='number of batch')
    parser.add_argument('--loss-type', default='mse',  choices=['gls_thresh', "mse"], type=str, help='Ny')
    parser.add_argument('--seed', default=0, type=int, help='Ny')
    parser.add_argument('--display-step', default=500, type=int, help='Nx') 
    parser.add_argument('--early-stopper', default=5000, type=int, help='Nx') 
    
    parser.add_argument('--data-name', default="PDE_compound", type=str, help='code') # 0: pytorch gpu, 1: pytorch cpu, 2: python cpu
    parser.add_argument('--normalize-type', default="standard", choices=['min_max', "standard","None"], help='npy files') # 0: No, 1: Yes
    parser.add_argument('--train-ratio', default='0.4', type=float, help='avaiable data')
    parser.add_argument('--data-efficient', default=0, type=int, help='Nx') 
    parser.add_argument('--noise-level', default='0.01', type=float, help='noise')
    parser.add_argument('--mode', default='train', type=str, help='Nx') 
    
    
    args = parser.parse_args()
    
    
    set_seed(args.seed)
    device = torch.device('cuda:0')
    # check args
    args = check_args(args)

    logging.info(args)
    model = Denoise_NN(args)
    
    # test
    # nn = DNN2(3,256,1,3)
    # ckpt = './dso/task/pde/ckpt_remote/noise0.01_data_amount0.4/Cahn_Hilliard_2D/300000.ckpt'
    # ckpt = './out/checkpoint/Cahn_Hilliard_2D_noise=0.01_data_ratio=0.4_standard/best.ckpt'
    # total_state_dict = torch.load(ckpt, map_location='cpu')
    # model.nn.load_state_dict(total_state_dict['model'])
    # model.nn.to(device)
    # model.to_tensor(device)
    if args.mode == 'train':
        model.train(args.maxit, device, early_stopper=args.early_stopper, display_step=args.display_step )
        
        model.evaluate()
        
        model.save_data()
        
    elif args.mode == 'test':
        test(model, device)
        
    else:
        raise NotImplementedError
    
    
    
    