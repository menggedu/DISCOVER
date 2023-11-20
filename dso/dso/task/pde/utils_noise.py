import numpy as np
# import torch.
import torch
import scipy
from pyDOE import lhs
import matplotlib.pyplot as plt
import math



def cut_bound_quantile(x, t, quantile=0.1):
    low_x, low_t= [np.quantile(x[i],quantile, axis=0) for i in range(len(x))],np.quantile(t,quantile,axis = 0)
    up_x,up_t = [np.quantile(x[i],1-quantile, axis =0) for i in range(len(x))], np.quantile(t,1-quantile,axis =0)
    x_dim = len(low_x)
    x_len = len(x[0])
    x_limit = np.ones(x_len, dtype=np.bool) 
    # import pdb;pdb.set_trace()
    for i in range(x_dim):
        
        # x_limit_cur = np.logical_and(x[:,i]>low_x,x[:,i]<up_x)
        x_limit_cur = np.logical_and(x[i][:,0]>low_x[i],x[i][:,0]<up_x[i])
        x_limit = np.logical_and(x_limit_cur, x_limit)
        
    t_limit = np.logical_and(t>=low_t, t<=up_t).reshape(-1)
    limit = np.logical_and(x_limit,t_limit).reshape(-1)
   
    x= [x[i][limit,:] for i in range(len(x))]
    t= t[limit,:]
    return x, t

def cut_bound(result,percent, test=False):
    # import pdb;pdb.set_trace()
    r_shape = result.shape
    low_bound = [ math.floor(percent*dim) for dim in r_shape ]
    up_bound = [ math.ceil((1-percent)*dim) for dim in r_shape ]
    if len(r_shape)==2:
        result = result[low_bound[0]:up_bound[0],low_bound[1]:up_bound[1]]
        # result = result[5:-5,5:-5]
    elif len(r_shape) == 3:
        result = result[low_bound[0]:up_bound[0],low_bound[1]:up_bound[1], low_bound[2]:up_bound[2]]
    elif len(r_shape) == 4:
        result = result[low_bound[0]:up_bound[0],low_bound[1]:up_bound[1], low_bound[2]:up_bound[2],low_bound[3]:up_bound[3]]

    return result           

def tensor2np(tensor):
    array = tensor.cpu().data.numpy()
    return array

def np2tensor(array, device,requires_grad =False):
    tensor =  torch.from_numpy(array).float().to(device)
    if requires_grad:
        tensor.requires_grad = True
    return tensor

def normalize(U,normalize_type):
    
    #normalize u
    if normalize_type == 'min_max':
        # compute extrema
        U_min = np.min(U,axis =0)
        U_max = np.max(U - U_min, axis = 0)
        U = (U - U_min)/U_max
        normalize_params= [ U_min,U_max]
    if normalize_type == 'None':
        return U, [0,1]
    else:
        
        U_mean = U.mean(axis =0)
        U_std = U.std(axis=0)
        U = ((U-U_mean) / (U_std))
        normalize_params= [ U_mean, U_std]
        
    return U, normalize_params

def unnormalize(x, normalize_params):
    """_summary_

    Args:
        x (_type_):  normalized data
        normalize_params (_type_): (min,max) or (mean, std)

    Returns:
        _type_: _description_
    """
    n1, n2 = normalize_params
    x = x*n2+n1
    return x


def load_PI_data(dataset,
                 noise_level,
                 data_ratio,
                 pic_path,
                 coll_num = 50000,
                 spline_sample = False
                 ):
    """_summary_
    load label_data available:
    load collocation points for PINN
    generate meta data
    """
    if dataset == 'Burgers2':
        data = scipy.io.loadmat('./dso/task/pde/data/burgers2.mat')
        
        t = np.real(data['t'].flatten()[:,None])
        x = np.real(data['x'].flatten()[:,None])
        Exact = np.real(data['usol']).T  # t first
    else:
        assert False, f"Dataset {dataset} is not existed"
    X, T = np.meshgrid(x,t)
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]              

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0) 
    
    x_len = len(x)
    total_num = X_star.shape[0]
    if spline_sample:
    # select 10 position with all time steps
        N_u_s = int(x_len*data_ratio)
        print(f"spline sample number is {N_u_s}")
        idx_s = np.random.choice(x.shape[0], N_u_s, replace=False)
        X0 = X[:, idx_s]
        T0 = T[:, idx_s]
        Exact0 = Exact[:, idx_s]
        
        N_u_t = int(t.shape[0]*1)
        idx_t = np.random.choice(t.shape[0], N_u_t, replace=False)
        X0 = X0[idx_t, :]
        T0 = T0[idx_t, :]
        Exact0 = Exact0[idx_t, :]
        X_u_meas = np.hstack((X0.flatten()[:,None], T0.flatten()[:,None]))
        u_meas = Exact0.flatten()[:,None] 
    else:
        sample_num = int(total_num*data_ratio)
        print(f"random sample number: {sample_num} ")
        ID = np.random.choice(total_num, sample_num, replace = False)
        X_u_meas = X_star[ID,:]
        u_meas = u_star[ID,:] 

      
    # Training measurements, which are randomly sampled spatio-temporally
    Split_TrainVal = 0.8
    N_u_train = int(X_u_meas.shape[0]*Split_TrainVal)
    idx_train = np.random.choice(X_u_meas.shape[0], N_u_train, replace=False)
    X_u_train = X_u_meas[idx_train,:]
    u_train = u_meas[idx_train,:]
    
    
    # Validation Measurements, which are the rest of measurements
    idx_val = np.setdiff1d(np.arange(X_u_meas.shape[0]), idx_train, assume_unique=True)
    X_u_val = X_u_meas[idx_val,:]
    u_val = u_meas[idx_val,:]
            
    # Collocation points
    
    N_f = coll_num

    X_f_train = lb + (ub-lb)*lhs(2, N_f)
#    X_f_train = lb + (ub-lb)*sobol_seq.i4_sobol_generate(2, N_f)        
    X_f_train = np.vstack((X_f_train, X_u_train))
    
    # Option: Add noise
    noise = noise_level
    print("noise", noise)
    u_train_noise = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
    u_val = u_val + noise*np.std(u_val)*np.random.randn(u_val.shape[0], u_val.shape[1])
    
    x_mid = x[len(x)//2]
    t_mid = t[len(t)//2]
    x_id = X_u_train[:,0]==x_mid
    t_id =X_u_train[:,1] == t_mid
    x_id_all = X_star[:,0]==x_mid
    t_id_all =X_star[:,1] == t_mid
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(1,3,1)
    ax.scatter(X_u_train[x_id][:,1],u_train_noise[x_id], label = 'noise')
    ax.plot(t,u_star[x_id_all], label = 'true')
    ax.legend()
    ax = fig.add_subplot(1,3,2)
    ax.scatter(X_u_train[t_id][:,0],u_train_noise[t_id], label = 'noise')
    ax.plot(x,u_star[t_id_all], label = 'true')
    ax.legend()
    ax = fig.add_subplot(1,3,3)
    ax.scatter(X_u_train[:,0],X_u_train[:,1])
    plt.savefig(pic_path+'data.png',dpi=300)
    return X_u_train, u_train_noise, X_f_train, X_u_val, u_val, [lb, ub], [X_star, u_star]