import numpy as np
import torch
import scipy
from pyDOE import lhs
import  matplotlib.pyplot as plt
from dso.task.pde.utils_noise import np2tensor,tensor2np,load_PI_data

class Dataset():
    def __init__(self,
                 name,
                 noise,data_ratio,
                 device):
        self.name = name
        self.device = device
        self.data_ratio = data_ratio
        self.noise = noise
        
        
        self.init()
    
    def init(self):
            # load label data and collocation points
        X_u_train, u_train, X_f_train, X_u_val, u_val, [lb, ub], [x_star, u_star] = load_PI_data(self.name,
                                                                                                 self.noise,
                                                                                                 self.data_ratio
                                                                                                 )
        self.x = X_u_train[:,0:1]
        self.t = X_u_train[:,1:2]
        self.x_f = X_f_train[:,0:1]
        self.t_f = X_f_train[:,1:2]
        self.x_val = X_u_val[:,0:1]
        self.t_val = X_u_val[:,1:2]
        
        
        # self.X_u_train = np2tensor(X_u_train, self.device)
        # self.X_f_train = np2tensor(X_f_train,self.device, requires_grad=True)
        # self.X_u_val = np2tensor(X_u_val,self.device, requires_grad=False)
        self.x = np2tensor(self.x, self.device)
        self.t = np2tensor(self.t, self.device)
        self.x_f = np2tensor(self.x_f, self.device, requires_grad=True)
        self.t_f = np2tensor(self.t_f, self.device, requires_grad=True)
        self.x_val = np2tensor(self.x_val, self.device)
        self.t_val = np2tensor(self.t_val, self.device)
        self.u_train = np2tensor(u_train, self.device)
        

        self.u_val = np2tensor(u_val,self.device, requires_grad=False)
        # convert to tensor
        
        # full-field
        self.x_star = np2tensor(x_star, self.device, requires_grad = True)
        self.u_star =  u_star
       
        self.lb = np2tensor(lb, self.device)
        self.ub = np2tensor(ub, self.device )
    
    def resample(self, lb,ub,num, split=10):
        
        ub_t, lb_t = ub[-1], lb[-1]
        delta_t = (ub_t-lb_t)/split
        x_list= []
        t_list= []
        sub_num = num//split
        
        def sample(lb, ub, num_sample):

            X_f_train = lb + (ub-lb)*lhs(2, num_sample)
            return X_f_train
        
        for i in range(split):
            cur_lb = lb_t+i*delta_t
            cur_ub = lb_t+(i+1)*delta_t
            cur_xf = sample(cur_lb,cur_ub, sub_num)
            x = cur_xf[0:-1,:]
            t = cur_xf[-1:,:]
            x = np2tensor(x, self.device,requires_grad=True)
            t = np2tensor(t,self.device, requires_grad=True)
            x_list.append(x)
            t_list.append(t)
        
        return x_list, t_list

def load_1d_data(dataset,
                 noise_level,
                 data_ratio,
                 pic_path,
                 coll_num = 50000,
                 spline_sample = False
                 ):
 
    if dataset == 'Burgers2':
        data = scipy.io.loadmat('./dso/task/pde/data/burgers2.mat')
        
        t = np.real(data['t'].flatten()[:,None])
        x = np.real(data['x'].flatten()[:,None])
        Exact = np.real(data['usol']).T  # t first
    elif dataset == 'KS':
        data = scipy.io.loadmat('./dso/task/pde/data/kuramoto_sivishinky.mat') # course temporal grid 
        t = np.real(data['t'].flatten()[:,None])
        x = np.real(data['x'].flatten()[:,None])
        Exact = np.real(data['u']).T
        
    elif dataset == 'KS_sine': 
        data = scipy.io.loadmat('./dso/task/pde/data/KS_Sine.mat') # course temporal grid    
        t = np.real(data['t'].flatten()[:,None])
        x = np.real(data['x'].flatten()[:,None])
        Exact = np.real(data['usol']).T
        
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

