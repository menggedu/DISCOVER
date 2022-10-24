import numpy as np
import torch
import scipy
from pyDOE import lhs

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

