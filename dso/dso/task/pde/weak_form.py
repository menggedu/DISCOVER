import numpy as np
from scipy.special.orthogonal import p_roots
from numpy.polynomial.legendre  import leggauss
import torch
import pandas as pd
import scipy.io as scio
import math


from dso.task.pde.pde import PDETask, make_pde_metric
from dso.library import Library
from dso.functions import create_tokens,add_torch_tokens
from dso.task.pde.data_load import *
from dso.task.pde.utils_nn import load_noise_data,plot_field, plot_ut,torch_diff
from dso.task.pde.utils_noise import *
from dso.task.pde.utils_v1 import FiniteDiff, Diff, Diff2 ,Diff4
from dso.task.pde.utils_nn import ANN, PDEDataset,np2tensor, tensor2np


class WeakEvaluate:
    def __init__(self, n = 5, L = 2, x_low= 20, t_low=20, x_up=80, t_up=80, x_num=1200, t_num=300):
        
        [self.x,self.w] = leggauss(n)
        self.n = n
        self.L = L
        self.x_low= x_low
        self.t_low=t_low 
        self.x_up=x_up
        self.t_up=t_up 
        self.x_num=x_num
        self.t_num=t_num    


    def reconstruct_input(self,device, x_low= 20, t_low=20, x_up=80, t_up=80, x_num=1200, t_num=300 ):

        """
        convert original x (ndim,1) to (ndim*5 ,1)
        1d
        """
        #define meta domain
        x = np.linspace(x_low, x_up, num =x_num)
        t = np.linspace(t_low, t_up, num =t_num)
        x1,t1 = np.meshgrid(x+0.5*self.L, t)
        x2,t2 = np.meshgrid(x-0.5*self.L, t)
        # import pdb;pdb.set_trace()
        x1 = x1.reshape(-1,1)
        x2 = x2.reshape(-1,1)
        t1 = t1.reshape(-1,1)
        t2 = t2.reshape(-1,1)

        x = x.reshape(-1)
        x = np.expand_dims(x, axis = 1).repeat(self.n, axis = 1)
        x = x+0.5*self.L*self.x
        x = x.reshape(-1,1)

        t = t.reshape(-1)
        # t = np.expand_dims(t, axis = 1).repeat(self.n, axis = 1)
        t = t.reshape(-1,1)

        xw,tw = np.meshgrid(x, t)
        xw = xw.reshape(-1,1)
        tw = tw.reshape(-1,1)
        return xw, tw, x1,t1,x2,t2

    def glq_cal(self, f_value):
        if torch.is_tensor(f_value):
            f_value = tensor2np(f_value)
        f_value = f_value.reshape(-1,len(self.x))
        f_sum = 0.5*self.L*np.dot(f_value,self.w)
        # G=0.5*self.L*np.dot(self.w*f(0.5*(b-a)*self.x+0.5*(b+a)))
        return f_sum

def f(x):
    return x

if __name__ == "__main__":

    xmid = np.array([0,1])
    L = 1
    x_up = xmid-L/2
    x_down = xmid+L/2

    we = WeakEvaluate()
    values = we.gauss(f,x_up, x_down)
    print(values)
