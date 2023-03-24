

import numpy as np
import scipy.io as scio
import pickle
import math
from sklearn.metrics import mean_squared_error


# from dso.task.pde import *
# from utils_v1 import *
# from utils_v2 import *
# from utils_noise import *
from dso.task.pde.utils_v1 import *
from dso.task.pde.utils_v2 import *
from dso.task.pde.utils_noise import *


def load_data(dataset,noise_level=0, data_amount = 1, training=False):
    
    X = []
    
    # data_dir = f'./dso/task/pde/noise_data'
    # noise_path1 = f'{data_dir}/{dataset}_data_amount{data_amount}_noise{noise_level}.npy'
    noise_path = f'./dso/task/pde/noise_data_new/{dataset}_noise={noise_level}_data_ratio={data_amount}.npz'
    if dataset == 'chafee-infante': # 301*200的新数据
        if noise_level>0:  
            # u= np.load(noise_path1)
            # u_true = np.load("./dso/task/pde/data/chafee_infante_CI.npy")
            # u=  np.load('./dso/task/pde/data/chafee-infante_noise0.01.npy')
            data = np.load(noise_path)
            u = data['U_pred'].T
            # u= np.load(noise_path1)
            # import pdb;pdb.set_trace()
        else: 
            u = np.load("./dso/task/pde/data/chafee_infante_CI.npy")
        # import pdb;pdb.set_trace()
        x = np.load("./dso/task/pde/data/chafee_infante_x.npy").reshape(-1,1)
        t = np.load("./dso/task/pde/data/chafee_infante_t.npy").reshape(-1,1)
        n_input_var = 1
        sym_true = 'add,add,u1,n3,u1,diff2,u1,x1'
        
        n, m = u.shape 

        # right_side = 'right_side = uxx-u+u**3'
    elif dataset == 'Burgers':
        if noise_level>0:
            # u = np.load(noise_path)
            data = np.load(noise_path)
            u = data['U_pred'].T
            data = scio.loadmat('./dso/task/pde/data/burgers.mat')
        else: 
            data = scio.loadmat('./dso/task/pde/data/burgers.mat')
            u=data.get("usol")
        x=np.squeeze(data.get("x")).reshape(-1,1)
        t=np.squeeze(data.get("t").reshape(-1,1))
        sym_true = 'add,mul,u1,diff,u1,x1,diff2,u1,x1'
        right_side_origin = 'right_side_origin = -1*u_origin*ux_origin+0.1*uxx_origin'
        n_input_var = 1
        
   # Kdv -0.0025uxxx-uux
    elif dataset == 'Kdv':
        if noise_level>0:
            data = np.load(noise_path)
            u = data['U_pred'].T
            data = scio.loadmat('./dso/task/pde/data/Kdv.mat')
            # u = np.load(noise_path)
            # u=  np.load('./dso/task/pde/data/Kdv_noise0.01.npy')
        else: 
            
            data = scio.loadmat('./dso/task/pde/data/Kdv.mat')
            u=data.get("uu")
        n,m=u.shape
        x=np.squeeze(data.get("x")).reshape(-1,1)
        t=np.squeeze(data.get("tt").reshape(-1,1))
        # path = "C:\\Users\\lthpc\\Nutstore\\1\\code\\code\\PDE_RL_dso\\PDE-FIND-master\\PDE-FIND-master\\Datasets\\kdv.mat"
        # data = scio.loadmat(path)
        # u = np.real(data['usol'])
        # x = data['x'][0]
        # t = data['t'][:,0]
        n,m = u.shape #512, 201
        dt = t[1]-t[0]
        dx = x[1]-x[0]
        sym_true = 'add,mul,u1,diff,u1,x1,diff3,u1,x1'

        right_side_origin = 'right_side_origin = -0.0025*uxxx_origin-u_origin*ux_origin'
        n_input_var = 1
        
    elif dataset == 'PDE_divide':
        if noise_level>0:
            data = np.load(noise_path)
            u = data['U_pred'].T
            # u= np.load(noise_path1)
        else: 
            u=np.load("./dso/task/pde/data/PDE_divide.npy").T
        nx = 100
        nt = 251
        x=np.linspace(1,2,nx).reshape(-1,1)
        t=np.linspace(0,1,nt).reshape(-1,1)

        sym_true = 'add,div,diff,u1,x1,x1,diff2,u1,x1'
        right_side_origin = 'right_side_origin = -config.divide(ux_origin, x_all) + 0.25*uxx_origin'
        n_input_var = 1
        
    elif dataset == 'PDE_compound':
        if noise_level>0:
            data = np.load(noise_path)
            u = data['U_pred'].T

        else:
            u=np.load("./dso/task/pde/data/PDE_compound.npy").T
        nx = 100
        nt = 251
        x=np.linspace(1,2,nx).reshape(-1,1)
        t=np.linspace(0,0.5,nt).reshape(-1,1)
        n, m = u.shape 
        if training:
            u = u[int(n*0.1):int(n*0.9), int(m*0):int(m*1)]
            x = x[int(n*0.1):int(n*0.9)]
            t = t[int(m*0):int(m*1)]

        sym_true = 'add,mul,u1,diff2,u1,x1,mul,diff,u1,x1,diff,u1,x1'
        right_side_origin = 'right_side_origin = u_origin*uxx_origin + ux_origin*ux_origin'
        n_input_var = 1
       
    elif dataset == 'Burgers2':
        data = scipy.io.loadmat('./dso/task/pde/data/burgers2.mat')
        
        t = np.real(data['t'].flatten()[:,None])
        x = np.real(data['x'].flatten()[:,None])
        u = np.real(data['usol']) # x first
        # sym_true = 'add,mul,u1,diff2,u1,x1,mul,diff,u1,x1,diff,u1,x1'
        sym_true = 'add,mul,u1,diff,u1,x1,diff2,u1,x1'
        n_input_var = 1
        # import pdb;pdb.set_trace()
    elif dataset == 'KS':
        data = scipy.io.loadmat('./dso/task/pde/data/kuramoto_sivishinky.mat') # course temporal grid 
        # import pdb;pdb.set_trace()
        t = np.real(data['t'].flatten()[:,None])
        x = np.real(data['x'].flatten()[:,None])
        u = np.real(data['u'])
           
        sym_true = 'add,mul,u1,diff,u1,x1,add,diff2,u1,x1,diff4,u1,x1'
        n_input_var = 1
    elif dataset == 'KS_sine':
        data = scipy.io.loadmat('./dso/task/pde/data/KS_Sine.mat') # course temporal grid 
        
        t = np.real(data['t'].flatten()[:,None])
        x = np.real(data['x'].flatten()[:,None])
        u = np.real(data['usol'])
           
        sym_true = 'add,mul,u1,diff,u1,x1,add,diff2,u1,x1,diff4,u1,x1'
        n_input_var = 1
        
    elif dataset == 'RRE':
        t = np.arange(1,10001)*0.01
        t=t.reshape(-1,1)
        measured_data_points = [10,12,14,16,18]
        x = np.array(measured_data_points)*0.01
        x= x.reshape(-1,1)
        # x = np.linspace(np.min(depth),np.max(depth),20)
        data_path = './dso/task/pde/data/loam_S1'
        u = np.load(data_path+'/collected_theta_clean.npy')
        sym_true = 'add,diff,u1,x1,add,diff2,u1,x1,n2,diff,u1,x1'
        n_input_var = 1
        n, m = u.shape
        ut = np.zeros((n, m))
        dt = t[1]-t[0]
        ut = (u[:,1:]-u[:,:-1])/dt
        ut= ut[2:-2,]
        X.append(x)
        return u,X,t,ut,sym_true, n_input_var,None
        # import pdb;pdb.set_trace()
    else:
        assert False, "Unknown dataset"
    
        # self.dt = self.t[1]-self.t[0]
    # n, m = u.shape
    # if noise_level>0 and training==True:
    #     print("*"*4)
    #     u = u[5:-5, 5:-5]
    #     x = x[5:-5]
    #     t = t[5:-5]
    # import pdb;pdb.set_trace()
    n, m = u.shape
    ut = np.zeros((n, m))
    dt = t[1]-t[0]
    X.append(x)
 
    print("noise level:" , noise_level)

    for idx in range(n):
        ut[idx, :] = FiniteDiff(u[idx, :], dt)
        
    if noise_level>0 and training:
        # ut = ut[math.floor(n*0.1):math.ceil(n*0.9), math.floor(m*0.1):math.ceil(m*0.9)]
        ut = ut[math.floor(n*0.02):math.ceil(n*0.98), math.floor(m*0.02):math.ceil(m*0.98)]
    # x fist
    # x fist
    return u,X,t,ut,sym_true, n_input_var,None



def load_data_2D(dataset,noise_level=0, data_amount = 1, training=False):
    # from dso.task.pde.utils_v2 import *
    X= []
    data_dir = f'./dso/task/pde/noise_data'
    noise_path = f'{data_dir}/{dataset}_data_amount{data_amount}_noise{noise_level}.npy'
    if dataset == 'Cahn_Hilliard_2D':
        path='./dso/task/pde/data/ch.npz'
        path='./dso/task/pde/data/ch_ac.npz'
        path='./dso/task/pde/data/noise_ch.npz'
        if noise_level>0:
            # noise_path = f'{data_dir}_new/Cahn_Hilliard_2D_noise=0.01_data_ratio=0.4.npz'
            # data = np.load(noise_path)
            # u = data['U_pred']
            u = np.load(noise_path)
        else:  
            data = np.load(path)
            u = data['u']

            x,y=data['x'].reshape(1,-1,1),data['x'].reshape(1,1,-1)
        t_len,n,m= u.shape
        h = 1
        x = np.linspace(-0.5 * h, h * (64 + 0.5), 64 + 2)[1:-1].reshape(1,1,-1)
        y = np.linspace(-0.5 * h, h * (64 + 0.5), 64 + 2)[1:-1].reshape(1,1,-1)
        # t= data['t']
        t= np.linspace(0,10,500,endpoint=False)
        X.append(x)
        X.append(y)
        n_input_var = 2
    
        # sym_true = 'add,Diff2,sub,n3,u1,u1,add,Diff2,u1,x1,Diff2,u1,x2,x1,Diff2,sub,sub,n3,u1,u1,add,Diff2,u1,x1,Diff2,u1,x2,x2'
        
        # sym_true = 'sub,add,sub,n3,u1,diff2,lap,lap,diff2,lap,u1,x1,x2,lap,u1,lap,lap,u'
        #1th
        # sym_true = 'Diff2,Diff2,u1,x1,x2'
        #17th
        # sym_true = 'sub,sub,lap,sub,lap,u1,u1,lap,lap,u1,diff2,diff2,u1,x2,x1'
        
        sym_true = 'sub,lap,sub,n3,u1,u1,lap,lap,u1'
       
    
        ut = np.zeros((t_len, n,m))
       
        ut= (u[1:]-u[0:t_len-1])/(t[1]-t[0])
        u=u[:-1]
     
        if training:
            u_test= u[249:251]
            ut_test = ut[249:251]
            if noise_level >0:
                pass
            else:
                u= u[:3]#
                ut = ut[:3] #ut[240:260]

        else:
            u_test, ut_test = None, None  
        
    elif dataset == 'Allen_Cahn_2D':
        path = './dso/task/pde/data/bcpinn_ac.npz'
        if noise_level>0:  
            u = np.load(noise_path)
        else:
            
            data = np.load(path)
            u = data['u']
        t_len,n,m= u.shape

        h = 1/64
        y = np.linspace(-0.5 * h, 1 + h * 0.5, 64 + 2)[1:-1].reshape(1,1,-1)
        x = np.linspace(-0.5 * h, 1 + h * 0.5, 64 + 2)[1:-1].reshape(1,1,-1)
        
        t= np.linspace(0,5,100,endpoint=False)

        X.append(x)
        X.append(y)
        
        n_input_var = 2
        sym_true = 'n3,mul,div,div,u1,mul,Diff,n3,x2,x1,x2,n2,u1,x1'
        # sym_true = 'add,sub,lap,u1,n3,u1,u'
        sym_true = 'add,add,Diff2,u1,x1,Diff2,u1,x2,sub,u1,n3,u1'

        ut = np.zeros((t_len, n,m))
  
        ut= (u[1:]-u[0:t_len-1])/(t[1]-t[0])
        u=u[:-1]
        
        if training:
            u_test=u[39:41]
            ut_test = ut[39:41]
            u = u[:3]#[38:43]      #[31:51]
            ut = ut[:3]#[38:43]  #[31:51]
        else:
            u_test, ut_test = None, None  
        
        # lap = (Diff2_2(u1, x, 1)+Diff2_2(u1,x,2))

    else:
        assert False, "Unknown dataset"

    if noise_level>0:
        # import pdb;pdb.set_trace()
        t_len, n,m = ut.shape
        ut = ut[math.floor(t_len*0.1):math.ceil(t_len*0.9), math.floor(n*0.1):math.ceil(n*0.9),math.floor(m*0.1):math.ceil(m*0.9)]
        if u_test is not None:
            ut_test = ut_test[:,math.floor(n*0.1):math.ceil(n*0.9),math.floor(m*0.1):math.ceil(m*0.9)]
        
    return u,X,t,ut,sym_true, n_input_var,[u_test, ut_test]


if  __name__ ==  "__main__":
    import time
    st = time.time()
    u = np.random.rand(500,200)
    x = np.random.rand(500,1)
    su = np.sum(Diff3(u,x,0))

