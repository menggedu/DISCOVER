"""Load full simulation data of multi-dimenisonal systems. 
"""
import numpy as np
import scipy.io as scio
import scipy
import pickle
import math
from sklearn.metrics import mean_squared_error
from scipy.integrate import odeint
from numpy.fft import fft, ifft, fftfreq
from time import time

from dso.task.pde.utils_v1 import *
from dso.task.pde.utils_v2 import *
from dso.task.pde.utils_noise import *
from dso.task.pde.parameter_process import parametric_burgers_rhs
from dso.task.pde.utils_subgrid import *
from dso.task.pde.utils_noise import cut_bound


def load_data(dataset,noise_level=0, data_amount = 1, training=False,cut_ratio =0.03):
    """
    load data and pass them to the corresponding PDE task 
    """
    X = []
    
    noise_path = f'./dso/task/pde/noise_data_new/{dataset}_noise={noise_level}_data_ratio={data_amount}.npz'
    n_state_var = 1
    if dataset == 'chafee-infante': # 301*200
        if noise_level>0:  

            data = np.load(noise_path)
            u = data['U_pred'].T
            # u= np.load(noise_path1)
            # import pdb;pdb.set_trace()
        else: 
            u = np.load("./dso/task/pde/data_new/chafee_infante_CI.npy")
        x = np.load("./dso/task/pde/data_new/chafee_infante_x.npy").reshape(-1,1)
        t = np.load("./dso/task/pde/data_new/chafee_infante_t.npy").reshape(-1,1)
        n_input_var = 1
        sym_true = 'add,add,u1,n3,u1,diff2,u1,x1'
        
        n, m = u.shape 

    elif dataset == 'Burgers':
        if noise_level>0:
            # u = np.load(noise_path)
            data = np.load(noise_path)
            u = data['U_pred'].T
            data = scio.loadmat('./dso/task/pde/data_new/burgers.mat')
        else: 
            data = scio.loadmat('./dso/task/pde/data_new/burgers.mat')
            u=data.get("usol")
        x=np.squeeze(data.get("x")).reshape(-1,1)
        t=np.squeeze(data.get("t").reshape(-1,1))
        sym_true = 'add,mul,u1,diff,u1,x1,diff2,u1,x1'
        right_side_origin = 'right_side_origin = -1*u_origin*ux_origin+0.1*uxx_origin'
        n_input_var = 1

    elif dataset == 'Kdv':
        if noise_level>0:
            data = np.load(noise_path)
            u = data['U_pred'].T
            data = scio.loadmat('./dso/task/pde/data_new/Kdv.mat')
        else: 
            
            data = scio.loadmat('./dso/task/pde/data_new/Kdv.mat')
            u=data.get("uu")
        n,m=u.shape
        x=np.squeeze(data.get("x")).reshape(-1,1)
        t=np.squeeze(data.get("tt").reshape(-1,1))

        n,m = u.shape #512, 201
        dt = t[1]-t[0]
        dx = x[1]-x[0]
        sym_true = 'add,mul,u1,diff,u1,x1,diff3,u1,x1'

        right_side_origin = 'right_side_origin = -0.0025*uxxx_origin-u_origin*ux_origin'
        n_input_var = 1
        
    elif dataset == 'PDE_divide':
        u=np.load("./dso/task/pde/data_new/PDE_divide.npy").T
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
            u=np.load("./dso/task/pde/data_new/PDE_compound.npy").T
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

    elif dataset == 'KS2':
        data = scipy.io.loadmat('./dso/task/pde/data/KS.mat') # course temporal grid 
        # import pdb;pdb.set_trace()
        
        t = np.real(data['t'].flatten()[:,None])
        x = np.real(data['x'].flatten()[:,None])
        u = np.real(data['usol'])
           
        sym_true = 'add,mul,u1,diff,u1,x1,add,diff2,u1,x1,diff2,diff2,u1,x1,x1'
        n_input_var = 1
    elif dataset == "fisher":
    
        data=scipy.io.loadmat('./dso/task/pde/data/fisher_nonlin_groundtruth.mat')

        # D=data['D'] #0.02
        # r=data['r'] #10
        # K=data['K']
        x=np.squeeze(data['x'])[1:-1].reshape(-1,1)
        t=np.squeeze(data['t'])[1:-1].reshape(-1,1)
        u=data['U'][1:-1,1:-1].T
        sym_true = "add,mul,u1,diff2,u1,x1,add,n2,diff,u1,x1,add,u1,n2,u1"
        n_input_var = 1
        
    elif dataset == "fisher_linear":
        
        data=scipy.io.loadmat('./dso/task/pde/data/fisher_groundtruth.mat')

        # D=data['D'] #0.02
        # r=data['r'] #10
        # K=data['K']
        x=np.squeeze(data['x'])[1:-1].reshape(-1,1)
        t=np.squeeze(data['t'])[1:-1].reshape(-1,1)
        u=data['U'][1:-1,1:-1].T
        sym_true = "add,diff2,u1,x1,add,u1,n2,u1"
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
        return [u],X,t,ut,sym_true, n_input_var,None,n_state_var

    elif dataset == 'PAR':
        return load_real_data()
    else:
        assert False, "Unknown dataset"
    
    n, m = u.shape
    ut = np.zeros((n, m))
    dt = t[1]-t[0]
    X.append(x)
 
    print("noise level:" , noise_level)

    for idx in range(n):
        ut[idx, :] = FiniteDiff(u[idx, :], dt)
        
    if noise_level>0 and training:
        # ut = ut[math.floor(n*0.1):math.ceil(n*0.9), math.floor(m*0.1):math.ceil(m*0.9)]
        ut = ut[math.floor(n*0.03):math.ceil(n*0.97), math.floor(m*0.03):math.ceil(m*0.97)]
    # x fist
    return [u],X,t,ut,sym_true, n_input_var,None,n_state_var


def load_real_data():
    X = []
    data = np.load('./dso/task/pde/data/PAR.npz')
    # import pdb;pdb.set_trace()
    gPAR2, gPAR6= data['gPAR2'],data['gPAR6']
    gv = data['gV']
    x, t=  data['X_PAR'], data['T_PAR']
    dx = x[1]-x[0]
    dt = t[1]-t[0]
    x =x.reshape(-1,1)
    t =t.reshape(-1,1)
    ut = np.zeros(gPAR2.shape)
    ut = (gPAR2[:,1:]-gPAR2[:,:-1])/dt
    ut = (gPAR6[:,1:]-gPAR6[:,:-1])/dt
    # ut= ut[30:54,:]
    print(ut.shape)
    ut = cut_bound(ut, percent=0.1)
    X.append(x)
    u = gPAR2 
    sym_true ='add,u1,mul,n2,u1,u2'
    sym_true ='add,u2,mul,n2,u2,u1'
    n_input_var=1
    n_state_var=2
    return [gPAR2[:,:-1], gPAR6[:,:-1], gv[:,:-1]],X,t,ut, sym_true,n_input_var, None,n_state_var


def load_data_2D(dataset,noise_level=0, data_amount = 1, training=False,cut_ratio =0.03):
    # from dso.task.pde.utils_v2 import *
    X= []
    data_dir = f'./dso/task/pde/noise_data'
    noise_path = f'{data_dir}/{dataset}_data_amount{data_amount}_noise{noise_level}.npy'
    n_state_var = 1
    if dataset == 'Cahn_Hilliard_2D':
        path='./dso/task/pde/data/ch.npz'
        path='./dso/task/pde/data/ch_ac.npz'
        path='./dso/task/pde/data_new/noise_ch.npz'
        if noise_level>0:
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
        n_state_var = 1
        
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
        path = './dso/task/pde/data_new/bcpinn_ac.npz'
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
        
    return [u],X,t,ut,sym_true, n_input_var,[u_test, ut_test],n_state_var

def load_data_MD_NU(dataset, noise_level=0, data_amount = 1, training = False, cut_ratio =0.03):
    if "2D" in dataset:
        return load_data_2D(dataset,noise_level=0, data_amount = 1, training=False,cut_ratio =0.03)
    elif dataset == 'rd_MD_NU':
        data = scipy.io.loadmat('./dso/task/pde/data/reaction_diffusion_standard.mat') # grid 256*256*201
            
        t = np.real(data['t'].flatten()[:,None])
        x = np.real(data['x'].flatten()[:,None])
        y = np.real(data['y'].flatten()[:,None])
        Exact_u = np.transpose(data['u'],(2,0,1))
        Exact_v = np.transpose(data['v'],(2,0,1))

        t_len,n,m = Exact_u.shape
        dt = t[1,0]-t[0,0]
        ut = (Exact_u[1:,:,:] - Exact_u[:-1,::])/dt
        # import pdb;pdb.set_trace()
        sym_true = "add,Diff2,u1,x1,add,Diff2,u1,x2,add,mul,u1,n2,u2,add,n3,u1,add,n3,u2,add,mul,n2,u1,u2,u1"
        X = [x,y]
        n_input_var = 2
        if noise_level>0 and training:
            # ut = ut[math.floor(n*0.1):math.ceil(n*0.9), math.floor(m*0.1):math.ceil(m*0.9)]
            ut = ut[math.floor(t_len*0.03):math.ceil(t_len*0.97),math.floor(n*0.03):math.ceil(n*0.97), math.floor(m*0.03):math.ceil(m*0.97)]
        
        n_state_var = 2
        return [Exact_u,Exact_v,],X,t,ut, sym_true,n_input_var,[None,None],n_state_var
    elif "ns" in dataset:
        if dataset == 'ns_MD_NU':
            data = scipy.io.loadmat('./dso/task/pde/data/Vorticity_ALL.mat')        
            steps = 151
            n = 449
            m = 199
            dt = 0.2
            dx = 0.02
            dy = 0.02
            
            xmin = 100
            xmax = 425
            ymin = 15
            ymax = 185 
            W = data['VORTALL'].reshape(n,m,steps)   # vorticity
            U = data['UALL'].reshape(n,m,steps)      # x-component of velocity
            V = data['VALL'].reshape(n,m,steps)      # y-component of velocity  
            W = W[xmin:xmax,ymin:ymax,:]
            U = U[xmin:xmax,ymin:ymax,:]
            V = V[xmin:xmax,ymin:ymax,:]
            n,m,steps = W.shape
            
            W = np.transpose(W.reshape(n,m,steps),(2,0,1))  # vorticity
            U = np.transpose(U.reshape(n,m,steps),(2,0,1))      # x-component of velocity
            V = np.transpose(V.reshape(n,m,steps),(2,0,1))      # y-component of velocit
            t_data = np.arange(steps).reshape(( -1,1))*dt 
            x_data = np.arange(n).reshape(( 1,-1,1))*dx
            y_data = np.arange(m).reshape((1,1,-1))*dy
            
            
            steps,n,m = W.shape
            dt =t_data[1,0]-t_data[0,0]
            ut = W
            ut[1:-1,:,:] = (W[2:,:,:] - W[:-2,::])/(2*dt)
            # import pdb;pdb.set_trace()
            if noise_level>0 and training:
                    # ut = ut[math.floor(n*0.1):math.ceil(n*0.9), math.floor(m*0.1):math.ceil(m*0.9)]
                ut = ut[math.floor(steps*0.03):math.ceil(steps*0.97),math.floor(n*0.03):math.ceil(n*0.97), math.floor(m*0.03):math.ceil(m*0.97)]
            X = [x_data, y_data]
            n_input_var = 2
            n_state_var = 3
            sym_true ="add,mul,u2,Diff,u1,x1,add,mul,u3,Diff,u1,x2,add,Diff2,u1,x2,Diff2,u1,x1" 
        elif dataset == "ns_transport_MD_NU":
            center_box = scipy.io.loadmat('./dso/task/pde/data/domain_profile.mat')
            center_box_coordinates  = scipy.io.loadmat('./dso/task/pde/data/domain_coordinates.mat') #127*127*500
            center_UF   = center_box['center_U']
            center_VF   = center_box['center_V']
            center_vorF = center_box['center_vor']
            center_x   = center_box_coordinates['center_x']
            center_y   = center_box_coordinates['center_y']

            max_time   = 500

            U   = center_UF[:,:,0:max_time]#0:max_time-1:10]
            V   = center_VF[:,:,0:max_time]#0:max_time-1:10]
            W = center_vorF[:,:,0:max_time]#:max_time-1:10]
            steps = max_time

            # Cut out 
            xmin = 80
            xmax = 123
            ymin = 2
            ymax = 123
            # t_range = np.linspace(0, 490, 50) 0:steps:10      
            W = W[xmin:xmax,ymin:ymax,:]#.transpose(1,0,2) #t_range]
            U = U[xmin:xmax,ymin:ymax,:]#.transpose(1,0,2)#t_range]
            V = V[xmin:xmax,ymin:ymax,:]#.transpose(1,0,2)#t_range]
            n,m,steps = W.shape
            
            dx = center_x[0,1] - center_x[0,0]
            dy = center_y[1,0] - center_y[0,0]
            print(" dx is ", dx , " dy is ", dy)
            dt = 0.0005
            
            W = np.transpose(W.reshape(n,m,steps),(2,0,1))  # vorticity
            U = np.transpose(U.reshape(n,m,steps),(2,0,1))      # x-component of velocity
            V = np.transpose(V.reshape(n,m,steps),(2,0,1))      # y-component of velocit
            t_data = np.arange(steps).reshape(( -1,1))*dt 
            x_data = np.arange(n).reshape(( 1,-1,1))*dx
            y_data = np.arange(m).reshape((1,1,-1))*dy
            
            # plt.imshow(U[300,:,:])
            # plt.show()    
            # import pdb;pdb.set_trace()        
            steps,n,m = W.shape
            dt =t_data[1,0]-t_data[0,0]
            ut = np.zeros(W.shape)
            ut[:-1,:,:] = (W[1:,:,:] - W[:-1,:,:])/(dt)
            # import pdb;pdb.set_trace()
            if noise_level>0 and training:
                    # ut = ut[math.floor(n*0.1):math.ceil(n*0.9), math.floor(m*0.1):math.ceil(m*0.9)]
                ut = ut[math.floor(steps*cut_ratio):math.ceil(steps*(1-cut_ratio)),math.floor(n*cut_ratio):math.ceil(n*(1-cut_ratio)), math.floor(m*cut_ratio):math.ceil(m*(1-cut_ratio))]
            X = [x_data, y_data]
            n_input_var = 2
            sym_true ="add,mul,u3,Diff,u1,x1,add,mul,u2,Diff,u1,x2,add,Diff2,u1,x2,Diff2,u1,x1"
        else:
            assert False, "Not existed"

        return [W,U,V], X, t_data, ut, sym_true, n_input_var, [None, None],n_state_var
    elif dataset=="rd_3d_MD_NU":
        data = np.load('./dso/task/pde/data/RD_3D.npz')
        x, y, z = data['x'], data['y'], data['z']
        uv = data['uv']
        u = np.transpose(uv[:,:,:,:,0], (3, 0,1,2))
        v = np.transpose(uv[:,:,:,:,1],(3,0,1,2))
        t =data['t']
        dt = t[1]-t[0]
        t_data = t.reshape(-1,1)
        x_data = x.reshape(1,-1,1,1)
        y_data = y.reshape(1,1,-1,1)
        z_data = z.reshape(1,1,1,-1)
        X = [x_data, y_data, z_data]
        sym_true = 'add,mul,u1,n2,u2,add,Diff2_3,u1,x1,add,Diff2_3,u1,x2,Diff2_3,u1,x3'
        # (x0)' = 0.013 1 + -0.996 x0x1x1 + 0.021 x0_33 + 0.020 x0_22 + 0.022 x0_11
        # (x1)' = -0.054 x1 + 1.033 x0x1x1 + 0.010 x1_33 + 0.011 x1_22 + 0.011 x1_11
        # import pdb;pdb.set_trace()
        ut = np.zeros(u.shape)
        vt = np.zeros(v.shape)
        ut[1:,:,:,:] = (u[1:,:,:,:] - u[:-1,:,:,:])/dt
        vt[1:,:,:,:] = (v[1:,:,:,:] - v[:-1,:,:,:])/dt
        n_input_var = 3
        n_state_var =2 
        t_len,n, m,p = u.shape
        if noise_level>0 and training:
            ut = cut_bound(ut,0.05)
            vt = cut_bound(vt,0.05)
        return [u,v], X, t_data, ut, sym_true, n_input_var, [None, None],n_state_var
    
def load_param_data(dataset,noise_level=0, data_amount = 1, training=False,cut_ratio =0.03):
    X=[]
    u_list = []
    if dataset == 'Burgers_param':
        n = 256
        m = 256

        # Set up grid
        x = np.linspace(-8,8,n+1)[:-1];   dx = x[1]-x[0]
        t = np.linspace(0,10,m);          dt = t[1]-t[0]
        k = 2*np.pi*fftfreq(n, d = dx)
        params = (k, -1, 0.1, 0.25)
        # Initial condition
        u0 = np.exp(-(x+1)**2)  
        u = odeint(parametric_burgers_rhs, u0, t, args=(params,)).T
        sym_true = 'add,add,mul,sin,x2,mul,u1,diff,u1,x1,mul,u1,diff,u1,x1,diff2,u1,x1'
        sym_true = 'add,add,mul,const,mul,sin,x2,mul,u1,diff,u1,x1,mul,const,mul,u1,diff,u1,x1,mul,const,diff2,u1,x1'
        n_input_var = 2
        dt = t[1]-t[0]
        xx,tt = np.meshgrid(x,t)
        xx = xx.T
        tt = tt.T
        u_list.append(u)
        
        diff2 = Diff2(u,xx,1)
        udiffu = u*Diff(u,xx,1)
        u_list.append(diff2)
        u_list.append(udiffu)
    else:
        assert False, "wrong dataset"
    n, m = u.shape
    ut = np.zeros((n, m))
   
    X.append(xx)
    X.append(tt)
    X.append("t info")
 
    print("noise level:" , noise_level)
    # import pdb;pdb.set_trace()
    for idx in range(n):
        ut[idx, :] = FiniteDiff(u[idx, :], dt)
    if noise_level>0 and training:
        # ut = ut[math.floor(n*0.1):math.ceil(n*0.9), math.floor(m*0.1):math.ceil(m*0.9)]
        ut = ut[math.floor(n*0.03):math.ceil(n*0.97), math.floor(m*0.03):math.ceil(m*0.97)]    
    
    return u_list,X,t,ut,sym_true, n_input_var,None

def load_subgrid_data(dataset,noise_level=0, data_amount = 1, training=False, data_info=None,cut_ratio = None):
    """
    """
    if dataset == 'eddy_force':
        # lev = data_info['']
        q = ds['q'].data  #.reshape(-1)
        u = ds['u'].data  #.reshape(-1)
        v = ds['v'].data  #.reshape(-1)

        u_list = [q,u,v]
        X = []
        ut = ds['q_subgrid_forcing'].data
        t=None
        n_input_var = 0
        sym_true = "laplacian,adv,u1"
        sym_true =  'add,laplacian,adv,u1,add,laplacian,laplacian,adv,u1,\
add,laplacian,laplacian,laplacian,adv,u1,\
add,laplacian,laplacian,u1,\
add,laplacian,laplacian,laplacian,u1,\
add,adv,adv,ddx,laplacian,u3,\
adv,adv,ddy,laplacian,u2'
        sym_true =  'add,laplacian,laplacian,adv,u1,\
add,laplacian,laplacian,laplacian,adv,u1,\
add,adv,mul,adv,u3,ddx,u1,laplacian,adv,u1'
#         sym_true =  'add,laplacian,adv,u1,add,laplacian,laplacian,adv,u1,\
# add,laplacian,laplacian,laplacian,adv,u1,\
# adv,mul,adv,u3,ddx,u1'
        sym_true =  'add,laplacian,laplacian,adv,u1,\
add,laplacian,laplacian,laplacian,adv,u1,\
add,adv,mul,adv,u3,ddx,u1,\
laplacian,adv,u1'
        print(sym_true.strip())
        # new_u = laplacian(adv(q))
        # u_list.append(new_u)
    return u_list,X,t,ut,sym_true, n_input_var,None

def load_subgrid_test():
    q = ds_test['q'].data  #.reshape(-1)
    u = ds_test['u'].data  #.reshape(-1)
    v = ds_test['v'].data  #.reshape(-1)

    u_list = [q,u,v]
    X = []
    ut = ds_test['q_subgrid_forcing'].data

    return u_list, ut
    


if  __name__ ==  "__main__":
    import time
    st = time.time()
    u = np.random.rand(500,200)
    x = np.random.rand(500,1)
    su = np.sum(Diff3(u,x,0))

