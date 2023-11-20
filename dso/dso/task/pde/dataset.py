"""
load noisy and sparse data for the PINN traing 

"""
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
    elif dataset == 'KS2':
        data = scipy.io.loadmat('./dso/task/pde/data/KS.mat') # course temporal grid    
        t = np.real(data['t'].flatten()[:,None])
        x = np.real(data['x'].flatten()[:,None])
        Exact = np.real(data['usol']).T
        # import pdb;pdb.set_trace()
    elif dataset == 'fisher':
        data=scipy.io.loadmat('./dso/task/pde/data/fisher_nonlin_groundtruth.mat')

        x=np.squeeze(data['x'])[1:-1].reshape(-1,1)
        t=np.squeeze(data['t'])[1:-1].reshape(-1,1)
        Exact=data['U'][1:-1,1:-1]
        
    elif dataset == 'fisher_linear':
        data=scipy.io.loadmat('./dso/task/pde/data/fisher_groundtruth.mat')

        x=np.squeeze(data['x'])[1:-1].reshape(-1,1)
        t=np.squeeze(data['t'])[1:-1].reshape(-1,1)
        Exact=data['U'][1:-1,1:-1]

    elif dataset == 'PDE_divide': 
        Exact=np.load("./dso/task/pde/data/PDE_divide.npy")[1:-1,1:-1]
        nx = 100
        nt = 251
        x=np.linspace(1,2,nx)[1:-1].reshape(-1,1)
        t=np.linspace(0,1,nt)[1:-1].reshape(-1,1)
        # t firstaction
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
    X_f_train = np.vstack((X_f_train, X_u_train))
    
    # Option: Add noise
    noise = noise_level
    print("noise", noise)

    u_train_noise = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1]) 
    u_val = u_val + noise*np.std(u_val)*np.random.randn(u_val.shape[0], u_val.shape[1])
    # u_train_noise = add_noise(u_train)
    # u_val = add_noise(u_val)

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

    return X_u_train, u_train_noise, X_f_train, X_u_val, u_val, [lb, ub], [X_star, u_star], [Exact.shape]

def add_noise(un):
    noise_level = 1
    for j in range(un.shape[0]):
        for i in range(un.shape[1]):
            un[j,i]=un[j,i]*(1+0.01*noise_level*np.random.uniform(-1,1))
    return un

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
    # ut = (gPAR2[:,1:]-gPAR2[:,:-1])/dt
    # ut = (gPAR6[:,1:]-gPAR6[:,:-1])/dt
    # ut= ut[30:54,:]
    X.append(x)
    u = gPAR2 
    sym_true ='add,u1,mul,n2,u1,u2'
    n_input_var=1
    return [gPAR2[:,:-1], gPAR6[:,:-1], gv[:,:-1]],X,t,ut, sym_true,n_input_var, None

def load_2d2U_data(dataset,
                 noise_level,
                 data_ratio,
                 pic_path,
                 coll_num = 100000,
                 spline_sample = False
                 ):
    if dataset == 'Allen_Cahn_2D':
        path = './dso/task/pde/data/bcpinn_ac.npz'
        data = np.load(path)
        U = data['u'].transpose(1,2,0) # x,y,t

        dx = 1/64
        dy =dx
        h=dx
        y_data = np.linspace(-0.5 * h, 1 + h * 0.5, 64 + 2)[1:-1].reshape(-1,1)
        x_data = np.linspace(-0.5 * h, 1 + h * 0.5, 64 + 2)[1:-1].reshape(-1,1)
        t_data = np.linspace(0,5,101).reshape(-1,1)
        n,m,steps = U.shape

        dt =t_data[1,0]-t_data[0,0]

        w_data = U.reshape(n*m, steps)

        # t_data = np.arange(steps).reshape((1, -1))*dt         
        t_data = np.tile(t_data,(m*n,1))
        
        # This part reset the coordinates
        # x_data = np.arange(n).reshape((-1, 1))*dx 
        x_data = np.tile(x_data, (1, m))
        x_data = np.reshape(x_data, (-1, 1))
        x_data = np.tile(x_data, (1, steps))
        
        # y_data = np.arange(m).reshape((1, -1))*dy 
        y_data = np.tile(y_data, (n, 1))
        y_data = np.reshape(y_data, (-1, 1))
        y_data = np.tile(y_data, (1, steps))
        
        # coll data
        # Preprocess data #2(compatible with NN format)
        t_star = np.reshape(t_data,(-1,1))
        x_star = np.reshape(x_data,(-1,1))
        y_star = np.reshape(y_data,(-1,1))        
        w_star = np.reshape(w_data,(-1,1))
        # import pdb;pdb.set_trace()
        X_star = np.hstack((x_star, y_star, t_star))
        
        # ordinary sample
        total_num = len(X_star)
        sample_num = int(total_num*data_ratio)
        print(f"random sample number: {sample_num} ")
        ID = np.random.choice(total_num, sample_num, replace = False)
        X_meas = X_star[ID,:]
        w_meas = w_star[ID,:]    
        
        # Training measurements
        Split_TrainVal = 0.8
        N_train = int(sample_num*Split_TrainVal)
        idx_train = np.random.choice(X_meas.shape[0], N_train, replace=False)
        X_train = X_meas[idx_train,:]
        w_train = w_meas[idx_train,:]        
        
        # Validation Measurements, which are the rest of measurements
        idx_val = np.setdiff1d(np.arange(X_meas.shape[0]), idx_train, assume_unique=True)
        X_val = X_meas[idx_val,:]
        w_val = w_meas[idx_val,:]    
    
        # Doman bounds        
        lb = X_star.min(0)
        ub = X_star.max(0)    
        
        # Collocation points
        N_f = coll_num
        X_f = lb + (ub-lb*2)*lhs(3, N_f)
        X_f = np.vstack((X_f, X_train))
        
        # add noise
        w_train = w_train + noise_level*np.std(w_train)*np.random.randn(w_train.shape[0], w_train.shape[1])
        w_val = w_val + noise_level*np.std(w_val)*np.random.randn(w_val.shape[0], w_val.shape[1])

        return X_train, w_train, X_f, X_val, w_val, [lb, ub], [X_star, w_star],[w_star.shape]
        # X, Y, T = np.meshgrid(x, y, t)  
        # sym_true = 'add,add,Diff2,u1,x1,Diff2,u1,x2,sub,u1,n3,u1'

    elif "rd" == dataset: 
        data = scipy.io.loadmat('./dso/task/pde/data/reaction_diffusion_standard.mat') # grid 256*256*201
            
        t = np.real(data['t'].flatten()[:,None])
        x = np.real(data['x'].flatten()[:,None])
        y = np.real(data['y'].flatten()[:,None])
        Exact_u = data['u']
        Exact_v = data['v']

        X, Y, T = np.meshgrid(x, y, t)
        X_star = np.hstack((X.flatten()[:,None], Y.flatten()[:,None], T.flatten()[:,None]))
        u_star = Exact_u.flatten()[:,None] 
        v_star = Exact_v.flatten()[:,None]              
    
        # Doman bounds
        lb = X_star.min(0)
        ub = X_star.max(0)    
                    
        # Measurements: Spatially random but temporally continuous
        N_uv_s =3600
        
        # Use these commands when N_uv_s is larger than X.shape[0] or X.shape[1]
        idx = np.random.choice(X.shape[0]*X.shape[1], N_uv_s, replace = False)
        idx_remainder = idx%(X.shape[0])
        idx_s_y = np.floor(idx/(X.shape[0]))
        idx_s_y = idx_s_y.astype(np.int32)
        idx_idx_remainder = np.where(idx_remainder == 0)[0]
        idx_remainder[idx_idx_remainder] = X.shape[0]
        idx_s_x = idx_remainder-1            
                
        # Random sample temporally
        N_t_s = 50
        # import pdb;pdb.set_trace()
        idx_t = np.random.choice(X.shape[2], N_t_s, replace=False)
        idx_t = idx_t.astype(np.int32)
        
        X1 = X[idx_s_x, idx_s_y, :]
        X2 = X1[:, idx_t]
        Y1 = Y[idx_s_x, idx_s_y, :]
        Y2 = Y1[:, idx_t]
        T1 = T[idx_s_x, idx_s_y, :]
        T2 = T1[:, idx_t]
        Exact_u1 = Exact_u[idx_s_x, idx_s_y, :]
        Exact_u2 = Exact_u1[:, idx_t]
        Exact_v1 = Exact_v[idx_s_x, idx_s_y, :]
        Exact_v2 = Exact_v1[:, idx_t]
        
        X_star_meas = np.hstack((X2.flatten()[:,None], Y2.flatten()[:,None],
                                  T2.flatten()[:,None]))
        u_star_meas = Exact_u2.flatten()[:,None] 
        v_star_meas = Exact_v2.flatten()[:,None] 
        
        # Training measurements, which are randomly sampled spatio-temporally
        Split_TrainVal = 0.8
        N_u_train = int(N_uv_s*N_t_s*Split_TrainVal)
        idx_train = np.random.choice(X_star_meas.shape[0], N_u_train, replace=False)
        X_star_train = X_star_meas[idx_train,:]
        u_star_train = u_star_meas[idx_train,:]
        v_star_train = v_star_meas[idx_train,:]
        
        # Validation Measurements, which are the rest of measurements
        idx_val = np.setdiff1d(np.arange(X_star_meas.shape[0]), idx_train, assume_unique=True)
        X_star_val = X_star_meas[idx_val,:]
        u_star_val = u_star_meas[idx_val,:]
        v_star_val = v_star_meas[idx_val,:]
                
        # Collocation points
        N_f = coll_num
#       
        X_f = lb + (ub-lb)*lhs(3, N_f)
        # X_f = X_f_pseu 
       
        # add noise
        u_star_train = u_star_train + noise_level*np.std(u_star_train)*np.random.randn(u_star_train.shape[0], u_star_train.shape[1])
        v_star_train = v_star_train + noise_level*np.std(v_star_train)*np.random.randn(v_star_train.shape[0], v_star_train.shape[1])
        u_star_val = u_star_val + noise_level*np.std(u_star_val)*np.random.randn(u_star_val.shape[0], u_star_val.shape[1])
        v_star_val = v_star_val + noise_level*np.std(v_star_val)*np.random.randn(v_star_val.shape[0], v_star_val.shape[1])
                

        X_star_train = X_star_train.astype(np.float32)
        u_star_train = u_star_train.astype(np.float32)
        v_star_train = v_star_train.astype(np.float32)
        X_f = X_f.astype(np.float32)
        X_star_val = X_star_val.astype(np.float32)
        u_star_val = u_star_val.astype(np.float32)
        v_star_val = v_star_val.astype(np.float32)
        
        # concat
        uv_star_train = np.concatenate([u_star_train,v_star_train], axis = 1)
        uv_star = np.concatenate([u_star,v_star], axis = 1)
        uv_star_val = np.concatenate([u_star_val,v_star_val], axis = 1)
        return X_star_train,uv_star_train , X_f, X_star_val, uv_star_val, [lb, ub], [X_star, uv_star], [uv_star.shape]
    
    elif dataset == 'rd_3d_MD_NU':
        data = np.load('./dso/task/pde/data/RD_3D.npz')
        x, y, z = data['x'], data['y'], data['z']
        uv = data['uv']
        Exact_u =uv[:,:,:,25:75,0]
        Exact_v = uv[:,:,:,25:75,1]
        t =data['t'][25:75]
        dt = t[1]-t[0]
        t_data = t.reshape(-1,1)
        x_data = x.reshape(-1,1)
        y_data = y.reshape(-1,1)
        z_data = z.reshape(-1,1)
        
        # x_f,y_f = np.meshgrid(x, y)
        # t_f = np.ones(x_f.shape[0]*x_f.shape[1])*0.5
        # X_f_pseu = np.hstack((x_f.flatten()[:,None], y_f.flatten()[:,None], t_f.flatten()[:,None]))
        
        X, Y, Z, T = np.meshgrid(x_data, y_data,z_data, t)
        X_star = np.hstack((X.flatten()[:,None], Y.flatten()[:,None],Z.flatten()[:,None],  T.flatten()[:,None]))
        u_star = Exact_u.flatten()[:,None] 
        v_star = Exact_v.flatten()[:,None]              
            
        # ordinary sample
        total_num = len(X_star)
        sample_num = int(total_num*data_ratio)
        print(f"random sample number: {sample_num} ")
        ID = np.random.choice(total_num, sample_num, replace = False)
        X_meas = X_star[ID,:]   
        u_meas = u_star[ID,:]
        v_meas = v_star[ID,:]
        # bounds
        lb = X_star.min(0)
        ub = X_star.max(0)    
        mid = (ub-lb)*0.05
        lb+=mid
        ub-=mid
        # Training measurements
        Split_TrainVal = 0.8
        N_train = int(sample_num*Split_TrainVal)
        idx_train = np.random.choice(X_meas.shape[0], N_train, replace=False)
        X_train = X_meas[idx_train,:]
        u_train = u_meas[idx_train,:]        
        v_train = v_meas[idx_train,:]    
        # Validation Measurements, which are the rest of measurements
        idx_val = np.setdiff1d(np.arange(X_meas.shape[0]), idx_train, assume_unique=True)
        X_val = X_meas[idx_val,:]
        u_val = u_meas[idx_val,:]    
        v_val = v_meas[idx_val,:]
        
        # Collocation points
        N_f = coll_num
        X_f = lb + (ub-lb)*lhs(4, N_f)
        X_f = np.vstack((X_f, X_train))
        
        # add noise
        u_train = u_train + noise_level*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
        u_val = u_val + noise_level*np.std(u_val)*np.random.randn(u_val.shape[0],u_val.shape[1])
        v_train = v_train + noise_level*np.std(v_train)*np.random.randn(v_train.shape[0], v_train.shape[1])
        v_val = v_val + noise_level*np.std(v_val)*np.random.randn(v_val.shape[0], v_val.shape[1])
        # Training measurements, which are randomly sampled spatio-temporally
       
        X_star_train = X_train.astype(np.float32)
        u_star_train = u_train.astype(np.float32)
        v_star_train = v_train.astype(np.float32)
        X_f = X_f.astype(np.float32)
       
        X_star_val = X_val.astype(np.float32)
        u_star_val = u_val.astype(np.float32)
        v_star_val = v_val.astype(np.float32)        
        # concat
        uv_star_train = np.concatenate([u_star_train,v_star_train], axis = 1)
        uv_star = np.concatenate([u_star,v_star], axis = 1)
        uv_star_val = np.concatenate([u_star_val,v_star_val], axis = 1)
        return X_star_train, uv_star_train, X_f, X_star_val, uv_star_val, [lb, ub], [X_star, uv_star], [uv_star.shape[0]]    
    elif 'ns' in dataset:
    
        if dataset == "ns_MD_NU":
            data = scipy.io.loadmat('./dso/task/pde/data/Vorticity_ALL.mat')        
            steps = 151
            n = 449
            m = 199
            dt = 0.2
            dx = 0.02
            dy = 0.02
            
            W = data['VORTALL'].reshape(n,m,steps)   # vorticity
            U = data['UALL'].reshape(n,m,steps)      # x-component of velocity
            V = data['VALL'].reshape(n,m,steps)      # y-component of velocity
 
            xmin = 100
            xmax = 425 #325
            ymin = 15
            ymax = 185  #      
            W = W[xmin:xmax,ymin:ymax,:]
            U = U[xmin:xmax,ymin:ymax,:]
            V = V[xmin:xmax,ymin:ymax,:]
            n,m,steps = W.shape
            
            # shape = [x,y,t]
            w_data = W.reshape(n*m, steps)
            u_data = U.reshape(n*m, steps)
            v_data = V.reshape(n*m, steps)  #(y,x,t)
            t_data = np.arange(steps).reshape((1, -1))*dt         
            t_data = np.tile(t_data,(m*n,1))
            
            # This part reset the coordinates
            x_data = np.arange(n).reshape((-1, 1))*dx 
            x_data = np.tile(x_data, (1, m))
            x_data = np.reshape(x_data, (-1, 1))
            x_data = np.tile(x_data, (1, steps))
            
            y_data = np.arange(m).reshape((1, -1))*dy 
            y_data = np.tile(y_data, (n, 1))
            y_data = np.reshape(y_data, (-1, 1))
            y_data = np.tile(y_data, (1, steps))
            # Preprocess data #2(compatible with NN format)
            t_star = np.reshape(t_data,(-1,1))
            x_star = np.reshape(x_data,(-1,1))
            y_star = np.reshape(y_data,(-1,1))        
            u_star = np.reshape(u_data,(-1,1))
            v_star = np.reshape(v_data,(-1,1))
            w_star = np.reshape(w_data,(-1,1))
            
            X_star = np.hstack((x_star, y_star, t_star))
            # import pdb;pdb.set_trace()                    
            ## Spatially randomly but temporally continuously sampled measurements
            N_s = 1000        
            N_t = 60
            # if steps<N_t:
            #     N_t = steps
            idx_s = np.random.choice(x_data.shape[0], N_s, replace = False)
            idx_t = np.random.choice(steps, N_t, replace = False)
            
            
            t_meas = t_data[idx_s, :]
            t_meas = t_meas[:, idx_t].reshape((-1,1))
            x_meas = x_data[idx_s, :]
            x_meas = x_meas[:, idx_t].reshape((-1,1))
            y_meas = y_data[idx_s, :]
            y_meas = y_meas[:, idx_t].reshape((-1,1))
            u_meas = u_data[idx_s, :]
            u_meas = u_meas[:, idx_t].reshape((-1,1))
            v_meas = v_data[idx_s, :]
            v_meas = v_meas[:, idx_t].reshape((-1,1))
            w_meas = w_data[idx_s, :]
            w_meas = w_meas[:, idx_t].reshape((-1,1))
            
            
            X_meas = np.hstack((x_meas, y_meas, t_meas))
            
            # Training measurements
            Split_TrainVal = 0.8
            N_train = int(N_s*N_t*Split_TrainVal)
            idx_train = np.random.choice(X_meas.shape[0], N_train, replace=False)
            X_train = X_meas[idx_train,:]
            u_train = u_meas[idx_train,:]
            v_train = v_meas[idx_train,:]
            w_train = w_meas[idx_train,:]        
            
            # Validation Measurements, which are the rest of measurements
            idx_val = np.setdiff1d(np.arange(X_meas.shape[0]), idx_train, assume_unique=True)
            X_val = X_meas[idx_val,:]
            u_val = u_meas[idx_val,:]
            v_val = v_meas[idx_val,:]
            w_val = w_meas[idx_val,:]       
        
            # Doman bounds        
            lb = X_star.min(0)
            ub = X_star.max(0)    
            
            # Collocation points
            N_f = coll_num
            X_f = lb + (ub-lb)*lhs(3, N_f)
            X_f = np.vstack((X_f, X_train))
            
            # add noise
            
            u_train = u_train + noise_level*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
            v_train = v_train + noise_level*np.std(v_train)*np.random.randn(v_train.shape[0], v_train.shape[1])
            w_train = w_train + noise_level*np.std(w_train)*np.random.randn(w_train.shape[0], w_train.shape[1])
            u_val = u_val + noise_level*np.std(u_val)*np.random.randn(u_val.shape[0], u_val.shape[1])
            v_val = v_val + noise_level*np.std(v_val)*np.random.randn(v_val.shape[0], v_val.shape[1])
            w_val = w_val + noise_level*np.std(w_val)*np.random.randn(w_val.shape[0], w_val.shape[1])
            
            # concat
            uvw_star_train = np.concatenate([w_train,u_train,v_train], axis = 1)
            uv_star = np.concatenate([w_star,u_star,v_star], axis = 1)
            uvw_star_val = np.concatenate([w_val,u_val,v_val], axis = 1)
            return X_train,uvw_star_train , X_f, X_val, uvw_star_val, [lb, ub], [X_star, uv_star], [(n,m,steps)]
        
        elif dataset == 'ns_transport_MD_NU':
            
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
            W = center_vorF[:,:,0:max_time]#0:max_time-1:10]
            steps = max_time
            
            # Cut out 
            xmin = 80
            xmax = 123
            ymin = 2
            ymax = 123
            # t_range = np.linspace(0, 490, 50)   0:steps:10    
            W = W[xmin:xmax,ymin:ymax,:].transpose(1,0,2) #t_range]
            U = U[xmin:xmax,ymin:ymax,:].transpose(1,0,2)#t_range]
            V = V[xmin:xmax,ymin:ymax,:].transpose(1,0,2)#t_range]
            n,m,steps = W.shape
            # import pdb;pdb.set_trace()
            
            dx = center_x[0,1] - center_x[0,0]
            dy = center_y[1,0] - center_y[0,0]
            # print(" dx is ", dx , " dy is ", dy)
            dt = 0.0005
            w_data = W.reshape(n*m, steps)
            u_data = U.reshape(n*m, steps)
            v_data = V.reshape(n*m, steps)  #(y,x,t)

            t_data = np.arange(steps).reshape((1, -1))*dt         
            t_data = np.tile(t_data,(m*n,1))
            
            # This part reset the coordinates
            x_data = np.arange(n).reshape((-1, 1))*dx 
            x_data = np.tile(x_data, (1, m))
            x_data = np.reshape(x_data, (-1, 1))
            x_data = np.tile(x_data, (1, steps))
            
            y_data = np.arange(m).reshape((1, -1))*dy 
            y_data = np.tile(y_data, (n, 1))
            y_data = np.reshape(y_data, (-1, 1))
            y_data = np.tile(y_data, (1, steps))
            
            # coll data
            # t_f = t_data.reshape(n,m,steps)[xmin+10:xmax-10,ymin+10:ymax+10,0:499:10]
            # x_f = x_data.reshape(n,m,steps)[xmin+10:xmax-10,ymin+10:ymax+10,0:499:10]
            # y_f = y_data.reshape(n,m,steps)[xmin+10:xmax-10,ymin+10:ymax+10,0:499:10]
            # X_f = np.hstack((x_f, y_f, t_f))
            # Preprocess data #2(compatible with NN format)
            t_star = np.reshape(t_data,(-1,1))
            x_star = np.reshape(x_data,(-1,1))
            y_star = np.reshape(y_data,(-1,1))        
            u_star = np.reshape(u_data,(-1,1))
            v_star = np.reshape(v_data,(-1,1))
            w_star = np.reshape(w_data,(-1,1))
            
            X_star = np.hstack((x_star, y_star, t_star))
            
            # ordinary sample
            total_num = len(X_star)
            sample_num = int(total_num*data_ratio)
            print(f"random sample number: {sample_num} ")
            ID = np.random.choice(total_num, sample_num, replace = False)
            X_meas = X_star[ID,:]
            w_meas = w_star[ID,:]    
            u_meas = u_star[ID,:]
            v_meas = v_star[ID,:]
            
            #spline
            # N_s = 1000        
            # N_t = 100
            # if steps<N_t:
            #     N_t = steps
            # idx_s = np.random.choice(x_data.shape[0], N_s, replace = False)
            # idx_t = np.random.choice(steps, N_t, replace = False)
            
            
            # t_meas = t_data[idx_s, :]
            # t_meas = t_meas[:, idx_t].reshape((-1,1))
            # x_meas = x_data[idx_s, :]
            # x_meas = x_meas[:, idx_t].reshape((-1,1))
            # y_meas = y_data[idx_s, :]
            # y_meas = y_meas[:, idx_t].reshape((-1,1))
            # u_meas = u_data[idx_s, :]
            # u_meas = u_meas[:, idx_t].reshape((-1,1))
            # v_meas = v_data[idx_s, :]
            # v_meas = v_meas[:, idx_t].reshape((-1,1))
            # w_meas = w_data[idx_s, :]
            # w_meas = w_meas[:, idx_t].reshape((-1,1))
            # X_meas = np.hstack((x_meas, y_meas, t_meas))
            
            # Training measurements
            Split_TrainVal = 0.8
            N_train = int(sample_num*Split_TrainVal)
            idx_train = np.random.choice(X_meas.shape[0], N_train, replace=False)
            X_train = X_meas[idx_train,:]
            u_train = u_meas[idx_train,:]
            v_train = v_meas[idx_train,:]
            w_train = w_meas[idx_train,:]        
            
            # Validation Measurements, which are the rest of measurements
            idx_val = np.setdiff1d(np.arange(X_meas.shape[0]), idx_train, assume_unique=True)
            X_val = X_meas[idx_val,:]
            u_val = u_meas[idx_val,:]
            v_val = v_meas[idx_val,:]
            w_val = w_meas[idx_val,:]       
        
            # Doman bounds        
            lb = X_star.min(0)
            ub = X_star.max(0)    
            
            # Collocation points
            N_f = coll_num
            X_f = 2*lb + (ub-lb*2)*lhs(3, N_f)
            X_f = np.vstack((X_f, X_train))
            
            # add noise
            
            u_train = u_train + noise_level*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
            v_train = v_train + noise_level*np.std(v_train)*np.random.randn(v_train.shape[0], v_train.shape[1])
            w_train = w_train + noise_level*np.std(w_train)*np.random.randn(w_train.shape[0], w_train.shape[1])
            u_val = u_val + noise_level*np.std(u_val)*np.random.randn(u_val.shape[0], u_val.shape[1])
            v_val = v_val + noise_level*np.std(v_val)*np.random.randn(v_val.shape[0], v_val.shape[1])
            w_val = w_val + noise_level*np.std(w_val)*np.random.randn(w_val.shape[0], w_val.shape[1])
            
            # concat
            uvw_star_train = np.concatenate([w_train,u_train,v_train], axis = 1)
            uv_star = np.concatenate([w_star,u_star,v_star], axis = 1)
            uvw_star_val = np.concatenate([w_val,u_val,v_val], axis = 1)
            return X_train,uvw_star_train , X_f, X_val, uvw_star_val, [lb, ub], [X_star, uv_star]
        else:
            assert False, "Unknown dataset"