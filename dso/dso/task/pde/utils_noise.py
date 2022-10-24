import numpy as np
# import torch.
import torch
import scipy
from pyDOE import lhs
import matplotlib.pyplot as plt
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

def PolyDiff(u, x, deg = 3, diff = 1, width = 5):
    
    """
    u = values of some function
    x = x-coordinates where values are known
    deg = degree of polynomial to use
    diff = maximum order derivative we want
    width = width of window to fit to polynomial

    This throws out the data close to the edges since the polynomial derivative only works
    well when we're looking at the middle of the points fit.
    """

    u = u.flatten()
    x = x.flatten()

    n = len(x)
    du = np.zeros((n - 2*width,1))

    # Take the derivatives in the center of the domain
    for j in range(width, n-width):

        # Note code originally used an even number of points here.
        # This is an oversight in the original code fixed in 2022.
        points = np.arange(j - width, j + width + 1)

        # Fit to a polynomial
        poly = np.polynomial.chebyshev.Chebyshev.fit(x[points],u[points],deg)

        # Take derivatives

        du[j-width,0] = poly.deriv(m=diff)(x[j])

    return du

def poly_diff(u,dxt,name='x',diff=1, width_x = 20, width_t=10 ):
    n, m = u.shape
    
    m2 = m-2*width_t
    offset_t = width_t
    n2 = n-2*width_x
    offset_x = width_x
    u_out = np.zeros((n,m))
    if name == 't':
        ut = np.zeros((n2,m2))
        dt = dxt[1]-dxt[0]       
        T = np.linspace(0,(m-1)*dt,m)
        for i in range(n2):
            ut[i,:] = PolyDiff(u[i+offset_x,:],T,diff=1,width=width_t,deg=5)[:,0]
        # ut = np.reshape(ut, (n2*m2,1),  order='F')
        
        return ut
    else:
    
        ux = np.zeros((n2,m2))
        dx = dxt[1]-dxt[0] 
        for i in range(m2):
            ux[:,i] = PolyDiff(u[:,i+offset_t],np.linspace(0,(n-1)*dx,n),diff=diff,width=width_x,deg=5)[:,0]
        u[offset_x:n-offset_x,offset_t:m-offset_t] = ux
        return u
        
def ConvSmoother(x, p, sigma):
    """
    Smoother for noisy data

    Inpute = x, p, sigma
    x = one dimensional series to be smoothed
    p = width of smoother
    sigma = standard deviation of gaussian smoothing kernel
    """

    n = len(x)
    y = np.zeros(n)
    g = np.exp(-np.power(np.linspace(-p,p,2*p),2)/(2.0*sigma**2))

    for i in range(n):
        a = max([i-p,0])
        b = min([i+p,n])
        c = max([0, p-i])
        d = min([2*p,p+n-i])
        y[i] = np.sum(np.multiply(x[a:b], g[c:d]))/np.sum(g[c:d])
        
    return y


def load_PI_data(dataset,
                 noise_level,
                 data_ratio,
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
    
    N_f = 50000

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
    # plt.show()
    return X_u_train, u_train_noise, X_f_train, X_u_val, u_val, [lb, ub], [X_star, u_star]