
import numpy as np
import scipy.io as scio
import pickle
def load_data(dataset):
    if dataset == 'chafee-infante': # 301*200的新数据
        u = np.load("./dso/task/pde/data/chafee_infante_CI.npy")
        x = np.load("./dso/task/pde/data/chafee_infante_x.npy").reshape(-1,1)
        t = np.load("./dso/task/pde/data/chafee_infante_t.npy").reshape(-1,1)
        n_input_var = 1
        sym_true = 'add,add,u,n3,u,diff2,u,x1'
        # right_side = 'right_side = uxx-u+u**3'
    elif dataset == 'Burgers':
        data = scio.loadmat('./dso/task/pde/data/burgers.mat')
        u=data.get("usol")
        x=np.squeeze(data.get("x")).reshape(-1,1)
        t=np.squeeze(data.get("t").reshape(-1,1))
        sym_true = 'add,mul,u,diff,u,x1,diff2,u,x1'
        right_side_origin = 'right_side_origin = -1*u_origin*ux_origin+0.1*uxx_origin'
        n_input_var = 1

   # Kdv -0.0025uxxx-uux
    elif dataset == 'Kdv':
        data = scio.loadmat('./dso/task/pde/data/Kdv.mat')
        u=data.get("uu")
        x=np.squeeze(data.get("x")).reshape(-1,1)
        t=np.squeeze(data.get("tt").reshape(-1,1))
        sym_true = 'add,mul,u,diff,u,x1,diff3,u,x1'
        right_side_origin = 'right_side_origin = -0.0025*uxxx_origin-u_origin*ux_origin'
        n_input_var = 1

    elif dataset == 'PDE_divide':

        u=np.load("./dso/task/pde/data/PDE_divide.npy").T
        nx = 100
        nt = 251
        x=np.linspace(1,2,nx).reshape(-1,1)
        t=np.linspace(0,1,nt).reshape(-1,1)
        sym_true = 'add,div,diff,u,x1,x1,diff2,u,x1'
        right_side_origin = 'right_side_origin = -config.divide(ux_origin, x_all) + 0.25*uxx_origin'
        n_input_var = 1

    elif dataset == 'PDE_compound':
        u=np.load("./dso/task/pde/data/PDE_compound.npy").T
        nx = 100
        nt = 251
        x=np.linspace(1,2,nx).reshape(-1,1)
        t=np.linspace(0,0.5,nt).reshape(-1,1)
        n, m = u.shape 
        u = u[int(n*0.1):int(n*0.9), int(m*0):int(m*1)]
        x = x[int(n*0.1):int(n*0.9)]
        t = t[int(m*0):int(m*1)]
        sym_true = 'add,mul,u,diff2,u,x1,mul,diff,u,x1,diff,u,x1'
        right_side_origin = 'right_side_origin = u_origin*uxx_origin + ux_origin*ux_origin'
        n_input_var = 1
    elif dataset == 'Cahn_Hilliard':
        path='./dso/task/pde/data/ch.npz'
        data = np.load(path)
        u = data['u']
        n,m= u.shape[1:]
        x,y=data['x'].data['y']
        t= data['t']
        # u = u[int(n*0.1):int(n*0.9), int(m*0.1):int(m*0.9)]
        # x = x[int(n*0.1):int(n*0.9)]
        # t = t[int(m*0.1):int(m*0.9)]
        n_input_var = 2
        sym_true = 'add,diff2,sub,n3,u,u,x1,diff4,u,x1'
    else:
        assert False, "Unknown dataset"

    return u,x,t,sym_true, n_input_var

def FiniteDiff(u, dx):
    
    n = u.size
    ux = np.zeros(n)

    # for i in range(1, n - 1):
    ux[1:n-1] = (u[2:n] - u[0:n-2]) / (2 * dx)

    ux[0] = (-3.0 / 2 * u[0] + 2 * u[1] - u[2] / 2) / dx
    ux[n - 1] = (3.0 / 2 * u[n - 1] - 2 * u[n - 2] + u[n - 3] / 2) / dx
    return ux


def FiniteDiff2(u, dx):
    # import pdb;pdb.set_trace()
    n = u.size
    ux = np.zeros(n)

    # for i in range(1, n - 1):
    ux[1:n-1] = (u[2:n] - 2 * u[1:n-1] + u[0:n-2]) / dx ** 2

    ux[0] = (2 * u[0] - 5 * u[1] + 4 * u[2] - u[3]) / dx ** 2
    ux[n - 1] = (2 * u[n - 1] - 5 * u[n - 2] + 4 * u[n - 3] - u[n - 4]) / dx ** 2
    return ux


def Diff(u, dxt, name='x'):
    """
    Here dx is a scalar, name is a str indicating what it is
    """

    n, m = u.shape
    uxt = np.zeros((n, m))

    if name == 'x':
        dxt = dxt[2]-dxt[1]
        for i in range(m):
            uxt[:, i] = FiniteDiff(u[:, i], dxt)

    elif name == 't':
        for i in range(n):
            uxt[i, :] = FiniteDiff(u[i, :], dxt)

    else:
        NotImplementedError()

    return uxt


def Diff2(u, dxt, name='x'):
    """
    Here dx is a scalar, name is a str indicating what it is
    """

    n, m = u.shape
    uxt = np.zeros((n, m))

    if name == 'x':
        dxt = dxt[2]-dxt[1]
        for i in range(m):
            uxt[:, i] = FiniteDiff2(u[:, i], dxt)

    elif name == 't':
        for i in range(n):
            uxt[i, :] = FiniteDiff2(u[i, :], dxt)

    else:
        NotImplementedError()

    return uxt

def Diff3(u, dxt, name='x'):
    """
    Here dx is a scalar, name is a str indicating what it is
    """

    n, m = u.shape
    uxt = np.zeros((n, m))

    if name == 'x':
        dxt = dxt[2]-dxt[1]
        for i in range(m):
            uxt[:, i] = FiniteDiff2(u[:, i], dxt)
            uxt[:,i] = FiniteDiff(uxt[:,i],dxt )
    elif name == 't':
        for i in range(n):
            uxt[i, :] = FiniteDiff2(u[i, :], dxt)
            uxt[i,:] = FiniteDiff(uxt[:,i],dxt )

    else:
        NotImplementedError()

    return uxt

def Diff4(u, dxt, name='x'):
    """
    Here dx is a scalar, name is a str indicating what it is
    """

    n, m = u.shape
    uxt = np.zeros((n, m))

    if name == 'x':
        dxt = dxt[2]-dxt[1]
        for i in range(m):
            uxt[:, i] = FiniteDiff2(u[:, i], dxt)
            uxt[:,i] = FiniteDiff2(uxt[:,i],dxt )
    elif name == 't':
        for i in range(n):
            uxt[i, :] = FiniteDiff2(u[i, :], dxt)
            uxt[i,:] = FiniteDiff2(uxt[:,i],dxt )

    else:
        NotImplementedError()

    return uxt
if  __name__ ==  "__main__":
    import time
    st = time.time()
    u = np.random.rand(500,200)
    x = np.random.rand(500,1)
    su = np.sum(Diff3(u,x))
    import utils
    su1 = np.sum(utils.Diff3(u,x))
    print(f"time : {time.time()-st}")
    print(su)
    print(su1)