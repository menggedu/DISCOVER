
import numpy as np
# from pde_find import *
from numba import jit,njit


def FiniteDiff(u, dx):
    
    n = u.size
    ux = np.zeros(n)

    # for i in range(1, n - 1):
    ux[1:n-1] = (u[2:n] - u[0:n-2]) / (2 * dx)

    ux[0] = (-3.0 / 2 * u[0] + 2 * u[1] - u[2] / 2) / dx
    ux[n - 1] = (3.0 / 2 * u[n - 1] - 2 * u[n - 2] + u[n - 3] / 2) / dx
    return ux


def FiniteDiff2(u, dx):

    n = u.size
    ux = np.zeros(n)

    ux[1:n-1] = (u[2:n] - 2 * u[1:n-1] + u[0:n-2]) / dx ** 2

    ux[0] = (2 * u[0] - 5 * u[1] + 4 * u[2] - u[3]) / dx ** 2
    ux[n - 1] = (2 * u[n - 1] - 5 * u[n - 2] + 4 * u[n - 3] - u[n - 4]) / dx ** 2
    return ux



# @jit(nopython=True)  
def Diff(u, dxt, dim, name='x'):
    """
    Here dx is a scalar, name is a str indicating what it is
    """

    n, m = u.shape
    uxt = np.zeros((n, m))
    
    if len(dxt.shape) == 2:
        dxt = dxt[:,0]
    if name == 'x':
        dxt = dxt[2]-dxt[1]
        # for i in range(m):
        #     uxt[:, i] = FiniteDiff(u[:, i], dxt)
        uxt[1:n-1,:] = (u[2:n,:] - u[0:n-2,:]) / (2 * dxt)

        uxt[0,:] = (-3.0 / 2 * u[0,:] + 2 * u[1,:] - u[2,:] / 2) / dxt
        uxt[n - 1,:] = (3.0 / 2 * u[n - 1,:] - 2 * u[n - 2,:] + u[n - 3,:] / 2) / dxt
    # elif name == 't':
    #     for i in range(n):
    #         uxt[i, :] = FiniteDiff(u[i, :], dxt)

    else:
        assert False
        NotImplementedError()

    return uxt

# @jit(nopython=True)  
def Diff2(u, dxt, dim, name='x'):
    """
    Here dx is a scalar, name is a str indicating what it is
    """
    if len(dxt.shape) == 2:
        dxt = dxt[:,0]
    n, m = u.shape
    uxt = np.zeros((n, m))
    dxt = dxt[2]-dxt[1]
    if name == 'x':

        uxt[1:n-1,:] =(u[2:n,:] - 2 * u[1:n-1,:] + u[0:n-2,:]) / dxt ** 2

        uxt[0,:] = (2 * u[0,:] - 5 * u[1,:] + 4 * u[2,:] - u[3,:]) / dxt ** 2
        uxt[n - 1,:] = (2 * u[n - 1,:] - 5 * u[n - 2,:] + 4 * u[n - 3,:] - u[n - 4,:]) / dxt ** 2
    else:
        assert False
        NotImplementedError()

    return uxt

# @jit(nopython=True)  
def Diff3(u, dxt, dim, name='x'):
    """
    Here dx is a scalar, name is a str indicating what it is
    """

    n, m = u.shape
    uxt = np.zeros((n, m))

    if name == 'x':
        uxt=Diff2(u,dxt,dim,name)
        uxt = Diff(uxt,dxt, dim, name)
        # dxt = dxt[2]-dxt[1]

        # for i in range(m):
        #     uxt[:, i] = FiniteDiff2(u[:, i], dxt)
        #     uxt[:,i] = FiniteDiff(uxt[:,i],dxt )

    else:
        assert False
        NotImplementedError()

    return uxt

# @jit(nopython=True)  
def Diff4(u, dxt, dim, name='x'):
    """
    Here dx is a scalar, name is a str indicating what it is
    """

    n, m = u.shape
    uxt = np.zeros((n, m))

    
    if name == 'x':
        uxt=Diff2(u,dxt,dim,name)
        uxt = Diff2(uxt,dxt, dim, name)
    else:
        assert False
        NotImplementedError()
 

    return uxt


if  __name__ ==  "__main__":
    import time
    st = time.time()
    u = np.random.rand(500,200)
    x = np.random.rand(500,1)
    su = np.sum(Diff3(u,x,0))
    # import utils
    # su1 = np.sum(utils.Diff3(u,x))
    print(f"time : {time.time()-st}")
    print(su)
    print(su1)