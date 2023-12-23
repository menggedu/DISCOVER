
import numpy as np

from numba import jit,njit


# @jit(nopython=True)
def Diff_3(u, dxt, name=1):
    """
    Here dx is a scalar, name is a str indicating what it is
    """

    # import pdb;pdb.set_trace()
    if u.shape == dxt.shape:
        return u/dxt
    t,n,m,p = u.shape
    uxt = np.zeros((t, n, m, p))
    if len(dxt.shape) == 2:
        import pdb;pdb.set_trace()
        dxt = dxt[:,0]
    dxt = dxt.ravel()
    # import pdb;pdb.set_trace()
    dxt = dxt[2]-dxt[1]
    if name == 1:
        uxt[:,1:n-1,:,:] = (u[:,2:n,:,:]-u[:,:n-2,:,:])/2/dxt
        
        # uxt[:,0,:,:] = (u[:,1,:,:]-u[:,-1,:,:])/2/dxt
        # uxt[:,-1,:,:] = (u[:,0,:,:]-u[:,-2,:,:])/2/dxt
        uxt[:,0,:,:] = (-3.0 / 2 * u[:,0,:,:] + 2 * u[:,1,:,:] - u[:,2,:,:] / 2) / dxt
        uxt[:,n - 1,:,:] = (3.0 / 2 * u[:,n - 1,:,:] - 2 * u[:,n - 2,:,:] + u[:,n - 3,:,:] / 2) / dxt
    elif name == 2:
        
        uxt[:,:,1:m-1,:] = (u[:,:,2:m,:]-u[:,:,:m-2,:])/2/dxt
        
        # uxt[:,:,0,:] = (u[:,:,1,:]-u[:,:,-1,:])/2/dxt
        # uxt[:,:,-1,:] = (u[:,:,0,:]-u[:,:,-2,:])/2/dxt
        uxt[:,:,0,:] = (-3.0 / 2 * u[:,:,0,:] + 2 * u[:,:,1,:] - u[:,:,2,:] / 2) / dxt
        uxt[:,:,m- 1,:] = (3.0 / 2 * u[:,:,m - 1,:] - 2 * u[:,:,m - 2,:] + u[:,:,m - 3,:] / 2) / dxt
    elif name == 3:
        uxt[:,:,:,1:p-1] = (u[:,:,:,2:p]-u[:,:,:,:p-2])/2/dxt
        uxt[:,:,:,0] = (-3.0 / 2 * u[:,:,:,0] + 2 * u[:,:,:,1] - u[:,:,:,2] / 2) / dxt
        uxt[:,:,:,p - 1] = (3.0 / 2 * u[:,:,:,p - 1] - 2 * u[:,:,:,p - 2] + u[:,:,:,p - 3] / 2) / dxt
    else:
        assert False, 'not supported'     

    return uxt

# @jit(nopython=True)
def Diff2_3(u, dxt, name=1): 
    """
    Here dx is a scalar, name is a str indicating what it is
    """
  
    
    if u.shape == dxt.shape:
        return u/dxt
    t,n,m,p = u.shape
    uxt = np.zeros((t, n, m,p))
    dxt = dxt.ravel()
    # try: 
    dxt = dxt[2]-dxt[1]
    if name == 1:
        
        uxt[:,1:n-1,:,:]= (u[:,2:n,:,:] - 2 * u[:,1:n-1,:,:] + u[:,0:n-2,:,:]) / dxt ** 2
        # uxt[:,0,:,:] = (u[:,1,:,:]+u[:,-1,:,:]-2*u[:,0,:,:])/dxt ** 2
        # uxt[:,-1,:,:] = (u[:,0,:,:]+u[:,-2,:,:]-2*u[:,-1,:,:])/dxt ** 2
        uxt[:,0,:,:] = (2 * u[:,0,:,:] - 5 * u[:,1,:,:] + 4 * u[:,2,:,:] - u[:,3,:,:]) / dxt ** 2
        uxt[:,n - 1,:,:] = (2 * u[:,n - 1,:,:] - 5 * u[:,n - 2,:,:] + 4 * u[:,n - 3,:,:] - u[:,n - 4,:,:]) / dxt ** 2
    elif name == 2:

        uxt[:,:,1:m-1,:]= (u[:,:,2:m,:] - 2 * u[:,:,1:m-1,:] + u[:,:,0:m-2,:]) / dxt ** 2
        # uxt[:,:,0,:] = (u[:,:,1,:]+u[:,:,-1,:]-2*u[:,:,0,:,:])/dxt ** 2
        # uxt[:,:,-1,:] = (u[:,:,0,:]+u[:,:,-2,:]-2*u[:,:,-1,:])/dxt ** 2  
        uxt[:,:,0,:] = (2 * u[:,:,0,:] - 5 * u[:,:,1,:] + 4 * u[:,:,2,:] - u[:,:,3,:]) / dxt ** 2
        uxt[:,:,m - 1,:] = (2 * u[:,:,m - 1,:] - 5 * u[:,:,m - 2,:] + 4 * u[:,:,m - 3,:] - u[:,:,m - 4,:]) / dxt ** 2
    elif name == 3:
        uxt[:,:,:,1:m-1]= (u[:,:,:,2:m] - 2 * u[:,:,:,1:m-1] + u[:,:,:,0:m-2]) / dxt ** 2
        # uxt[:,:,:,0] = (u[:,:,:,1]+u[:,:,:,-1]-2*u[:,:,:,0])/dxt ** 2
        # uxt[:,:,-1] = (u[:,:,0]+u[:,:,-2]-2*u[:,:,-1])/dxt ** 2  
        uxt[:,:,0] = (2 * u[:,:,0] - 5 * u[:,:,1] + 4 * u[:,:,2] - u[:,:,3]) / dxt ** 2
        uxt[:,:,:,p - 1] = (2 * u[:,:,:,p - 1] - 5 * u[:,:,:,p - 2] + 4 * u[:,:,:,p - 3] - u[:,:,:,p- 4]) / dxt ** 2       
    else:
        NotImplementedError()
# except:
    #     import pdb;pdb.set_trace()

    return uxt

# @jit(nopython=True)
# def Laplace(u,x):
#     x1,x2 = x
#     uxt = Diff2_2(u,x1, name = 1)
#     uxt += Diff2_2(u,x2, name = 2)
#     return uxt
