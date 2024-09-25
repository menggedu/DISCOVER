import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset,DataLoader
import logging
import os 
import math
# print(os.getcwd())

from dso.task.pde.utils_noise import normalize, unnormalize, np2tensor,tensor2np
from dso.task.pde.data_load import load_data, load_data_2D
from dso.task.pde.utils_v1 import Diff, Diff2, Diff3

sym_true_dict= {
    "Burgers": 'add_t,mul_t,u,diff1_t,u,x1,diff2_t,u,x1',
    "chafee-infante": 'add_t,add_t,u,n3_t,u,diff2_t,u,x1',
    "PDE_compound":'add_t,mul_t,u,diff2_t,u,x1,mul_t,diff1_t,u,x1,diff1_t,u,x1',
    "Kdv":'add_t,mul_t,u,diff1_t,u,x1,diff3_t,u,x1'
}
def Softplus(x):
    return torch.log(1.0+torch.exp(x))

activation_func = {
    "Softplus": Softplus,
    "sin":torch.sin,
    "tanh": torch.tanh
    
}
class ANN(nn.Module):
    def __init__(self,
                 number_layer = 1,
                 input_dim = 2,
                 n_hidden = 1024,
                 out_dim=1,
                 activation = 'sin'):
        super(ANN,self).__init__()
        
        self.input_layer = nn.Linear(input_dim, n_hidden)
        
        self.mid_linear= nn.ModuleList()
        for _ in range(number_layer-1):
            self.mid_linear.append(nn.Linear(n_hidden, n_hidden))
            
        self.out_layer = nn.Linear(n_hidden,out_dim)
         
        self.activation = torch.tanh if activation not in activation_func else  activation_func[activation]
        # self.activation_name = activation
        
    def forward(self,x):
        
        out = self.activation((self.input_layer(x)))
      
        for i in range(len(self.mid_linear)):
            out = self.mid_linear[i](out)
            out = self.activation(out)
        out = self.out_layer(out)
        # if self.activation_name == 'Softplus':
        #     out = self.activation(out)
        return out   
        
class PDEDataset(Dataset):
    def __init__(self, data_name,noise_level,normalize_type, train_ratio, data_efficient):
        self.data_name = data_name
        self.noise_level = noise_level
        x_flat, y_flat, self.U_params, self.x_params,self.u_shape, self.U = self.preprocess(normalize_type,data_efficient)
        # import pdb;pdb.set_trace()
        x_train,x_val, y_train,y_val = self.split_data(x_flat,y_flat,train_ratio)
        logging.info(f"train shape: {x_train.shape[0]}")
        logging.info(f"total shape: {x_flat.shape[0]}")
        self.x = torch.from_numpy(x_train).float()
        self.y = torch.from_numpy(y_train).float()
        self.x_val = torch.from_numpy(x_val).float()
        self.y_val = torch.from_numpy(y_val).float()
        self.x_flat = torch.from_numpy(x_flat).float()
        self.y_flat = torch.from_numpy(y_flat).float()
        self.y_flat_true = torch.from_numpy(self.U.reshape(-1,1)).float()
        self.U = torch.from_numpy(self.U).float()
        
        # self.x, self.y, self.normalize_params = self.preprocess(normalize_type)
    
    @property
    def u_true(self):
        return self.U
    
    @property
    def normalize_params(self):
        return self.U_params, self.x_params
    
    @property
    def U_shape(self):
        return self.u_shape
    
    @property
    def train_data(self):
        return self.x, self.y
    
    @property
    def test_data(self):
        return self.x_val, self.y_val
    
    @property
    def data(self):
        return self.x_flat, self.y_flat, self.y_flat_true
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        # print(index)
        return self.x[index], self.y[index]
    
    def preprocess(self, normalize_type, data_efficient):
        
        """_summary_
         process data including normalization (0-1) or normal

        Returns:
            _type_: _description_
        """
        '''
        Loads coordinate vectors and surface data from .npy files at specified location. 
        Coordinate vectors are converted to mesh grids and flattened out. Surface data
        are flattened out and scaled to [0, 1]. Clean data are loaded in and scaled using
        min and max values from noisy surface data.
        
        Input:
            
        Output:
        
            x_flat  : array of ordered pairs of coordinate vectors [N x 2]
            y_flat  : array of flattened noisy surface values [N x 1]
            U_min   : minimum value of noisy surface
            U_max   : maximum value of (noisy surface - U_min)
            U.shape : surface shape  
        '''
        
        # import data file
        load_class = load_data_2D if "2D" in self.data_name else load_data
        U,X,t,ut,sym_true, n_input_var,test_list = load_class(self.data_name)
        # load coordinate vectors (independent variables)
        x_coord_vecs = [*X,t] # [time*space]
        # import pdb;pdb.set_trace()
        if data_efficient:
            U = U[200:300,:,:]
            t =  t[200:300]
        if len(X) == 1:
            U = U.T # time first
            # reshape inputs and outputs
            x_coord_mats = np.meshgrid(*x_coord_vecs)
            x_flat = np.asarray([X.reshape(-1) for X in x_coord_mats]).T # [N x 2]
            
        else:
            xx,yy= np.meshgrid( X[0].reshape(-1,1),X[1].reshape(-1,1))
            xx,tt= np.meshgrid(xx,t)
            yy,tt= np.meshgrid(yy,t)
            x_flat= np.concatenate(( xx.reshape(-1,1), yy.reshape(-1,1), tt.reshape(-1,1)), axis = 1)         
                
        U_true = U.copy()
        U_shape = U.shape
        U = U_true+self.noise_level*np.std(U_true)*np.random.randn(*U_true.shape)

        # import pdb;pdb.set_trace()
        x_flat, x_params = normalize(x_flat, normalize_type) # [N,m]
        y_flat = U.reshape(-1,1) # [N x 1]
        if len(X) == 1:
            y_flat, U_params = normalize(y_flat,normalize_type)
        else:
            U_params = [0,1]
        
        U_true = U_true.reshape(-1,1)
        U_true = (U_true-U_params[0])/U_params[1]
        U_true = U_true.reshape(U_shape)
        
        return x_flat, y_flat, U_params, x_params, U_shape, U_true
    

    
    def split_data(self, X, Y, train_ratio, val_ratio = 0.2):
        
        data_num = Y.shape[0]
        
        # shuffle and split into train/validation samples
        # sample = np.random.choice([True, False], data_num , p=[train_ratio, 1.0-train_ratio])
        # x_train = X[sample]
        # y_train = Y[sample]
        # x_val = X[~sample]
        # y_val = Y[~sample]
        # import pdb;pdb.set_trace()
        totalID = list(range(0,data_num))
        import random 
        random.shuffle(totalID)
        offset = int(data_num * train_ratio)
        val_num = int(data_num*val_ratio)
        x_train = X[totalID[:offset]]
        y_train = Y[totalID[:offset]]
        if train_ratio>0.9:
            x_val = X[totalID[:offset]]
            y_val = Y[totalID[:offset]]
        else:
            x_val = X[totalID[offset:val_num+offset]]
            y_val = Y[totalID[offset:val_num+offset]]

        
        return x_train,x_val, y_train,y_val
        

def torch_diff(u, xt, order = 1, dim=None):
    grad = torch.autograd.grad(outputs=u,inputs = xt,
                                grad_outputs = torch.ones_like(u),
                                 create_graph=True)[0]
    if dim is not None:
        grad = grad[:,dim:dim+1]    
    for _ in range(order-1):
        grad = torch.autograd.grad(outputs=grad,inputs = xt,
                                grad_outputs = torch.ones_like(u),
                                 create_graph=True)[0]
        if dim is not None:
            grad = grad[:,dim:dim+1]
    return grad

def Laplacian_t(u, x):

    diff2 = torch_diff(u, x[0], order = 2)
    # diff2_y = torch_diff(u, x[1], order = 2)
    
    # diff2_z = torch_diff(u, x[2], order = 2)
    # import pdb;pdb.set_trace()
    for i in range(len(x)-1):
        diff2_ = torch_diff(u, x[i+1], order = 2)
        diff2+=diff2_
    return diff2
    

def plot_u_diff( u_diff1,u_diff2,u_diff3, x, t):
    t_shape, x_shape = u_diff1.shape
    # t = np.arange(t_shape)
    # x = np.arange(x_shape)
    cmap = 'seismic'
    xx, tt = np.meshgrid(x,t)
    
    
    u_diff1 = tensor2np(u_diff1).reshape(t_shape,x_shape)
    fig = plt.figure(figsize=(8,6))
    
    
    ax = fig.add_subplot(131)
    c = ax.pcolormesh(xx, tt, u_diff1, cmap = cmap)
    fig.colorbar(c, ax=ax)

    u_diff2 = tensor2np(u_diff2).reshape(t_shape,x_shape)   
    ax = fig.add_subplot(132)
    c = ax.pcolormesh(xx, tt, u_diff2, cmap = cmap)
    fig.colorbar(c, ax=ax)
    u_diff3 = tensor2np(u_diff3).reshape(t_shape,x_shape)
    ax = fig.add_subplot(133)
    c = ax.pcolormesh(xx, tt, u_diff3, cmap = cmap)
    fig.colorbar(c, ax=ax)
     
    plt.show()
    
def plot_field(u_true,u,u_nn, x,t, name , cache):
    t_shape, x_shape= u.shape
    try:
        if torch.is_tensor(x):
            x = tensor2np(x)
            x = np.unique(x)
        if torch.is_tensor(t):
            t =tensor2np(t)
            t= np.unique(t)
    except:
        import pdb;pdb.set_trace()
    path = cache['path']
    noise = cache['noise']
    iter = cache['iter']
    title = cache['exp'] if 'exp' in cache else "pretrain"
    
    cmap = 'seismic'
    xx, tt = np.meshgrid(x,t)
    # import pdb;pdb.set_trace()
    
    fig = plt.figure(figsize=(21,15))
    ax = fig.add_subplot(431)
    c = ax.pcolormesh(xx, tt, u_true, cmap = cmap)
    fig.colorbar(c, ax=ax)
    
    ax = fig.add_subplot(432)
    if cache['generation_type'] == 'AD':
        # c = ax.pcolormesh(xx[50:150,200:500], tt[50:150,200:500], u_nn[50:150,200:500], cmap = cmap)
        c = ax.pcolormesh(xx, tt, u_nn, cmap = cmap)
    else:
        c = ax.pcolormesh(xx, tt, u, cmap = cmap)
    fig.colorbar(c, ax=ax)
    
    ax = fig.add_subplot(433)
    if cache['generation_type'] == 'AD':
        # c = ax.pcolormesh(xx[50:150,200:500], tt[50:150,200:500], (u_true-u_nn)[50:150,200:500], cmap = cmap)
        c = ax.pcolormesh(xx, tt, (u_true-u_nn), cmap = cmap)
    else:
        c = ax.pcolormesh(xx, tt, u_true-u, cmap = cmap)
    fig.colorbar(c, ax=ax)
    
    ax = fig.add_subplot(434)
    ax.plot(t, u_true[:,x_shape//2], label = "true")
    ax.plot(t, u[:,x_shape//2], label = 'fd')
    ax.plot(t, u_nn[:,x_shape//2], label = 'NN')
    ax.set_xlabel("x len")
    ax.legend()
    
    ax = fig.add_subplot(435)
    ax.plot(x, u_true[t_shape//2,:], label = "true")
    # ax.plot(x, u[t_shape//2,:], label = 'fd')
    ax.plot(x, u_nn[t_shape//2,:], label = 'NN')
    ax.set_xlabel("t len")
    ax.legend()
    ax.set_title(title)
    
    ax = fig.add_subplot(436)
    ax.plot(x, u_true[t_shape//3,:], label = "true")
    ax.plot(x, u[t_shape//3,:], label = 'fd')
    ax.set_xlabel("t len")
    ax.legend()
    ax.set_title(title)

    ax = fig.add_subplot(437)
    ax.plot(t, u_true[:,x_shape//3], label = "true")
    ax.plot(t, u[:,x_shape//3], label = 'fd')
    ax.plot(t, u_nn[:,x_shape//3], label = 'NN')
    ax.set_xlabel("x len")
    ax.legend()
    
    ax = fig.add_subplot(438)
    ax.plot(x, u_true[t_shape//3,:], label = "true")
    # ax.plot(x, u[t_shape//2,:], label = 'fd')
    ax.plot(x, u_nn[t_shape//3,:], label = 'NN')
    ax.set_xlabel("t len")
    ax.legend()
    ax.set_title(title)
    
    ax = fig.add_subplot(439)
    ax.plot(x, u_true[t_shape//3,:], label = "true")
    ax.plot(x, u[t_shape//3,:], label = 'fd')
    ax.set_xlabel("t len")
    ax.legend()
    ax.set_title(title)

    ax = fig.add_subplot(4,3,10)
    ax.plot(t, u_true[:,2*x_shape//3], label = "true")
    ax.plot(t, u[:,2*x_shape//3], label = 'fd')
    ax.plot(t, u_nn[:,2*x_shape//3], label = 'NN')
    ax.set_xlabel("x len")
    ax.legend()
    
    ax = fig.add_subplot(4,3,11)
    ax.plot(x, u_true[2*t_shape//3,:], label = "true")
    # ax.plot(x, u[t_shape//2,:], label = 'fd')
    ax.plot(x, u_nn[2*t_shape//3,:], label = 'NN')
    ax.set_xlabel("t len")
    ax.legend()
    ax.set_title(title)
    
    ax = fig.add_subplot(4,3,12)
    ax.plot(x, u_true[2*t_shape//3,:], label = "true")
    ax.plot(x, u[2*t_shape//3,:], label = 'fd')
    ax.set_xlabel("t len")
    ax.legend()
    ax.set_title(title)

    plt.savefig(path+name+f"_{iter}.png")
    if name == 'u':
        np.save(path+name+f"_{iter}.npy",u)
    
def plot_ut(ut_true, ut_noise_fd, ut_diff, x, t):
    t_shape, x_shape = ut_true.shape
    # t = np.arange(t_shape)
    # x = np.arange(x_shape)
    cmap = 'seismic'
    xx, tt = np.meshgrid(x,t)
    # import pdb;pdb.set_trace()
    
    ut_diff = tensor2np(ut_diff).reshape(t_shape,x_shape)
    fig = plt.figure(figsize=(16,9))
    
    
    ax = fig.add_subplot(231)
    c = ax.pcolormesh(xx, tt, ut_true, cmap = cmap)
    fig.colorbar(c, ax=ax)
    
    ax = fig.add_subplot(232)
    c = ax.pcolormesh(xx, tt, ut_diff, cmap = cmap)
    fig.colorbar(c, ax=ax)
    
    ax = fig.add_subplot(233)
    c = ax.pcolormesh(xx, tt, ut_noise_fd, cmap = cmap)
    fig.colorbar(c, ax=ax)
    
    ax = fig.add_subplot(234)
    c = ax.pcolormesh(xx, tt, ut_true-ut_true, cmap = cmap)
    fig.colorbar(c, ax=ax)
    
    ax = fig.add_subplot(235)
    c = ax.pcolormesh(xx, tt,ut_true- ut_diff, cmap = cmap)
    fig.colorbar(c, ax=ax)
    
    ax = fig.add_subplot(236)
    c = ax.pcolormesh(xx, tt, ut_true-ut_noise_fd, cmap = cmap)
    fig.colorbar(c, ax=ax)
    
    plt.show()

def cut_bound_quantile(x, t, quantile=0.1):
    low_x, low_t= np.quantile(x,quantile),np.quantile(t,quantile)
    up_x,up_t =np.quantile(x,1-quantile), np.quantile(t,1-quantile)
    x_limit = np.logical_and(x>low_x,x<up_x)
    t_limit = np.logical_and(t>low_t, t<up_t)
    limit = np.logical_or(x_limit,t_limit).reshape(-1)

    x= x[limit,:]
    t= t[limit,:]
    return x, t
    
    
def load_noise_data(dataset, noise_level, data_ratio,  use_meta_data=False):
    """load data
        return u, x,t,  

    Args:
        dataset (_type_): _description_
        noise_level (_type_): _description_
        data_ratio (_type_): _description_
        device (str, optional): _description_. Defaults to 'cpu'.
        use_meta_data (bool, optional): _description_. Defaults to False.
    """
    #1. load model
    device = torch.device('cuda')
    ckpt_path = f'./out/checkpoint/{dataset}_noise={noise_level}_data_ratio={data_ratio}/best.ckpt'
    model = load_NN(ckpt_path, device)
    
    #2.prepare input data
    data_path = f'./dso/task/pde/noise_data_new/{dataset}_noise={noise_level}_data_ratio={data_ratio}.npz'
    data = np.load(data_path) # trainging_data
    t_len, x_len = data['U_pred'].shape
    xt = data['xt'] if not use_meta_data else data['xt_all']

    U_mean, U_std, x_mean, x_std = data['U_mean'], data['U_std'], data['x_mean'], data['x_std']
    U_mean, U_std, x_mean, x_std = np2tensor(U_mean, device), np2tensor(U_std, device), np2tensor(x_mean,device),np2tensor(x_std,device) 
    
    nn_x, nn_t = xt[:,:-1], xt[:,-1:]
    
    #load_origin_data
    u_true,x_list, t,ut_true,sym_true, n_input_var,_  = load_data(dataset)
    u_noise,_, t,ut_noise,sym_true, n_input_var,_  = load_data(dataset, noise_level=noise_level, data_amount=data_ratio)
    
    # if use_meta_data:
    #     nn_x = nn_x.reshape(t_len,x_len)[int(t_len*0.1):int(t_len*0.9), int(x_len*0.1):int(x_len*0.9)]
    #     nn_t = nn_t.reshape(t_len,x_len)[int(t_len*0.1):int(t_len*0.9), int(x_len*0.1):int(x_len*0.9)]
    #     nn_x = nn_x.reshape(-1,1)
    #     nn_t = nn_t.reshape(-1,1)
        
    # nn_x, nn_t = cut_bound(nn_x,nn_t)
    nn_x_tensor = np2tensor(nn_x, device, requires_grad = True)
    nn_t_tensor = np2tensor(nn_t,device, requires_grad = True)
    nn_xt = torch.cat((nn_x_tensor, nn_t_tensor), axis = 1)
    
        
    #3. normalize input and unnormalize output
    # import pdb;pdb.set_trace()
    nn_xt_normalize =  (nn_xt-x_mean)/ x_std #normalize(nn_xt, [x_mean, x_std])
    u = model(nn_xt_normalize)
    u = unnormalize(u,[U_mean, U_std])
    
    # test_ fd and ad
    ut = torch_diff(u, nn_t_tensor, 1)
    # grad = torch.autograd.grad(outputs=u.sum(),inputs = nn_xt,
    #                              create_graph=True)[0]
    # ux = grad[:,0].reshape(-1,1)
    # uxx = torch.autograd.grad(outputs=ux.sum(),inputs = nn_xt,
    #                              create_graph=True)[0][:,0].reshape(-1,1)
    # uxxx = torch.autograd.grad(outputs=uxx.sum(),inputs = nn_xt,
    #                              create_graph=True)[0][:,0].reshape(-1,1)
    ux = torch_diff(u,nn_x_tensor, order = 1)
    
    uxx = torch_diff(u,nn_x_tensor, order=2)
    uxxx = torch_diff(u,nn_x_tensor, order=3)
    uux = torch_diff(u*ux,nn_x_tensor, order= 1)



    dx = x_list[0]
    ux_true = Diff(u_true, dx, 'x')
    uxx_true = Diff2(u_true, dx, 'x')
    uxxx_true = Diff3(u_true, dx, 'x')
    uux_true =Diff3(ux_true*u_true,dx,'x')
    
    ux_noise = Diff(u_noise, dx, 'x')
    uxx_noise = Diff2(u_noise, dx, 'x')
    uux_noise = Diff(u_noise*ux_noise,dx,'x')
    uxxx_noise = Diff3(u_noise, dx, 'x')
    
    # plot_ut(u_true.T, u, u_noise.T, dx, t)
    plot_ut(ut_true.T, ut, ut_noise.T, dx, t)
    plot_ut(ux_true.T, ux, ux_noise.T, dx, t)
    plot_ut(uxx_true.T, uxx, uxx_noise.T,  dx, t)
    plot_ut(uxxx_true.T, uxxx, uxxx_noise.T,  dx, t)
    # import pdb;pdb.set_trace()
    # if use_meta_data:
    #     ut = ut[math.floor(n*0.1):math.ceil(n*0.9), math.floor(m*0.1):math.ceil(m*0.9)]
    # x first
    
    return u,[nn_x_tensor],nn_t_tensor,ut,sym_true_dict[dataset], 1, None

def LBFGS(model_param,
        lr=1.0,
        max_iter=100000,
        max_eval=None,
        history_size=50,
        tolerance_grad=1e-7,
        tolerance_change=1e-7,
        line_search_fn="strong_wolfe"):

    optimizer = torch.optim.LBFGS(
        model_param,
        lr=lr,
        max_iter=max_iter,
        max_eval=max_eval,
        history_size=history_size,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        line_search_fn=line_search_fn
        )

    return optimizer
    
    
    
    
    
def load_NN(ckpt_path, device):
    '''
    load  a  denoising model
    '''  
    print(f"loading model from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location = 'cpu')
    args = state_dict['args']
    model = ANN( args.num_layer,
                 args.input_dim ,
                 args.hidden_num ,
                 args.out_dim, args.activation)
    model.load_state_dict(state_dict['model'])
    model.to(device)
    return model
    

    
    
if __name__ == '__main__':
    from dso.task.pde.data_load import  load_data
    import math
    _ = load_noise_data('Kdv', 0.1,1.0, use_meta_data=True)
    # w1 = torch.ones(4,4)
    # w1.requires_grad = True
    # w2 = torch.ones(2,1)
    # w2.requires_grad = True
    # x1 = torch.ones((4,1))*math.pi
    # x1.requires_grad =True
    # x0=0.5*math.pi*torch.ones((4,1))
    # x0.requires_grad =True
    # # print(x.is_leaf)
    # y1 = torch.matmul(w1,torch.cat((x1,x0), axis=1))
    # print(y1)
    # y=torch.sin(torch.matmul(y1,w2))
    # print(y)
    
    # # diff1_1 = torch_diff(y,x//,order=1)
    # diff1 = torch_diff(y,x1,order=1)
    # diff2 = torch_diff(y,x1,order=2)
    # # print(ux)
    # # print()
    # print(diff1)
    # print(diff2)
    
    


    