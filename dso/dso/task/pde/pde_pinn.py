import numpy as np
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
from dso.task.pde.weak_form import WeakEvaluate


class PDEPINNTask(PDETask):
    """
    Class for the PINN-based Task in R_DISCOVER (MODE2).

    """

    task_type = "pde_pinn"
    model = None
    def __init__(self, function_set, dataset, metric="residual",
                 metric_params=(0.01,), extra_metric_test=None,
                 extra_metric_test_params=(), reward_noise=0.0,
                 reward_noise_type="r", threshold=1e-12,
                 data_noise_level=0,
                 data_amount = 1,
                 use_meta_data = False,
                 weak_params = None,
                 use_torch = False,
                 sym_true_input =None,
                 max_depth=4,
                 normalize_variance=False, protected=False,
                 spatial_error = True, 
                 decision_tree_threshold_set=None,
                 cut_ratio = 0.03,
                 n_input_var = None,
                 add_const = False,
                 eq_num=1
                 ):


        # super(PDETask).__init__()
        self.name = dataset
        if "MD_NU" in dataset:
            load_class = load_data_MD_NU
            # n_state_var=3
        elif "2D" in dataset:
            load_class = load_data_MD_NU
            # n_state_var=1
        else:
            load_class = load_data
            n_state_var=1
        self.u,self.x,self.t, ut,sym_true, n_input_var,test_list,n_state_var = load_class(dataset, data_noise_level, data_amount, True,cut_ratio =cut_ratio)
        self.u_true,self.x,self.t, ut_true,sym_true, n_input_var,test_list,_ = load_class(dataset, 0, data_amount, True,cut_ratio =cut_ratio)
        self.shape = self.u[0].shape
        self.spatial_error = spatial_error
        self.ut=ut
        self.ut_true=ut_true
        self.add_const = add_const
        self.eq_num = eq_num
        # if not isinstance(ut, list):
        #     self.ut = [ut]
        #     self.ut_true = [ut_true]
            
        self.sym_true = sym_true if sym_true_input is None else sym_true_input
        self.sym_true_input = sym_true_input

        self.ut =self.ut.reshape(-1,1) 
        self.ut_true = self.ut_true.reshape(-1,1)
        self.noise_level = data_noise_level
        self.max_depth = max_depth
        self.cut_ratio = cut_ratio
        if torch.is_tensor(self.ut):
            self.ut = tensor2np(self.ut)


        self.x_ori = self.x
        self.t_ori = self.t
        """
        Configure train/test reward metrics.
        """
        self.threshold = threshold
        self.metric, self.invalid_reward, self.max_reward = make_pde_metric(metric, *metric_params)
        self.extra_metric_test = extra_metric_test
        self.metric_test = None

        self.u_test,self.ut_test = None,None

        """
        Configure reward noise.
        """
        self.reward_noise = reward_noise
        self.reward_noise_type = reward_noise_type
        self.normalize_variance = normalize_variance
        assert reward_noise >= 0.0, "Reward noise must be non-negative."

        self.rng = None
        self.scale = None
            
        # Set the Library 
        tokens = create_tokens(n_input_var=n_input_var, # if n_input_var is not None else 1,
                               function_set=function_set,
                               protected=protected,
                               n_state_var=n_state_var,
                               decision_tree_threshold_set=decision_tree_threshold_set,
                               task_type='pde')
        self.library = Library(tokens)
        
        if use_torch:
            torch_tokens = add_torch_tokens(function_set, protected = protected)
            self.library.add_torch_tokens(torch_tokens)
        # Set stochastic flag
        self.stochastic = reward_noise > 0.0

        self.iter = 0

        # weak formulation discovery
        self.wf = None
        self.wf_flag = False
        if weak_params is not None:
            self.weak_params = weak_params
            self.wf = WeakEvaluate()
            self.wf_flag = True
        
    def reward_function(self,p):
        
        y_hat, y_right, w = p.execute(self.u, self.x, self.ut, wf = self.wf_flag)
        n = len(w)
        if p.invalid:
            # print(p.tokens)
            return self.invalid_reward, [0]

        # Compute metric
        r = self.metric(self.ut, y_hat,n)
        return r, w

    def evaluate(self, p):

        # Compute predictions on test data
        y_hat,y_right, w = p.execute(self.u, self.x, self.ut,wf = self.wf_flag)

        n = len(w)
        
        if p.invalid:
            nmse_test = None
            nmse_test_noiseless = None
            success = False

        else:
            # NMSE on test data (used to report final error)
            nmse_test = np.mean((self.ut - y_hat) ** 2)

            # NMSE on noiseless test data (used to determine recovery)
            # nmse_test_noiseless = np.mean((self.y_test_noiseless - y_hat) ** 2) / self.var_y_test_noiseless

            # Success is defined by NMSE on noiseless test data below a threshold
            success = nmse_test < self.threshold

        info = {
            "nmse_test" : nmse_test,
            "success" : success,

        }
        if self.u_test is not None:
        
            y_hat_test,y_right, w_test = p.execute(self.u_test, self.x, self.ut_test, test=True, wf = self.wf_flag)
            info.update({
                'w_test': w_test
            })

        if self.metric_test is not None:
            if p.invalid:
                m_test = None
                m_test_noiseless = None
            else:
                m_test = self.metric_test(self.y_test, y_hat,n)
                m_test_noiseless = self.metric_test(self.y_test_noiseless, y_hat,n)

            info.update({
                self.extra_metric_test : m_test,
                self.extra_metric_test + '_noiseless' : m_test_noiseless
            })

        return info

    def generate_meta_data(self, model, generation_type = 'AD', plot= False):
        """
        Generate meta data based on NN. 
        FD refers to the finite difference. Only generate data on regulard grids
        AD refers to the automatic differentiation. Use collocation data as trainging data
        Args:
            model (_type_): NN
            generation_type (str, optional): FD or AD. Defaults to 'AD'.
            plot (bool, optional): Plot intermediate results. Defaults to False.
        """
        print("generating metadata")
        u, x, cache = model.generate_meta_data()
        cache['iter'] = self.iter
        cache['generation_type'] = generation_type
        
        if generation_type == "AD" or generation_type == "FD":
            u_net = tensor2np(u).reshape(self.shape[1],self.shape[0])
            u_true = self.u_true[0]
            ut_true = self.ut_true.reshape(self.shape)
            #t first
            
            plot_field(u_true.T, u_net, u_net,self.x_ori[0],self.t_ori, 'u', cache)
            ut_net = np.zeros((self.shape))
            for idx in range(self.shape[0]):
                ut_net[idx, :] = FiniteDiff(u_net.T[idx, :], self.t_ori[1]-self.t_ori[0]) 
            ut_torch = torch_diff(u,x, order=1,dim=1)
            ut_torch = tensor2np(ut_torch).reshape(self.shape[1],self.shape[0])
            plot_field(ut_true.T,ut_net.T,ut_torch,self.x_ori[0], self.t_ori,'ut', cache)
            
            ux_true = Diff(u_true,self.x_ori[0],1)
            ux_torch = torch_diff(u,x, order=1,dim=0)
            ux_torch = tensor2np(ux_torch).reshape(self.shape[1],self.shape[0])
            ux_net = Diff(u_net.T,self.x_ori[0] ,1)
            plot_field(ux_true.T,ux_net.T,ux_torch,self.x_ori[0], self.t_ori,'ux', cache)
            
            uux_true = u_true*Diff(u_true,self.x_ori[0],1)
            uux_torch = u*torch_diff(u,x, order=1,dim=0)
            uux_torch = tensor2np(uux_torch).reshape(self.shape[1],self.shape[0])
            uux_net = u_net.T*Diff(u_net.T,self.x_ori[0] ,1)
            plot_field(uux_true.T,uux_net.T,uux_torch,self.x_ori[0], self.t_ori,'uux', cache)

            uxx_true = Diff2(u_true,self.x_ori[0],1)
            uxx_torch = torch_diff(u,x, order=2,dim=0)
            uxx_torch = tensor2np(uxx_torch).reshape(self.shape[1],self.shape[0])
            uxx_net = Diff2(u_net.T,self.x_ori[0] ,1)
            plot_field(uxx_true.T,uxx_net.T,uxx_torch,self.x_ori[0], self.t_ori,'uxx', cache)
            
            uxxx_true = Diff3(u_true,self.x_ori[0],1)
            uxxx_torch = torch_diff(u,x, order=3,dim=0)
            uxxx_torch = tensor2np(uxxx_torch).reshape(self.shape[1],self.shape[0])
            uxxx_net =Diff3(u_net.T,self.x_ori[0] ,1)
            plot_field(uxxx_true.T,uxxx_net.T,uxxx_torch,self.x_ori[0], self.t_ori,'uxxx', cache)
            
            uxxxx_true = Diff4(u_true,self.x_ori[0],1)
            uxxxx_torch = torch_diff(u,x, order=4,dim=0)
            uxxxx_torch = tensor2np(uxxxx_torch).reshape(self.shape[1],self.shape[0])
            uxxxx_net = Diff4(u_net.T,self.x_ori[0],1 )
            plot_field(uxxxx_true.T,uxxxx_net.T,uxxxx_torch,self.x_ori[0], self.t_ori,'uxxxx', cache)

            # rhs_torch = [cut_bound(uux_torch.reshape(self.shape[1],self.shape[0]).T,0.03).reshape(-1),cut_bound(uxx_torch.reshape(self.shape[1],self.shape[0]).T,0.03).reshape(-1),cut_bound(uxxxx_torch.reshape(self.shape[1],self.shape[0]).T,0.03).reshape(-1)]
            # rhs_true = [cut_bound(uux_true,0.03).reshape(-1),cut_bound(uxx_true,0.03).reshape(-1), cut_bound(uxxxx_true,0.03).reshape(-1)]
            # rhs_true = np.array(rhs_true).T
            # lhs_true = cut_bound(ut_true,0.03).reshape(-1)

            # rhs = np.array(rhs_torch).T
            # lhs = cut_bound(ut_torch.reshape(self.shape[1],self.shape[0]).T,0.03).reshape(-1)
            
            # w_best2 = np.linalg.lstsq(rhs, lhs)[0]
            # w_best = np.linalg.lstsq(rhs_true, lhs_true)[0]
            # print(w_best)
            # print(w_best2)
            self.iter+=1
        else:
            pass
            # t_len,n,m = self.shape
            # # import pdb;pdb.set_trace()
            # _,u_star = model.true_value
            # u_predict = u[:,0].reshape(n,m,t_len)
            # u_star1 = u_star[:,0].reshape(n,m,t_len)
            # u_predict = tensor2np(u_predict)
            # u_predict2 = u[:,1].reshape(n,m,t_len)
            # u_star2 = u_star[:,1].reshape(n,m,t_len)
            # u_predict2 = tensor2np(u_predict2)

            # for i in [10,25,50,75 ,100,125,150,175,200]:

            #     plot_predict(u_star1[:,:,i], u_predict[:,:,i], f'rd_{i}u')
            #     plot_predict(u_star2[:,:,i], u_predict2[:,:,i], f'rd2_{i}v')

        if self.wf is not None:
            self.weak_form_cal(x, model)
            return 

        if generation_type=='AD':
            self.AD_generate_1D(x,model)     
        elif generation_type == 'multi_AD':
            self.AD_generate_mD(model)
        elif generation_type =='FD':  
            self.FD_generate(u)
        elif generation_type =='FD_generate_2d':
            self.FD_generate_2d(u)
        else:
            print(generation_type)
            assert False

        def plot_diff_ad(u_true, u_noise_np,u_noise_nn, x_list,x_nn, dt):
            n,m = u_true.shape
            # true
            ut_true = np.zeros((n,m))
            for idx in range(n):
                ut_true[idx, :] = FiniteDiff(u_true[idx, :], dt)
            ux_true = Diff(u_true,x_list[0],1)
            uxx_true = Diff2(u_true,x_list[0],1)
            
            #ad
            ut_torch = torch_diff(u_noise_nn,x_nn, order=1,dim=1)
            ux_torch = torch_diff(u_noise_nn,x_nn,order=1,dim=0)
            uxx_torch = torch_diff(u_noise_nn,x_nn,order=2,dim=0)
            
            #fd
            ut_fd = np.zeros((n,m))
            for idx in range(n):
                ut_fd[idx, :] = FiniteDiff(u_noise_np[idx, :], dt) 
            ux_fd = Diff(u_noise_np,self.x[0],1)
            uxx_fd = Diff2(u_noise_np,self.x[0],1)
            
            plot_ut(u_true.T, self.u_new.T,u, self.x[0], self.t)  
            plot_ut(ut_true.T, ut_fd.T,ut_torch, self.x[0], self.t)   
            plot_ut(ux_true.T, ux_fd.T,ux_torch, self.x[0], self.t)    
            plot_ut(uxx_true.T, uxx_fd.T,uxx_torch, self.x[0], self.t)  
    
    def weak_form_cal(self, x, model):

        device = x.device
        xw,tw,x1,t1,x2,t2 = self.wf.reconstruct_input(device, **self.weak_params)
        
        #u calculation
        x1,x2 = np2tensor(x1,device, requires_grad=True), np2tensor(x2,device, requires_grad=True)
        t1,t2 = np2tensor(t1,device, requires_grad=True), np2tensor(t2,device, requires_grad=True)
        xt1 =  torch.cat((x1, t1), axis = 1)
        u1 = model.net_u(xt1)
        xt2 =  torch.cat((x2, t2), axis = 1)
        u2 = model.net_u(xt2)
        self.u = [u1,u2]
        self.x = [x1,x2]
        # ut calculation
        xw = np2tensor(xw, device, requires_grad = True)
        tw = np2tensor(tw,device, requires_grad=True)
        xt_w = torch.cat((xw, tw), axis = 1)
        uw = model.net_u(xt_w)
        utw = torch_diff(uw, xt_w, order = 1, dim = 1)
        utw = self.wf.glq_cal(utw)
        
        self.ut = utw.reshape(-1,1)

    def AD_generate_1D(self, x, model):
        device = x.device
        x = tensor2np(x)
        x,t = [x[:,:-1]],x[:,-1:]
        x,t = cut_bound_quantile(x,t,quantile = self.cut_ratio)
        x = np2tensor(x[0],device, requires_grad=True)
        t = np2tensor(t, device, requires_grad=True)
        self.x = [x]
        self.t = t
        xt = torch.cat((x, self.t), axis = 1)
        self.u = [model.net_u(xt)]
        ut = torch_diff(self.u[0], xt, order = 1, dim = 1)
        self.ut = tensor2np(ut)

    def AD_generate_mD(self,model):
        """ use collocation data as trainging data"""
        x_f,t_f = model.collocation_point
        device = model.device
        x_f,t_f = [tensor2np(x_f[i]) for i in range(len(x_f))], tensor2np(t_f)
        x_f,t_f = cut_bound_quantile(x_f, t_f, quantile = self.cut_ratio)
        x = [np2tensor(x_f[i], device, requires_grad=True) for i in range(len(x_f))]
        t = np2tensor(t_f, device, requires_grad=True)
        xt = torch.cat((*x, t), axis = 1)
        self.x = x
        self.t = t
        self.u = [model.net_u(xt)]
        # if self.u.shape[-1]>1:
        #     ut = [torch_diff(self.u[0][:,i:i+1], t, order = 1)  for i in range(self.u.shape[-1])]
        #     self.ut = [tensor2np(ut_) for ut_ in ut] 
        # else:
        ut = [torch_diff(self.u[0][:,i:i+1], t, order = 1) for i in range(self.eq_num)]
        self.ut_cache = [tensor2np(ut[i]) for i in range(len(ut))]

    def reset_ut(self,id ):
        self.ut = self.ut_cache[id] 
        
        if isinstance(self.sym_true_input, list):
            self.sym_true=self.sym_true_input[id]


    def FD_generate(self, u):
        n,m=self.shape
        self.u = [tensor2np(u).reshape(m,n).T ]
        # predicted results is t_first, discover process is x first

        dt = self.t[1]-self.t[0]
        self.ut = np.zeros((n,m))
        for idx in range(n):
            self.ut[idx, :] = FiniteDiff(self.u[0][idx, :], dt)
            
        # cut_bound
        n,m = self.ut.shape
        self.ut = cut_bound(self.ut, percent=self.cut_ratio)
        self.ut = self.ut.reshape(-1)  
        
    def FD_generate_2d(self, u):
        t_len,n,m=self.shape
        steps = t_len
        self.u_new = [tensor2np(u[:,0]).reshape(m,n,t_len).transpose(2,1,0) for i in range(u.shape[1]) ]
        # predicted results is t_first, discover process is x first
        max_time =500
        import numpy.linalg as nl # t,n,m
        W,U,V = self.u_new[0].transpose(1,2,0),self.u_new[1].transpose(1,2,0),self.u_new[2].transpose(1,2,0)
        Wn,Un,Vn = W.reshape(n*m, steps),U.reshape(n*m, steps),V.reshape(n*m, steps)
        uw,sigmaw,vw = nl.svd(W, full_matrices=False); vw = vw.T
        uu,sigmau,vu = nl.svd(U, full_matrices=False); vu = vu.T
        uv,sigmav,vv = nl.svd(V, full_matrices=False); vv = vv.T

        uwn,sigmawn,vwn = nl.svd(Wn, full_matrices=False); vwn = vwn.T
        uun,sigmaun,vun = nl.svd(Un, full_matrices=False); vun = vun.T
        uvn,sigmavn,vvn = nl.svd(Vn, full_matrices=False); vvn = vvn.T

        dim_w,dim_u,dim_v= 20,20,20
        # dim_w = 26
        # dim_u = 20
        # dim_v = 20

        Wn = uwn[:,0:dim_w].dot(np.diag(sigmawn[0:dim_w]).dot(vwn[:,0:dim_w].T)).reshape(n,m,steps).transpose(2,0,1)
        Un = uun[:,0:dim_u].dot(np.diag(sigmaun[0:dim_u]).dot(vun[:,0:dim_u].T)).reshape(n,m,steps).transpose(2,0,1)
        Vn = uvn[:,0:dim_v].dot(np.diag(sigmavn[0:dim_v]).dot(vvn[:,0:dim_v].T)).reshape(n,m,steps).transpose(2,0,1)
        self.u_new = [Wn, Un, Vn]

        dt = self.t[1]-self.t[0]
        self.ut_new = np.zeros((t_len,n,m))
        # import pdb;pdb.set_trace()
        for idx in range(t_len-1):
            self.ut_new[idx, :,:] = (self.u_new[0][idx+1,:,:] - self.u_new[0][idx,:,:])/(dt)#FiniteDiff(self.u[0][idx, :], dt)

        # n,m = self.ut.shape
        self.ut_new = cut_bound(self.ut_new, percent=self.cut_ratio)
        # import pdb;pdb.set_trace()
        self.ut = self.ut.reshape(self.ut_new.shape)
        self.name+='sm'
        plot_predict(self.u[0][10], self.u_new[0][10], name = self.name+'_w')
        plot_predict(self.ut[10], self.ut_new[10], name = self.name+'_wt')
        plot_predict(self.u[0][50], self.u_new[0][50], name =self.name+'_w2')
        plot_predict(self.ut[50], self.ut_new[50], name = self.name+'_wt2')

        plot_predict(self.u[1][10], self.u_new[1][10], name = self.name+'_u')
        # plot_predict(self.ut[10], self.ut_new[10], name = self.name+'_ut')
        plot_predict(self.u[1][50], self.u_new[1][50], name =self.name+'_u2')
        # plot_predict(self.ut[50], self.ut_new[50], name = self.name+'_ut2')
        self.ut = self.ut_new.reshape(-1)
        self.u = self.u_new
             
    def stability_test(self,p):
        """
        Evaluate the given program by balancing stability and accuracy.

        Args:
            p (_type_): Program instance

        Returns:
            _type_: two metrics include mse(accuracy) and cv(coefficients of variation)
        """
        self.mse, self.cv = self.cal_mse_cv(p)
        mse = np.array(self.mse)
        cv = np.array(self.cv)
        return mse, cv
        
    def cal_mse_cv(self,p, repeat_num = 100):

        try:
            y_hat,y_right, w = p.execute(self.u, self.x, self.ut, wf = self.wf_flag)
            assert self.ut.shape[0] == y_right.shape[0]
        except Exception as e:
            print(e)
            import pdb;pdb.set_trace()
            print(self.ut.shape, y_right.shape)
            
        N = self.ut.shape[0]
        cv = []
        mse = []
        
        def calculate_cv(ut, y_rhs):
            N_sub = N//2
            cv = []
            coefs = []
            for j in range(10):
                index_sub = np.random.choice(N//2, N_sub, replace = True)
                ut_cv = ut[index_sub]
                y_rhs_cv = y_rhs[index_sub]
                coef = np.linalg.lstsq(y_rhs_cv,ut_cv )[0]
                coefs.append(coef)
                
            coefs = np.array(coefs)
            for i in range(coefs.shape[1]):
                cv.append(np.abs(np.std(coefs[:,i])/np.mean(coefs[:,i])))
            return np.array(cv).mean()
              
        for i in range(repeat_num):
            index = np.random.choice(N, N//2, replace=True)
            ut_sub = self.ut[index]
            y_rhs_sub = y_right[index]
            
            w_sub = np.linalg.lstsq(y_rhs_sub, ut_sub)[0]
            y_hat_sub = y_rhs_sub.dot(w_sub)
            mse_sub = np.mean((ut_sub - y_hat_sub) ** 2)
            cv_sub = calculate_cv(ut_sub, y_rhs_sub)
            cv.append(cv_sub)
            mse.append(mse_sub)
            
        return mse, cv
    
    def switch_lhs(self, num):
        pass

def plot_predict(u1,u2, name):
    # from mpt_toolkits.axes_grid1 import make_axes_locatable
    fig,ax = plt.subplots(1,3, figsize=(40,10))
    im = ax[0].imshow(u1)
    plt.colorbar(im, ax = ax[0])
    im = ax[1].imshow(u2)
    plt.colorbar(im, ax = ax[1])
    im = ax[2].imshow(u1-u2)
    plt.colorbar(im, ax = ax[2])
    plt.savefig(f'./plot/pic/predict_{name}.png')
    
class PDEMultiPINNTask(PDEPINNTask):
    """
    Class for the symbolic regression task. Discrete objects are expressions,
    which are evaluated based on their fitness to a specified dataset.
    """

    task_type = "multi_pde_pinn"
    model = None
    
    
    def  AD_generate(self, x, model):
        super().AD_generate(x, model) 
        
             
        
        
        

if __name__ == '__main__':
    pass