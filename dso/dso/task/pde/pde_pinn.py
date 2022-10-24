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
from dso.task.pde.utils_v1 import FiniteDiff, Diff, Diff2 

class PDEPINNTask(PDETask):
    """
    Class for the symbolic regression task. Discrete objects are expressions,
    which are evaluated based on their fitness to a specified dataset.
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
                 use_torch = False,
                 normalize_variance=False, protected=False,
                 spatial_error = True, 
                 decision_tree_threshold_set=None,
                 n_input_var = None):


        # super(PDETask).__init__()
        self.name = dataset
        self.u,self.x,self.t, ut,sym_true, n_input_var,test_list = load_data(dataset, data_noise_level, data_amount, True)
        self.u_true,self.x,self.t, ut_true,sym_true, n_input_var,test_list = load_data(dataset, 0, data_amount, True)
        self.shape = self.u.shape
        self.spatial_error = spatial_error
        self.ut=ut
        self.sym_true = sym_true
        self.ut = self.ut.reshape(-1,1)
        self.ut_true = ut_true.reshape(-1,1)
        self.noise_level = data_noise_level
        if torch.is_tensor(self.ut):
            self.ut = tensor2np(self.ut)


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
        # import pdb;pdb.set_trace()
        tokens = create_tokens(n_input_var=n_input_var, # if n_input_var is not None else 1,
                               function_set=function_set,
                               protected=protected,
                               decision_tree_threshold_set=decision_tree_threshold_set,
                               task_type='pde')
        self.library = Library(tokens)
        torch_tokens = add_torch_tokens(function_set, protected = protected)
        self.library.add_torch_tokens(torch_tokens)
        # Set stochastic flag
        self.stochastic = reward_noise > 0.0

        self.iter = 0

        
    def reward_function(self,p):
        
        y_hat, y_right, w = p.execute_STR(self.u, self.x, self.ut)
        n = len(w)
        if p.invalid:
            # print(p.tokens)
            return self.invalid_reward, [0]

        # Compute metric
        r = self.metric(self.ut, y_hat,n)

        return r, w

    def evaluate(self, p):

        # Compute predictions on test data
        y_hat,y_right, w = p.execute_STR(self.u, self.x, self.ut)

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
        
            y_hat_test,y_right, w_test = p.execute_STR(self.u_test, self.x, self.ut_test, test=True)
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

    def generate_meta_data(self, model, plot= True):
        print("generating metadata")
        u, x, cache = model.generate_meta_data()
        cache['iter'] = self.iter
        
        if plot:
            u_net = tensor2np(u).reshape(self.shape[1],self.shape[0])
            u_true = self.u_true
            ut_true = self.ut_true.reshape(self.shape)
            #t first
            
            plot_field(u_true.T, u_net,u_net,self.x[0],self.t, 'u', cache)
            ut_net = np.zeros((self.shape))
            for idx in range(self.shape[0]):
                ut_net[idx, :] = FiniteDiff(u_net.T[idx, :], self.t[1]-self.t[0]) 
            ut_torch = torch_diff(u,x, order=1,dim=1)
            ut_torch = tensor2np(ut_torch).reshape(self.shape[1],self.shape[0])
            plot_field(ut_true.T,ut_net.T,ut_torch,self.x[0], self.t,'ut', cache)
            self.iter+=1
            
        n,m=self.shape
        self.u = tensor2np(u).reshape(m,n).T   
        # predicted results is t_first, discover process is x first

        dt = self.t[1]-self.t[0]
        self.ut = np.zeros((n,m))
        for idx in range(n):
            self.ut[idx, :] = FiniteDiff(self.u[idx, :], dt)
            
        # cut_bound
        n,m = self.ut.shape
        self.ut = self.ut[5:-5,5:-5]
        self.ut = self.ut.reshape(-1)


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
            
    def stability_test(self,p):
        self.mse, self.cv = self.cal_mse_cv(p)
        mse = np.array(self.mse)
        cv = np.array(self.cv)
        return mse, cv
        
    def cal_mse_cv(self,p, repeat_num = 100):

        try:
            y_hat,y_right, w = p.execute_STR(self.u, self.x, self.ut)
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
            # import pdb;pdb.set_trace()
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

            
            
            
        
        
             
        
        
        

if __name__ == '__main__':
    pass