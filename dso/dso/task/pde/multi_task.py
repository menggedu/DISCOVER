import numpy as np
import pandas as pd
import scipy.io as scio
import math
import xarray as xr
import itertools

from dso.task.pde.pde import PDETask, make_pde_metric
from dso.task.pde.utils_subgrid import Subgrid_forcing
from dso.library import Library
from dso.functions import subgrid_tokens_valid
from dso.task.pde.data_load import load_subgrid_test


'''
possible multi-task 
'''

class Multi_Dataset_Task(PDETask):

    task_type = "multi-task"
    def __init__(self,data_info,**kwargs):
        super().__init__(data_info,**kwargs)
        self.data_info = data_info
        

class PDESubgridTask(PDETask):
    """
    Class for subgrid forcing 
    """

    task_type = "pde_subgrid"
    def __init__(self,sample_ratio = 0.85,
                 validation_type='held-out',
                 **kwargs):
        super().__init__(**kwargs)

        
        self.sample_ratio = sample_ratio
        self.validation_type = validation_type
        print("validation:",validation_type)
        print(f"sample ratio: {sample_ratio}")
        
        # set middle results
        self.vals0, self.vals1 = [],[]
        self.vals0_test,self.vals1_test = [],[]
        self.terms = []
        
        t_step, lev_step, x_step, y_step = self.u[0].shape
        self.ut = self.ut.reshape(t_step, lev_step, x_step, y_step)
        
        
        self.ut = [self.ut[:,0,:,:, ],self.ut[:,1,:,:, ]]
        self.ut_ori = self.ut.copy()
        
        # operator for validation
        function_set=['ddx', 'ddy','laplacian','adv']
        subgrid_tokens = subgrid_tokens_valid(function_set)
        self.library.add_subgrid_tokens(subgrid_tokens)

        self.best_r = 0
        self.best_r_val = 0
        #validation data
        self.set_test_data()
        
    def set_test_data(self):
        self.u_test, self.ut_test = load_subgrid_test()
        self.ut_test = [self.ut_test[:,0,:,:, ],self.ut_test[:,1,:,:, ]]
    
    def reward_function(self,p):
        y_hat, y_rhs, w = p.execute(self.u, self.x, self.ut)
        # import pdb;pdb.set_trace()
        if p.invalid:
            return self.invalid_reward, [0]
        
        r = []
        for i in range(len(y_hat)):
            r_ = self.metric(self.ut[i], y_hat[i],len(w[i]))
            r.append(r_)
        # r1 = self.metric(self.ut1, y_hat_0,n1)
        shift= len(w[0])
        if p.cached_terms is not None:
            shift = len(p.cached_terms) * (-1)
        return np.mean(np.array(r)), w[0][:shift]
    
    def evaluate(self, p):
    
        # Compute predictions on test data
        y_hat,y_rhs,  w = p.execute(self.u, self.x, self.ut)

        n = len(w)
        # y_hat = p.execute(self.X_test)
        if p.invalid:
            nmse_test = None
            nmse_test_noiseless = None
            success = False

        else:
            # NMSE on test data (used to report final error)
            nmse_test_list = []
            for i in range(len(y_hat)):
                nmse_test_list.append( np.mean((self.ut[i].reshape(-1,1) - y_hat[i]) ** 2))
                
            nmse_test = np.mean(np.array(nmse_test_list))

            # NMSE on noiseless test data (used to determine recovery)
            # nmse_test_noiseless = np.mean((self.y_test_noiseless - y_hat) ** 2) / self.var_y_test_noiseless

            # Success is defined by NMSE on noiseless test data below a threshold
            success = nmse_test < self.threshold

        info = {
            "nmse_test" : nmse_test,
            "success" : success,

        }


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
    
    def sample_cached_terms(self,  ):
        # sample_indices = [
        if len(self.terms)>0:
            vals0 = np.array(self.vals0)  # shape = (n, txy)
            vals1 = np.array(self.vals1)
            
            indices = np.random.random(len(self.terms))
            saved_indices = np.argwhere(indices<self.sample_ratio).reshape(-1)
            vals0_used, vals1_used = vals0[saved_indices].T, vals1[saved_indices].T
            terms_used = [self.terms[t] for t in saved_indices.tolist() ]
            return terms_used,(vals0_used, vals1_used)
        else:
            return None,None
        
    def process_results(self,p):
        # cur_r = p.r_ridge
        remove_terms = []
        improvement = True
        new_terms = p.STRidge.terms
        # new_vals = p.STRidge.results #list [(txy,)...] 
        _,new_vals, w = p.execute(self.u, self.x, self.ut) #list [(txy,)...]

        start_id = len(self.vals0)
        if self.validation_type == 'held-out':
            p.switch_tokens(token_type = 'subgrid')

            new_vals_test_  = p.execute_test(self.u_test, [], self.ut_test)
                        
            t_shape,lev_shape, x_shape, y_shape = self.u_test[0].shape
            new_vals_test = [[res[:,i,:,:].reshape(-1) for res in new_vals_test_ ] for i in range(lev_shape)]

        assert len(new_vals[0]) == len(new_vals[1])
        term_num = len(self.terms)
        #save terms
        for i, (new_val0, new_val1) in enumerate(zip(new_vals[0], new_vals[1])):
            if not any(self.metric(new_val0, v, None) > 0.99 for v in self.vals0) and \
               not any(self.metric(new_val1, v, None) > 0.99 for v in self.vals1) and \
               np.abs(w[0][i])>1 and \
               np.abs(w[1][i])>1 :   
               # small coef
                self.terms.append(new_terms[i])
                self.vals0.append(new_val0)
                self.vals1.append(new_val1)
                if self.validation_type == 'held-out':
                    self.vals0_test.append(new_vals_test[0][i])
                    self.vals1_test.append(new_vals_test[1][i])
                print(f"new_term {new_terms[i]}")
        # import pdb;pdb.set_trace()
        (r0,w0),(r1,w1),cur_r = self.calculate_r(self.vals0, self.vals1, self.ut_ori)
        if self.best_r>cur_r:
            self.best_r = cur_r
        print(f"before removing redundants, the reward is ({r0}) ({r1}) ({cur_r})") 
        (r0_test,w0_test),(r1_test,w1_test),cur_r_test = self.calculate_r(self.vals0_test, self.vals1_test, self.ut_test)
        print(f"before removing redundants, the reward of validation is ({r0_test}) ({r1_test}) ({cur_r_test})")

        if len(self.vals0)>1:    
            cur_r, remove_term = self.remove_redundants(self.vals0, self.vals1,\
                                                            self.ut_ori, cur_r, start_id)
              
            if self.validation_type == 'held-out':
                _, _,cur_r_test = self.calculate_r(self.vals0_test, self.vals1_test, self.ut_test)
                print(f"before removing redundants, the reward of validation is {cur_r_test}")
                remove_term = -1
                while remove_term is not None:
                    cur_r_test, remove_term = self.remove_redundants(self.vals0_test, self.vals1_test,\
                                                                self.ut_test, cur_r_test, start_id)
                    
                    if remove_term is not None:
                        print(f"removing redundant {self.terms[remove_term]}, the reward of validation is {cur_r_test}")
                        del self.vals0[remove_term]
                        del self.vals1[remove_term]
                        del self.vals0_test[remove_term]
                        del self.vals1_test[remove_term]
                        del self.terms[remove_term]
               
            (r0,w0),(r1,w1),cur_r = self.calculate_r(self.vals0, self.vals1, self.ut_ori)
        
        if self.best_r>cur_r:
            self.best_r = cur_r

        print("current_terms: ")
        print([repr(t) for t in self.terms])
        print(f"layer1 corr:{r0}; layer2 corr:{r1}; correlation is {cur_r}")
        # self.best_r = cur_r  
        if (self.sample_ratio)==0:      
            diff = self.evaluate_diff(w0,w1)
            self.set_ut(diff)
        return improvement
            
    def remove_redundants(self, vals0_list, vals1_list,ut, best_r, start_id):
        """ remove redudants by eliminating terms iteratively"""
        n = len(vals0_list)
        remove_term = None
        
        gap_list = []
        r_list =  []
        for i in range(start_id, n):
            vals0 = [vals0_list[j] for j in range(n) if  j!=i ]
            vals1 = [vals1_list[j] for j in range(n) if  j!=i ]
            
            r0,w0 = self.corr_eval(vals0, ut[0])
            r1,w1 = self.corr_eval(vals1, ut[1])
            r_cur = (r0+r1)/2
            print(f"remove {i}, reward now is upper: {r0} lower:{r1} and total: {r_cur}")
            gap_list.append(best_r-r_cur)
            r_list.append(r_cur)
        
        min_index = np.argsort(gap_list)[0]
        
        if gap_list[min_index]<0.001:
            remove_term = min_index
            best_r = r_list[min_index]
 
        return best_r,remove_term


    def evaluate_diff(self, w0, w1):
        # y_hat,_,_= p.execute(self.u, self.x, self.ut)
        y_hat0 = np.matmul(np.array(self.vals0).T, w0)
        y_hat1 = np.matmul(np.array(self.vals0).T, w1)
        shape = self.ut[0].shape
        return [self.ut[0]-y_hat0.reshape(shape), self.ut[1]-y_hat1.reshape(shape)]
        
    def set_ut(self, ut_diff):
        self.ut = ut_diff

    def corr_eval(self, vals, q_sub_grid):
        result = np.array(vals).T #d x n
        w = np.linalg.lstsq(result, q_sub_grid.reshape(-1,1))[0]
        y_hat = result.dot(w) 
        corr = self.metric(q_sub_grid.reshape(-1,1), y_hat, len(w))
        return corr,w
    
    def calculate_r(self, vals0, vals1, ut_list):
        r0,w0 = self.corr_eval(vals0, ut_list[0])
        r1,w1 = self.corr_eval(vals1, ut_list[1])
        r = (r0+r1)/2
        return (r0,w0), (r1,w1), r
    
    def sum_results(self):
        (r0,r1), (w0,w1), r = self.calculate_r(self.vals0, self.vals1, self.ut_ori)
        print(f"correlation is for lev 0 : {r0}")
        print(f"correlation is for lev 1 : {r1}")
        return self.terms, [self.vals0, self.vals1], (w0,w1)
    
        #reduce_ small coef
        # retain_indices = []
        # for i in range(len(w_best)):
        #     if np.abs(w_best[i])<1e-3: #or np.abs(w_best[i])>1e4:
        #         continue
        #     else:
        #         retain_indices.append(i)
        # results = results[:,retain_indices]
        # w_best_new = np.linalg.lstsq(results, self.ut)[0]    
           

