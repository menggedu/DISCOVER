

import sys
import os
# print(os.path.abspath('.'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import numpy as np
import matplotlib, os, math, sys
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import convolve2d
import pdb
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import logging
from pyDOE import lhs
from torch.utils.tensorboard import SummaryWriter

from dso.task.pde.utils_nn import ANN, PDEDataset,np2tensor, tensor2np
from dso.task.pde.utils_noise import normalize, unnormalize,load_PI_data
from dso.task.pde.dataset import Dataset,load_1d_data,load_2d2U_data
from dso.utils import  print_model_summary, eval_result, tensor2np, l2_error

from dso.loss import mse_loss, pinn_loss

logger=logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",level=logging.INFO)

class Ceof(nn.Module):
    def __init__(self, coef_list, n_out = 1):
        super().__init__()
        self.coeff_vector = nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(coef_list[i]).reshape(-1,1)) for i in range(n_out)])
    
    def forward(self, term_list, group = 0):
        #for multi variable max(group) == n_out
        terms = torch.cat(term_list,axis = 1)
        coef = self.coeff_vector[group]
        # terms dim = col_num * term_num
        rhs = torch.mm(terms,coef)
            
        return rhs
        

class PINN_model:
    def __init__(self,
               output_file,
               config_pinn,
               dataset_name,
               device):
        # out_path
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        prefix, _ = os.path.splitext(output_file)
        # NN for pinn
        self.nn = ANN(
                 number_layer = config_pinn['number_layer'],
                 input_dim = config_pinn['input_dim'],
                 n_hidden = config_pinn[ 'n_hidden'],
                 out_dim=config_pinn['out_dim'],
                 activation = config_pinn['activation'])

        #residual loss coef
        self.coef_pde = config_pinn['coef_pde']
        logging.info(self.nn)
        total_params = print_model_summary(self.nn)
        logging.info(f"Total params are {total_params}")
        self.noise = config_pinn['noise']
        self.data_ratio = config_pinn['data_ratio']
        self.coll_data = config_pinn['coll_data']
        self.data_type = config_pinn['data_type']
        
        self.device = device
        self.nn.to(self.device)
        self.dataset_name = dataset_name
        self.extra_gradient= config_pinn.get('extra_gradient', False)
        
        # tensorboard
        tensorboard_file = f"{prefix}_logs"
        self.writer = SummaryWriter(tensorboard_file)
        self.pretrain_path= f"{prefix}_pretrain.ckpt"
        self.pinn_path = f"{prefix}_pinn"
        self.pic_path = f"{prefix}"
            
        # data convert
        
        self.load_data()
        
        # optimizer
        # if config_pinn['optimizer_pretrain'] =='LBFGS':
        #     pass
        # else:
        self.optimizer_pretrain =optim.Adam(self.nn.parameters(), lr=config_pinn['lr'])
        self.optimizer = optim.Adam(self.nn.parameters(), lr=1e-3)
        # self.blgs_optimizer = torch.optim.LBFGS(self.nn.parameters(), lr=0.1, 
        #                       max_iter = 80000, 
        #                       max_eval = None, 
        #                       tolerance_grad = 1e-05, 
        #                       tolerance_change = 1e-09, 
        #                       history_size = 100, 
        #                       line_search_fn = 'strong_wolfe')
        # 
        self.pretrain_epoch =200000
        self.pinn_epoch = config_pinn['pinn_epoch']
        self.pinn_cv_epoch = 2000
        self.cur_epoch = 0
        self.duration = config_pinn['duration']
        self.cache = {}
        self.cache['path'] = prefix
        self.cache['noise'] = self.noise
        self.pretrain_path_load = config_pinn.get('pretrain_path', None)
        
        
    def load_data(self):
        # load label data and collocation points
        if self.data_type == "1D_1U":
            load_class = load_1d_data
        elif self.data_type == '2D_2U':    
            load_class = load_2d2U_data
        
        X_u_train, u_train, X_f_train, X_u_val, u_val, [lb, ub], [x_star, u_star],self.shape = load_class(self.dataset_name,
                                                                                                 self.noise,
                                                                                                 self.data_ratio,
                                                                                                 self.pic_path,
                                                                                                 self.coll_data 
                                                                                                 )
        
        self.x = [X_u_train[:,i:i+1] for i in range(X_u_train.shape[-1]-1)]
        self.t = X_u_train[:,-1:]
        self.x_f = [X_f_train[:,i:i+1] for i in range(X_f_train.shape[-1]-1)]
        self.t_f = X_f_train[:,-1:]
        self.x_val = [X_u_val[:,i:i+1] for i in range(X_u_val.shape[-1]-1)]
        self.t_val = X_u_val[:,-1:]
        
        # self.x_all = np.vstack((self.x, self.x_val))
        # self.t_all = np.vstack((self.t, self.t_val))

        self.x = [np2tensor(self.x[i], self.device, requires_grad=True) for i in range(len(self.x))]
        self.t = np2tensor(self.t, self.device, requires_grad=True)
        self.x_f = [np2tensor(self.x_f[i], self.device, requires_grad=True) for i in range(len(self.x_f))]
        self.t_f = np2tensor(self.t_f, self.device, requires_grad=True)
        self.x_val = [np2tensor(self.x_val[i], self.device) for i in range(len(self.x_val))]
        self.t_val = np2tensor(self.t_val, self.device)
        self.u_train = np2tensor(u_train, self.device)
        

        self.u_val = np2tensor(u_val,self.device, requires_grad=False)
        # convert to tensor
        
        # full-field
        self.x_star = np2tensor(x_star, self.device, requires_grad = True)
        self.u_star =  u_star
       
        self.lb = np2tensor(lb, self.device)
        self.ub = np2tensor(ub, self.device )
        
    def local_sample(self, x,t, lb, ub,sample=False):
        # import pdb;pdb.set_trace()
        x,t,lb,ub=[tensor2np(x[i]) for i in range(len(x))],tensor2np(t),tensor2np(lb),tensor2np(ub)
        if sample:
            
            # import pdb;pdb.set_trace()
            delta_xt = (ub-lb)/100
            xt = np.concatenate((x,t), axis = 1)
            xt = np.repeat(xt,20,axis =0)
            xt_l = xt-delta_xt
            length = len(xt_l)
            print("length:", length)
            xt_new = xt_l+delta_xt*lhs(2,length)
            xt = np.vstack((xt_new, xt))
        else:
            xt = np.concatenate((*x,t), axis = 1)
            
        self.x_local = [np2tensor(xt[:,:-1], self.device, requires_grad=True)]
        self.t_local = np2tensor(xt[:,-1:], self.device, requires_grad=True)
        
    def closure(self,):
        self.blgs_optimizer.zero_grad()
        u_predict = self.net_u(torch.cat([*self.x, self.t], axis= 1))
        loss = mse_loss(u_predict, self.u_train).sum()
        loss.backward()
        
        u_val_predict = self.net_u(torch.cat([*self.x_val, self.t_val], axis = 1))
        loss_val = mse_loss(u_val_predict, self.u_val).sum()
        
        self.cur_epoch+=1
        
                        
        if (self.cur_epoch+1) % 500 == 0:      
            logging.info(f'epoch: {self.cur_epoch+1}, loss_u: {loss.item()} , loss_val:{loss_val.item()}' )
          
        self.writer.add_scalars("pretrain_bfgs/mse_loss",{
                "loss_train": loss.item(),
                "loss_val": loss_val.item()
            }, self.cur_epoch)
        return loss
    
    def pretrain(self):
        
        best_loss = 1e8
        last_improvement=0
        if self.pretrain_path_load is not None:
            self.load_model(self.pretrain_path_load)
        else:
            for i in range(self.pretrain_epoch):
                # train 
                self.optimizer_pretrain.zero_grad()
            
                u_predict = self.net_u(torch.cat([*self.x, self.t], axis= 1))
                loss = mse_loss(u_predict, self.u_train).sum()
                
                u_val_predict = self.net_u(torch.cat([*self.x_val, self.t_val], axis = 1))
                loss_val = mse_loss(u_val_predict, self.u_val).sum()
                
                # record 
                self.writer.add_scalars("pretrain/mse_loss",{
                    "loss_train": loss.item(),
                    "loss_val": loss_val.item()
                }, i)
                
                if loss_val.item()<best_loss:
                    best_loss = loss_val.item()
                    last_improvement=i
                    torch.save(self.nn.state_dict(), self.pretrain_path) 
                    
                if (i+1) % 500 == 0:      
                    logging.info(f'epoch: {i+1}, loss_u: {loss.item()} , loss_val:{loss_val.item()}' )
                            
                loss.backward()
                self.optimizer_pretrain.step()
                
                if i-last_improvement>self.duration:
                    logging.info(f"stop training at epoch: {i+1}, loss_u: {loss.item()} , loss_val:{loss_val.item()}")
                    break
        u_pred, l2 = self.evaluate()
        logging.info(f"full Field Error u: {l2}") 
        self.cur_epoch = self.pretrain_epoch
        
        #bfgs pretrain
        # self.blgs_optimizer.step(self.closure)
        # u_pred, l2 = self.evaluate()
        # logging.info(f"after bfgs full Field Error u: {l2}") 
        
        #concat validation and training point
        self.x = [torch.cat([self.x[i], self.x_val[i]], axis = 0) for i in range(len(self.x))]
        self.t = torch.cat([self.t, self.t_val], axis =0)
        self.u_train = torch.cat([self.u_train,self.u_val], axis = 0)

    def train_pinn_cv(self,program_pde, coef = 1):
        """
        trainable coefficients for postprocessin
        """
        expression =  program_pde.str_expression
        self.cache['exp'] = expression
        self.cache['coefs'] = []
        logging.info(f"start training constrained pinn with traversal {expression}")
        func_coef = program_pde.w
        
        self.coef = Ceof([func_coef],n_out = 1).to(self.device) 
        
        # reset optimizer
        self.set_optimizer()
        
        for epoch in range(self.pinn_cv_epoch):
            self.optimizer.zero_grad()
            u_predict = self.net_u(torch.cat([*self.x, self.t], axis= 1))
            loss_mse = mse_loss(u_predict, self.u_train).sum()
            loss_res1 = pinn_loss(self,  program_pde, self.x_f, self.t_f, self.coef, self.extra_gradient)
            # loss_res2= pinn_loss(self,  program_pde, self.x, self.t, program_pde.w)
            if torch.isnan(loss_res):
                logging.info("nan")
                loss_res=torch.tensor([0]).to(loss_mse)
            loss = coef*(loss_res1)+loss_mse
            loss.backward()
            self.optimizer.step()
            
            if (epoch+1)%10==0:
                logging.info(f"epoch: {epoch+1}, mse: {loss_mse.item()}, res_loss1:{loss_res.item()}, total_loss:{loss.item()}")
                logging.info(f" coefficients : {coefficients_str} ")
                
            self.writer.add_scalars("pinn/pinn_loss",{
                "loss_mse": loss_mse.item(),
                "loss_res": loss_res.item(),
                "loss":loss.item(),
                
            }, self.cur_epoch+epoch)
            
            cur_coefs = self.coef.coeff_vector[0].cpu().detach().numpy().tolist()
            coefficients_str = " | ".join([str(coe) for coe in cur_coefs])
            self.cache['coefs'].append(cur_coefs)
            
        
        u_pred, l2 = self.evaluate()
        logging.info(f"full Field Error u: {l2}")   
        
        torch.save(self.nn.state_dict(), self.pinn_path+'pinn_cv.ckpt')     
        self.cur_epoch+=self.pinn_cv_epoch 
        self.plot_coef()
         
    def train_pinn(self, program_pde, count = 1, coef=0.1, local_sample=True, last=False):
        expression =  program_pde.str_expression
        self.cache['exp'] = expression
        logging.info(f"\n************The {count}th itertion for pinn training.**************** ")
        logging.info(f"start training pinn with traversal {expression}")
        program_pde.switch_tokens()
        if last:
            self.pinn_epoch *=3
            coef = coef
            best_loss = 1e8
        logging.info(f"coef:{coef}")
        if local_sample:
            self.local_sample(self.x, self.t, self.lb, self.ub)
        
        for epoch in range(self.pinn_epoch):
            self.optimizer.zero_grad()
            u_predict = self.net_u(torch.cat([*self.x, self.t], axis= 1))
            loss_mse = mse_loss(u_predict, self.u_train).sum()
            # loss_res1 = self.pinn_loss_explicitly()
            loss_res1 = pinn_loss(self,  program_pde, self.x_f, self.t_f, program_pde.w, self.extra_gradient)
            
            #local samling based on observations    
            if local_sample:
                loss_res2= pinn_loss(self, program_pde, self.x_local, self.t_local, program_pde.w, extra_gradient=self.extra_gradient)
            else:
                loss_res2=torch.tensor([0]).to(loss_mse)
                
            if torch.isnan(loss_res1) or torch.isnan(loss_res2):
                logging.info("nan")
                loss_res1=torch.tensor([0]).to(loss_mse)
                loss_res2=torch.tensor([0]).to(loss_mse)
                
            loss = coef*(loss_res1+loss_res2)+loss_mse
            loss.backward()
            self.optimizer.step()
            
            if (epoch+1)%10==0:
                logging.info(f"epoch: {epoch+1}, mse: {loss_mse.item()}, res_loss1:{loss_res1.item()},res_loss2:{loss_res2.item()}, total_loss:{loss.item()}")
            
            self.writer.add_scalars("pinn/pinn_loss",{
                "loss_mse": loss_mse.item(),
                "loss_res1": loss_res1.item(),
                "loss_res2": loss_res2.item(),
                "loss":loss.item()
            }, self.cur_epoch+epoch)
            
            if last:
                if loss.item()< best_loss:
                    best_loss = loss.item()
                    last_improvement=epoch
                    torch.save(self.nn.state_dict(), self.pinn_path+f'{count}_best.ckpt')
                
                if epoch-last_improvement>200:
                    break

        u_pred, l2 = self.evaluate()
        logging.info(f"full Field Error u: {l2}")   
        torch.save(self.nn.state_dict(), self.pinn_path+f'{count}.ckpt')     
        self.cur_epoch+=self.pinn_epoch    
    
    def predict(self, x):
        out = self.net_u(x)
        return out
    
    def pinn_loss_explicitly(self):
        #  -0.947836 * diff_t(u1,x1) + -0.949334 * diff3_t(u1,x1) + -0.484893 * n2_t(u1)
        # -0.5738601300366815 * diff(mul(div(u,u),n2(u)),x1) + 0.11269937552160604 * u
        # + -0.25388272324859346 * mul(u,u) + 0.08210855914489139 * diff2(u,x1) 
        # 0.09732894599437714 * diff2_t(u1,x1) + -0.49593397974967957 * diff_t(n2_t(u1),x1)
        u = self.net_u(torch.cat([*self.x_f, self.t_f], axis = 1))
        # import pdb;pdb.set_trace()
        ut = torch.autograd.grad(outputs=u,inputs = self.t_f,
                            grad_outputs = torch.ones_like(u),
                                create_graph=True)[0]
        # import pdb;pdb.set_trace()
        grad_can1 = u**2
        # if torch.any(torch.isnan(grad_can1)):
        #     import pdb;pdb.set_trace()

        # grad1 = torch.autograd.grad(outputs=grad_can1,inputs =self.x_f[0],
        #                             grad_outputs = torch.ones_like(u),
        #                             create_graph=True)[0]
        
        grad1 = torch.autograd.grad(outputs=u,inputs =self.x_f[0],
                                    grad_outputs = torch.ones_like(u),
                                    create_graph=True)[0]
        grad2 = torch.autograd.grad(outputs=grad1,inputs = self.x_f[0],
                            grad_outputs = torch.ones_like(u),
                            create_graph=True)[0]

        grad3 = torch.autograd.grad(outputs=grad2,inputs = self.x_f[0],
                            grad_outputs = torch.ones_like(u),
                            create_graph=True)[0]
        grad4 = torch.autograd.grad(outputs=grad3,inputs = self.x_f[0],
                            grad_outputs = torch.ones_like(u),
                            create_graph=True)[0]  
        # print(grad2_)
        # print(grad1)
        # rhs = -0.495933979*grad1+0.097328945994 *grad2
        rhs = -grad2-grad4-u*grad1
        residual = rhs-ut
        return torch.mean(torch.pow(residual,2)) 
    
    def evaluate(self,):
        u_pred =  self.net_u(self.x_star)
        u_pred_array = tensor2np(u_pred)
        l2 = l2_error(self.u_star, u_pred_array)
        return u_pred, l2
        
    def net_u(self, X ):
        if isinstance(X, list):
            X = torch.cat([*X], axis = 1)
        H =  2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        return self.nn(H)
    
    def generate_meta_data(self):
        # finite_differenece 
        u_pred, _ = self.evaluate()
        # xt = tensor2np(self.x_star)
        # x = xt[:,:-1]
        return u_pred, self.x_star, self.cache
    
    def load_model(self, ckpt_path, keep_optimizer=True):
        logging.info(f"load model from {ckpt_path}")
        total_state_dict = torch.load(ckpt_path, map_location='cpu')
        self.nn.load_state_dict(total_state_dict)
        if not keep_optimizer:
            self.optimizer = optim.Adam(self.nn.parameters(), lr=1e-3)
    @property   
    def collocation_point(self):
        return self.x_f, self.t_f
    @property
    def true_value(self):
        return self.x_star, self.u_star
    def close(self):
        self.writer.close()
    
    def set_optimizer(self, coef_lr = 0.001):

        self.optimizer = torch.optim.Adam([{'params': self.nn.parameters(), 'lr':0.001}, {'params': self.coef.coeff_vector.parameters(), 'lr':coef_lr}])
        
    
    def plot_coef(self):
        coefs = self.cache['coefs']
        sub_num = len(coefs[1])
        fig = plt.figure(figsize=(12,4), )
        for i in range(sub_num):
            coe_list = [coef[i] for coef in coefs]
            plt.subplot(1,sub_num, i+1 )
            plt.plot(list(range(1,len(coe_list)+1)), coe_list)
            
            plt.xlabel('epoch')
            plt.ylabel("coefs")
        
        plt.savefig(self.pic_path+'coef.png' )

    def reconstructed_field_evaluation(self, path = None):
        if path is None:
            path = os.path.split(self.pretrain_path)

        ckpt_path = path+f'/dso_{self.dataset_name}_0_pinn2_best.ckpt'
        # import pdb;pdb.set_trace()
        self.load_model(ckpt_path)
        u_pred, _ = self.evaluate()
        u_pred1 = u_pred[:,0].reshape(self.shape[0], -1)
        np.savez(path+'/predicted.npz', u_pred=tensor2np(u_pred1), x_star= tensor2np(self.x_star), u_star = self.u_star)

        if u_pred.shape[1]>1:
            for i in range(1, u_pred.shape[1]):
                u_pred1 = u_pred[:,i].reshape(self.shape[0], -1)
                np.savez(path+f'/predicted{i}.npz', u_pred=tensor2np(u_pred1),x_star= tensor2np(self.x_star), u_star = self.u_star )
    
            
        



if __name__ == '__main__':
    
    pass
            