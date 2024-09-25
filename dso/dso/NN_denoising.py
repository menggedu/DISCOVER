

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


from dso.task.pde.utils_nn import ANN, PDEDataset
from dso.task.pde.utils_noise import normalize, unnormalize
from dso.utils import  print_model_summary, eval_result, tensor2np

logger=logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",level=logging.INFO)


class Denoise_NN():
    
    def __init__(self,args,
                 ):
        self.args=args
        # build the smooth net
        self.nn = ANN(number_layer = args.num_layer,
                 input_dim = args.input_dim,
                 n_hidden = args.hidden_num,
                 out_dim = args.out_dim,
                 activation = args.activation)
        logging.info(self.nn)
        total_params = print_model_summary(self.nn)
        logging.info(f"Total params are {total_params}")
        
        # dataset   `           `
        self.pde_data = PDEDataset(args.data_name, 
                                   args.noise_level, 
                                   args.normalize_type,
                                   args.train_ratio,
                                   args.data_efficient)
        
        batch_num = args.batch_num if args.batch_num is not None else 1
        self.train_data= DataLoader(self.pde_data, shuffle=False, batch_size = int(len(self.pde_data)/batch_num))
        self.x_val, self.y_val = self.pde_data.test_data
        self.x_train, self.y_train = self.pde_data.train_data
        self.x_flat, self.y_flat, self.y_flat_true= self.pde_data.data
        self.normalize_params = self.pde_data.normalize_params
        self.U_shape = self.pde_data.U_shape
        self.U_true = self.pde_data.u_true
        
        # training parameters
        self.loss_type = args.loss_type #gls_thresh_loss or mse
        self.optimizer =optim.Adam(self.nn.parameters(), lr=1e-3)
            
        # saving parameters
        self.task_name = f'{args.data_name}_noise={args.noise_level}_data_ratio={args.train_ratio}'
        self.model_dir = 'out'
        self.model_name = f"{self.model_dir}/checkpoint/{self.task_name}"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        # self.animations_dir = f'{self.model_dir}/animations'
        # if not os.path.exists(self.animations_dir):
        #     os.makedirs(self.animations_dir)
            
        # plotting parameters
        self.plot_dir = f'{self.model_dir}/plots'
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        
        #data-sa'vsave
        self.data_dir = './dso/task/pde/noise_data_new' 
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
            
    def mse_loss(self, y_pred,y_true):
        criterion = nn.MSELoss()
        loss = criterion(y_pred, y_true)
        return loss
    
    def gls_thresh_loss(self, y_pred, y_true):
        gamma = 1.0
        thresh = 0.0001
        y_pred_abs = torch.abs(y_pred)
        y_prop = (y_pred_abs>=thresh).to(y_pred)*y_pred_abs + (y_pred_abs<thresh).to(y_pred)
        
        mspe = torch.mean(torch.square((y_pred-y_true)/(y_prop**gamma)))
        
        penalty = torch.mean(torch.square((y_pred>1).to(y_pred)*y_pred_abs + (y_pred<0).to(y_pred)*(y_pred_abs+1.0)))
        return mspe+penalty
    
    def test(self, x_val, y_val):
        
        self.nn.eval()
        y_pred_val = self.nn(x_val)
        val_dict = {'info':''}
        if self.loss_type == 'gls_thresh':
            val_loss = self.gls_thresh_loss(y_pred_val, y_val)
            # val_loss = mspe_val+penalty_val
            # val_dict['penalty_val'] = penalty_val.cpu().data.item()   
            # val_dict['mspe'] = mspe_val.cpu().data.item()     
            # val_dict['info'] += f"val_penalty = {val_dict['penalty_val']:.6f} val_mspe = {val_dict['mspe']:.6f} "
        else:
            val_loss = self.mse_loss(y_pred_val, y_val)
        # val_dict['mse'] = val_mse.cpu().data.item()
        val_dict['loss'] = val_loss.cpu().data.item()
        val_dict['info'] += f"val_loss = {val_dict['loss']:.6f} "

        self.nn.train()
        
        return val_dict

        
    def evaluate(self, x=None, y=None, metrics='RMSE'):
        if x is None:
            x = self.x_flat
            y = self.y_flat_true
        logging.info('Evaluating model performance...')
        self.load_best_model()
        y_pred = self.nn(x)
        y_pred = tensor2np(y_pred)
        y_true = tensor2np(y)
        x_flat = tensor2np(self.x_flat)
        #unnormalize
        U_params = self.normalize_params[0]
        x_params = self.normalize_params[1]
        
        y_pred = unnormalize(y_pred, U_params)
        y_true = unnormalize(y_true, U_params)
        # XT = unnormalize(x_flat, x_params)
        
        # X = torch.from_numpy(XT[:,:-1].reshape(self.U_shape) )          # (time x space)
        # T = torch.from_numpy(XT[:,-1].reshape(self.U_shape) ) 
        
        RMSE, R_square= eval_result(y_pred, y_true)
        
        logging.info(f"RMSE is {RMSE}, R2 is {R_square}")
        
        self.plotter(-1, self.U_true, -1, normalize_res = True)
        
        
    def to_tensor(self, device):
        # data
        self.x_val= self.x_val.to(device)
        self.y_val= self.y_val.to(device)
        self.x_train= Variable(self.x_train).to(device)
        self.y_train= Variable(self.y_train).to(device)
        self.x_flat = self.x_flat.to(device)
        self.y_flat = self.y_flat.to(device)
        self.y_flat_true = self.y_flat_true.to(device)
        self.U_true= self.U_true.to(device)      # (time x space)
        
        self.U_noise = self.y_flat.reshape(self.U_shape)
        
    def train(self, epochs, device,early_stopper=1000, display_step=500, saved_path = None, load_path=None):
        
        '''
        Trains the surface fitting neural network. Data is loaded and split into
        training and validation sets. Then for some number of epochs, the surface
        fitter is trained until stopping criteria is reached. Every time 
        validation error improves, plots and saves.
        
        Input:
        
            epochs        : max number of epochs for training
            batch_size    : batch size for training
            early_stopper : early stopping based on validation loss
            new_model     : boolean for starting training from scratch
            
        Output:
        '''
        
        # plot initial network performance
        # self.plotter(-1, x_flat, y_flat, U_shape, 0)

        #saved_path _load path:
        if saved_path == None:
            saved_path = self.model_name
            if not os.path.exists(saved_path):
                os.makedirs(saved_path, exist_ok=True)
        
        # initialize best validation loss
        best_val_loss = float('inf')
        
        # initialize plot counter
        plot_count = 1
        
        # start_epoch
        start_epoch = 0
        
        # continue training from last time?
        if load_path is not None:
            start_epoch, best_val_loss = self.load_model(load_path)
        

        self.to_tensor(device)
        self.nn.to(device)
        # begin training
        for epoch in range(start_epoch, start_epoch+epochs):
            
            # for i, (x,y) in enumerate(self.train_data):
                
            #     x = Variable(x).to(device)
            #     y_true= Variable(y).to(device)
            # for i in range(1):
            x = self.x_train
            y_true=self.y_train
            
            self.optimizer.zero_grad()
            y_pred = self.nn(x)
            # import pdb;pdb.set_trace()
            if self.loss_type == 'gls_thresh':
                loss=self.gls_thresh_loss(y_pred, y_true)
                # loss=penalty+mspe
                # log_info += 'thresh_loss = %.6f '%(penalty.cpu().data.item())
            else:
                loss = self.mse_loss(y_pred, y_true)
            log_info = "step %d train_loss = %.6f "%(epoch+1, loss.cpu().data.item())
            loss.backward()
            self.optimizer.step()
            evals_val = self.test(self.x_flat, self.y_flat_true)
            evals_loss = evals_val['loss']
            log_info += evals_val['info']
            
            # if epoch==98500:
            #     self.save_model(saved_path+f'/best_98500.ckpt', epoch, best_val_loss)
                

            # save model if val loss improved
            if evals_loss < best_val_loss:
                
                # set the iteration for the last improvement to current
                last_improvement = epoch
                
                # update the best-known validation loss
                best_val_loss = evals_loss
                
                # printed to show improvement found
                improved_str = '*'*3
                
                # save model to JSON
                self.save_model(saved_path+f'/best.ckpt', epoch, best_val_loss)
                
                # plot learned surface

                    # update counter
    
            else:
                
                # printed to show no improvement 
                improved_str = ''
                
            if (epoch+1)%10000 == 0:
                self.plotter(epoch,self.U_noise, plot_count)
                plot_count += 1
                
            if (epoch+1)% display_step == 0:
                if (epoch+1)%10000 == 0:
                    self.save_model(saved_path+f'/{epoch+1}.ckpt', epoch, evals_loss)
                        
                # print the progress
                logger.info(log_info + improved_str)
            
            # if no improvement found for some time, stop training
            if epoch - last_improvement > early_stopper:
                
                logger.info("No improvement found in a while, stopping training.")
                self.plotter(epoch,self.U_noise, plot_count)
                break
                
            # if the cost explodes, kill the process
            # if math.isnan() or math.isinf(evals_train):
            #     sys.exit("Optimization failed, train cost = inf/nan.")
            # if math.isnan(evals_val) or math.isinf(evals_val):
            #     sys.exit("Optimization failed, val cost = inf/nan.")
        

    def save_model(self, ckpt_name, epoch, best_val):
        
        '''
        Saves weights of current surface fitting neural network. 
        '''
        state_dict = {
            'model': self.nn.state_dict(),
            'optim':self.optimizer.state_dict(),
            'epoch':epoch,
            'best_val': best_val,
            'args':self.args
        }
        torch.save(state_dict, ckpt_name) 

                
    def load_model(self, name):
        
        '''
        Loads weights into current surface fitting neural network. 
        '''
        logging.info(f"load model from {name}")
        total_state_dict = torch.load(name, map_location='cpu')
        self.nn.load_state_dict(total_state_dict['model'])
        self.optimizer.load_state_dict(total_state_dict['optim'])
        epoch = total_state_dict['epoch']
        best_val = total_state_dict['best_val']
        return epoch,best_val
    
    def load_best_model(self):
        
        name = self.model_name+'/best.ckpt'
        return self.load_model(name)
        
    def plotter(self, epoch,U_true, plot_count, normalize_res=True):
        
        '''
        Plots current progress of surface fitter. 
        
        Input:
        
            epoch      : current epoch
            X          : array of ordered pairs of coordinate vectors
            y_true     : array of flattened noisy surface values
            U_shape    : surface shape
            plot_count : plot iteration
            
        Output:
        '''
        
        # save name
        # import pdb;pdb.set_trace()
        name =self.plot_dir+'/'+self.task_name
        if not os.path.exists(name):
            os.makedirs(name)
        name += '/'+str(plot_count).zfill(3)+'.png'
        U_params = self.normalize_params[0]
        x_params = self.normalize_params[1]
        XT = self.x_flat.clone()
        
        # evaluate current network
        u_pred = self.nn(self.x_flat)
        
        # reshape into 2D arrays

        U_pred = u_pred.reshape(U_true.shape) # ((time x space)
        t_len = U_true.shape[0]
        t_plot = [1,2, t_len//2, t_len//2+1]
        XT = tensor2np(XT)
        
        U_pred =tensor2np(U_pred) 
        U_true = tensor2np(U_true)
        if normalize_res:
            XT = unnormalize(XT, x_params)
            U_pred = unnormalize(U_pred, U_params)
            U_true = unnormalize(U_true, U_params)
            # import pdb;pdb.set_trace()
        # compute maximum residual
        # import pdb;pdb.set_trace()
        fig = plt.figure(figsize=(16,9))
        if len(U_true.shape)==2:
            # import pdb;pdb.set_trace()
            X = XT[:,:-1].reshape(self.U_shape)           # (time x space)
            T = XT[:,-1].reshape(self.U_shape)           # (time x space)
            # import pdb;pdb.set_trace()
            max_res = np.max(np.abs(U_true - U_pred))
            
            # plot the surface
            x_steps = 2
            t_steps = x_steps*max(X.shape[1]//X.shape[0], 1)
            
            
            # learned surface and data
            ax = fig.add_subplot(121, projection='3d')
            ax.plot_surface(X, T, U_pred, color='r', alpha=0.4)
            # import pdb;pdb.set_trace()
            ax.scatter(X[::t_steps,::x_steps].reshape(-1), 
                    T[::t_steps, ::x_steps].reshape(-1), 
                    U_true[::t_steps,::x_steps].reshape(-1), c='k', s=1)
            
            # if np.max(U_true)>1:
            #     ax.set_zlim(np.min(U_true)-0.1,np.max(U_true)+0.1)
            # else:
            #     ax.set_zlim(0,1)
            plt.title('Learned Surface, Epoch = '+str(epoch+1))
            plt.xlabel('x')
            plt.ylabel('t')
        
            # surface and data residual 
            ax = fig.add_subplot(222)
            c = ax.pcolormesh(X.T, T.T, (U_true - U_pred).T)#, vmin=-max_res, vmax=max_res)
            ax.set_title('Residual')
            plt.xlabel('x')
            plt.ylabel('t')
            fig.colorbar(c, ax=ax)
            
            
            max_res = np.max(np.abs(U_pred))
            ax = fig.add_subplot(224)
            c = ax.pcolormesh(X.T, T.T,U_pred.T)#, vmin=-max_res, vmax=max_res)
            ax.set_title('U_pred')
            plt.xlabel('x')
            plt.ylabel('t')
            fig.colorbar(c, ax=ax)
            
        else:
            
            X1 = XT[:,0].reshape(self.U_shape)           # (time x space)
            X2 = XT[:,1].reshape(self.U_shape)  
            T = XT[:,-1].reshape(self.U_shape)           # (time x space)
            max_res = np.max(np.abs(U_true - U_pred))
            
            X1 = X1[:t_len]
            X2 = X2[:t_len]
            U_true,U_pred = U_true[t_plot], U_pred[t_plot]
            
            ax = fig.add_subplot(221)
            # c = ax.pcolormesh(X1, X2, (U_true - U_pred)[0], vmin=-max_res, vmax=max_res)
            c=ax.imshow((U_true - U_pred)[0], vmin=-max_res, vmax=max_res)
            ax.set_title('Residual')
            plt.xlabel('x')
            plt.ylabel('t')
            fig.colorbar(c, ax=ax)
            
            
            max_res = np.max(np.abs(U_pred))
            ax = fig.add_subplot(222)
            # c = ax.pcolormesh(X1, X2,U_pred[0], vmin=-max_res, vmax=max_res)
            c=ax.imshow(U_pred[0], vmin=-max_res, vmax=max_res)
            ax.set_title('U_pred')
            plt.xlabel('x')
            plt.ylabel('t')
            fig.colorbar(c, ax=ax)
            
            max_resi = np.abs((U_true - U_pred)[2]).max()
            ax = fig.add_subplot(223)
            # c = ax.pcolormesh(X1, X2, (U_true - U_pred)[0], vmin=-max_res, vmax=max_res)
            c=ax.imshow((U_true - U_pred)[2], vmin=-max_resi, vmax=max_resi)
            ax.set_title('Residual')
            plt.xlabel('x')
            plt.ylabel('t')
            fig.colorbar(c, ax=ax)
            
            
            max_res = np.max(np.abs(U_pred))
            ax = fig.add_subplot(224)
            # c = ax.pcolormesh(X1, X2,U_pred[0], vmin=-max_res, vmax=max_res)
            c=ax.imshow(U_pred[2], vmin=-max_res, vmax=max_res)
            ax.set_title('U_pred')
            plt.xlabel('x')
            plt.ylabel('t')
            fig.colorbar(c, ax=ax)
            
        # plt.show()
        # save and closeT
        plt.savefig(name)
        plt.close(fig)
    
    def predict(self, x=None):
        if x is None:
            x = self.x_flat
        return self.nn(x) 
 
    
    def save_data(self, X=None):
        

            
        # compute partial derivatives
        self.nn.eval()
        U_pred = self.predict(X)
        U_pred = U_pred.reshape(self.U_shape)
        
        # self.plotter(0, self.U_true, 0, normalize=True)
        U_pred = tensor2np(U_pred)
        x_train = tensor2np(self.x_train)
        y_train = tensor2np(self.y_train)
        x_all = tensor2np(self.x_flat)
        y_all = tensor2np(self.y_flat)
        y_all = y_all.reshape(self.U_shape)
        U_noise = tensor2np(self.U_noise)
        U_true = tensor2np(self.U_true)
        
        #unnormalize 
        U_params = self.normalize_params[0]
        X_params = self.normalize_params[1]
        U_mean, U_std = U_params
        x_mean,x_std = X_params
        
        x_train = unnormalize(x_train, X_params)
        x_all = unnormalize(x_all,X_params )
        U_noise = unnormalize(U_noise, U_params )
        U_true = unnormalize(U_true, U_params)
        U_pred = unnormalize(U_pred, U_params)
    
        
        np.savez(self.data_dir+'/'+self.task_name+'.npz',
                 U_noise = U_noise, U_true = U_true, U_pred = U_pred,
                 xt = x_train, xt_all = x_all,
                 U_mean = U_mean,  U_std = U_std,
                 x_mean = x_mean, x_std = x_std 
                 )
