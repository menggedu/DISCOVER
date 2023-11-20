

import sys
# from data_load import load_data
import os
# print(os.path.abspath('.'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# print(sys.path)
from dso.task.pde.data_load import load_data, load_data_2D

import numpy as np
import torch
import torch.nn as nn
import scipy.io as scio
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import random
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as Data
from sklearn.metrics import mean_squared_error


# noise
def prepocess_data(x,t,u_true):
    shape = u_true.shape
    if len(shape) == 2:
        n, m =shape

        
        u = u_true+data_noise_level*np.std(u_true)*np.random.randn(*u_true.shape)
        Y_raw = pd.DataFrame(u.reshape(-1,1))
        X1 = np.repeat(x[0].reshape(-1,1), m, axis=1)
        X2 = np.repeat(t.reshape(1,-1), n, axis=0)
        
        X_raw_norm = pd.concat([pd.DataFrame(X1.reshape(-1,1)), pd.DataFrame(X2.reshape(-1,1))], axis=1, sort=False)
        # np.save("./dso/task/pde/noise_data/chafee-infante_sample_data.npy", X_raw_norm.values)
        
        
        X = ((X_raw_norm-X_raw_norm.mean()) / (X_raw_norm.std()))
        Y = ((Y_raw-Y_raw.mean()) / (Y_raw.std()))
        return X, Y, u, Y_raw.std()[0], Y_raw.mean()[0]    
    elif len(shape) == 3:
        import pdb;pdb.set_trace()
        t_len, n,m = shape
        u = u_true+data_noise_level*np.std(u_true)*np.random.randn(*u_true.shape)
        Y_raw = pd.DataFrame(u.reshape(-1,1))
        
        xx,yy= np.meshgrid( x[0].reshape(-1,1),x[1].reshape(-1,1))
        # import pdb;pdb.set_trace()
        xx,tt= np.meshgrid(xx,t)
        yy,tt= np.meshgrid(yy,t)
       
        # import pdb;pdb.set_trace()
        input_data = np.concatenate(( tt.reshape(-1,1),xx.reshape(-1,1), yy.reshape(-1,1)), axis = 1)
        X_raw_norm = pd.DataFrame(input_data)

        X = ((X_raw_norm-X_raw_norm.mean(axis = 0)) / (X_raw_norm.std(axis=0)))
        # Y = ((Y_raw-Y_raw.mean()) / (Y_raw.std()))  
        Y = Y_raw
        return X, Y, u, 1,0

# MetaData Train the NN model


class DNN(nn.Module):
    def __init__(self,n_feature,n_hidden,n_output,
                 layer_num=3):
        super(DNN,self).__init__()
        self.fc1 = nn.Linear(n_feature, n_hidden)
        self.mid_linear= nn.ModuleList()
        for _ in range(layer_num):
            self.mid_linear.append(nn.Linear(n_hidden, n_hidden))
       
        self.predict = nn.Linear(int(n_hidden),n_output)
    def forward(self,x):
        out = torch.sin((self.fc1(x)))
        for layer in range(len(self.mid_linear)):
            out = torch.sin(layer(out))

        out = self.predict(out) 
        return out
    
class DNN2(nn.Module):
    def __init__(self,n_feature,n_hidden,n_output,
                 layer_num=3):
        super(DNN2,self).__init__()
        self.fc1 = nn.Linear(n_feature, n_hidden)

        self.fc2 = nn.Linear(n_hidden, n_hidden)
       
        self.predict = nn.Linear(int(n_hidden),n_output)
        
    def forward(self,x):
        out = torch.sin((self.fc1(x)))
        out = torch.sin((self.fc2(out)))
        out = torch.sin((self.fc2(out)))
        out = torch.sin((self.fc2(out)))
        out = self.predict(out) 
        return out

def build_model(num_feature,
            device,
            hidden_dim = 32,
           ):
     model = DNN2(num_feature, hidden_dim, 1).to(device)
     return model
 
def model_NN(x_train, y_train,
             x_test,y_test,
            num_feature,
            device,
            ckpt_path,
            hidden_dim = 32,
            max_epoch = 10*1000,
            
            ):

    logging.info('Building NN model')
    display_step = 500 #int(max_epoch/100)
    
    x = torch.from_numpy(x_train.values).float()
    y = torch.from_numpy(y_train.values).float()
    x, y =Variable(x).to(device), Variable(y).to(device)
    # x_test,y_test = Variable(torch.from_numpy(x_test.values).float()).to(device),Variable(torch.from_numpy(y_test.values).float()).to(device)
    # 训练模型
    train_data = Data.TensorDataset(x,y)
    batch_size = len(x)
    data_loader = Data.DataLoader(train_data, batch_size =batch_size,shuffle = True, drop_last = False )
    model = build_model(num_feature,device, hidden_dim)
    # if  os.path.exists(f'{ckpt_path}latest.ckpt'):
    #     logging.info("load model")
    #     model.load_state_dict(torch.load(f'{ckpt_path}latest.ckpt', map_location=device))
    criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.1)
    # optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum =0.9, weight_decay=1e-2)
    optimizer = optim.Adam(model.parameters())
    for epoch in range(max_epoch):
        # for i, (train_x,train_y) in enumerate(train_data):
        optimizer.zero_grad()
        # import pdb;pdb.set_trace()
        y_pred = model(x)
        loss = criterion(y_pred, y)

        loss.backward()

        optimizer.step()
     
        if (epoch+1)%display_step == 0:
            torch.save(model.state_dict(), ckpt_path+f'{epoch+1}.ckpt') 
            model.eval()
            y_test_pred = model(x_test)
            test_loss = criterion(y_test_pred, y_test)
            logging.info('step %d, loss= %.6f, test_loss= %.6f'%(epoch+1, loss.cpu().data,test_loss.cpu().data ))
            model.train()
    
    y_pred_train = model(x)
    y_pred_train = y_pred_train.cpu().data.numpy().flatten()
    return y_pred_train, model


# split data
def batch_data(X,Y,ratio = 0.8):
    data_num = Y.shape[0]
    total_ID = list(range(0,data_num))
    def split(full_list,shuffle=False,ratio=0.2):
        n_total = len(full_list)
        offset = int(n_total * ratio)
        if n_total==0 or offset<1:
            return [],full_list
        if shuffle:
            random.shuffle(full_list)
        sublist_1 = full_list[:offset]
        sublist_2 = full_list[offset:]
        # import pdb;pdb.set_trace()
        if len(sublist_2) == 0:
            sublist_2 = sublist_1
        return sublist_1,sublist_2
    train_index, test_index = split(total_ID, shuffle=True, ratio = ratio)
    # import pdb;pdb.set_trace()
    x_train, x_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], Y.iloc[train_index], Y.iloc[test_index]
    logging.info('index shape:')
    logging.info(np.shape(train_index))
    logging.info(np.shape(test_index))
    return x_train, x_test, y_train, y_test

def R2(y_test, y_pred):
    SSE = np.sum((y_test - y_pred)**2)          # 残差平方和
    SST = np.sum((y_test-np.mean(y_test))**2) #总体平方和
    R_square = 1 - SSE/SST # 相关性系数R^2
    return R_square

def test(model, ckpt_path,x,y_real, std, mean, normalize):
    
    # n,m = y_real.shape
    y_real = y_real.reshape(-1)
    y_pred = model(x)
    y_pred = y_pred.cpu().data.numpy()
    # if normalize:
    #     y_pred = y_pred*std+mean
    diff = (y_pred-y_real)
    RMSE = mean_squared_error(y_pred,y_real)** 0.5
    logging.info(f"RMSE: {RMSE}")
    logging.info(np.max(diff))
    logging.info(np.min(diff))
    logging.info(np.mean(np.abs(diff)))
    logging.info(np.median(np.abs(diff)))


def eval_result(y_test, y_pred, y_train, y_pred_train):

    logging.info('Evaluating model performance...')
    y_test = np.array(y_test).reshape((-1,1)) 
    y_pred = np.array(y_pred).reshape((-1,1))
    y_train = np.array(y_train).reshape((-1,1)) 
    y_pred_train = np.array(y_pred_train).reshape((-1,1))
   
    # logging.info('The std(y_pred_train) is:%f'%np.std(y_pred_train))
    if len(y_test) == 0:
        RMSE = 0
    else:
        RMSE = mean_squared_error(y_test, y_pred) ** 0.5
    RMSE_train = mean_squared_error(y_train, y_pred_train) ** 0.5
    logging.info(f'The RMSE of prediction is: {RMSE}')
    logging.info(f'The RMSE of prediction of training dataset is: {RMSE_train}')
    if len(y_test) == 0:
        R_square = 0
    else:
        R_square = R2(y_test, y_pred)
    R_square_train = R2(y_train, y_pred_train)
    
    logging.info(f'The R2 is: {R_square}')
    logging.info(f'The R2 of training dataset is: {R_square_train}')
    return RMSE, RMSE_train, R_square, R_square_train

if __name__ == '__main__':
    import sys
    import os
    import logging
    
    
    
    dataset =  sys.argv[1]
    data_noise_level = float(sys.argv[2])

    ratio = float(sys.argv[3])
    mode = int(sys.argv[4])
    file = sys.argv[5]
    device = torch.device('cuda:0')
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, filename=file)
    
    logging.info(f"data_noise_level: {data_noise_level}")
    logging.info(f'data ratio: {ratio}')
    seed = 0 
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    ckpt_path = f'./dso/task/pde/ckpt_remote/noise{data_noise_level}_data_amount{ratio}/{dataset}_2d_min/'
    os.makedirs(ckpt_path,exist_ok=True)
    max_epoch = 10*10000

    
    load_class = load_data_2D if '2D' in dataset else load_data
    # import pdb;pdb.set_trace()
    # u,x,t, ut,sym_true, n_input_var,test_list = load_class(dataset, data_noise_level, data_amount, training= True)

    u_true,x,t, ut,sym_true, n_input_var,test_list = load_class(dataset, 0, training=True)
    # data process
    hidden_dim= 64 
    number_feature = 2 if '2D' not in dataset else 3
    max_epoch*=1
    normalize = True
    X,Y,U_noise,std,mean = prepocess_data(x,t,u_true)
    u_shape = u_true.shape
    u_raw = u_true.reshape(-1)
    # import pdb;pdb.set_trace()
    x_train, x_test, y_train, y_test = batch_data(X,Y,ratio=ratio)
    # np.save("./dso/task/pde/noise_data/chafee-infante_sample_data.npy", x_train)
    # return 
    logging.info(y_test.shape)
    x_test, y_test = Variable(torch.from_numpy(x_test.values).float()).to(device), Variable(torch.from_numpy(y_test.values).float().to(device))
    x_all = torch.from_numpy(X.values).float().to(device)
    y_all = torch.from_numpy(u_raw.reshape(-1,1)).float().to(device)
    
    #train
    if mode == 0:
        y_pred_train, model = model_NN(x_train, y_train,x_test,y_test, number_feature,
                                        device,
                                        ckpt_path,
                                        hidden_dim = hidden_dim,
                                        max_epoch = max_epoch
                                                    )
        
        # predict
        y_all_predict = model(x_all)
        y_pred = model(x_test)
        y_all_pred = y_all_predict.cpu().data.numpy().flatten()
        if normalize:
            y_all_pred = y_all_pred*std+mean
        x_test, y_test = x_test.cpu().data.numpy().flatten(), y_test.cpu().data.numpy().flatten()
        y_pred, y_pred_train = y_pred.cpu().data.numpy().flatten(), y_pred_train
        # logging.info('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
        logging.info('The rmse of training is:', mean_squared_error(y_train, y_pred_train) ** 0.5)
        logging.info('The rmse of raw data  is:', mean_squared_error(u_raw, y_all_pred) ** 0.5)

        # eval
        RMSE, RMSE_train, R_square, R_square_train = eval_result (y_test, y_pred, y_train, y_pred_train)
        result_test_real = y_test*Y.std()[0]+Y.mean()[0]
        result_pred_real = y_pred*Y.std()[0]+Y.mean()[0]
        logging.info('Neural network generated')
    else:
        # device = 'cpu'
        # x_all = x_all.to(device)
        # x_test = x_test.to(device)
        # y_test = y_test.cpu().data.numpy().flatten()
        # x_train = x_train.to(device)
        # y_test = y_test.cpu().data.numpy().flatten()
        # import pdb;pdb.set_trace()
        best_rmse =100
        best_epoch = 0
        data_dir = f'./dso/task/pde/noise_data'
        os.makedirs(data_dir, exist_ok=True)
        
        for i in range(1,400):
            i = i*500
        # i='latest
            ckpt_path2= f'{ckpt_path}{i}.ckpt'
            if not os.path.exists(ckpt_path2):
                continue
            data_path = f'{data_dir}/{dataset}_data_amount{ratio}_noise{data_noise_level}_small.npy'
            model = build_model(number_feature,device, hidden_dim)
            # import pdb;pdb.set_trace()
            model.load_state_dict(torch.load(ckpt_path2, map_location=device))
            y_all_predict = model(x_all)
            y_all_pred = y_all_predict.cpu().data.numpy().flatten()
            if normalize:
                y_all_pred = y_all_pred*std+mean
            cur_rmse = mean_squared_error(u_raw, y_all_pred) ** 0.5
            logging.info(f'The rmse of raw data  is:{cur_rmse}')
            if best_rmse>cur_rmse:
                best_rmse = cur_rmse
                best_epoch = i
            # break
                
        # best_epoch =20000    
        ckpt_path_best= f'{ckpt_path}{best_epoch}.ckpt'
        model.load_state_dict(torch.load(ckpt_path_best, map_location=device))
        y_all_predict = model(x_all)
        y_all_pred = y_all_predict.cpu().data.numpy().flatten()
        if normalize:
            y_all_pred = y_all_pred*std+mean
        y_all_pred = y_all_pred.reshape(*u_shape)
        np.save(data_path, y_all_pred)
        logging.info(f"best epcoh is {best_epoch}, best rmse is {best_rmse}")
        torch.save(model.state_dict(), f'{ckpt_path}best.ckpt') 
            # logging.info(mean_squared_error(u_raw, ) ** 0.5)
            # test(model,ckpt_path, x_all, y_test,std,mean, normalize)