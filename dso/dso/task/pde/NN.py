

import sys
# from data_load import load_data
from dso.task.pde.data_load import load_data

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
    n, m = u_true.shape
    u = u_true+data_noise_level*np.std(u_true)*np.random.randn(n,m)
    Y_raw = pd.DataFrame(u.reshape(-1,1))
    X1 = np.repeat(x[0].reshape(-1,1), m, axis=1)
    X2 = np.repeat(t.reshape(1,-1), n, axis=0)
    # import pdb;pdb.set_trace()
    X_raw_norm = pd.concat([pd.DataFrame(X1.reshape(-1,1)), pd.DataFrame(X2.reshape(-1,1))], axis=1, sort=False)
    X = ((X_raw_norm-X_raw_norm.mean()) / (X_raw_norm.std()))
    Y = ((Y_raw-Y_raw.mean()) / (Y_raw.std()))
    return X, Y, u, Y_raw.std()[0], Y_raw.mean()[0]

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

    print('Building NN model')
    display_step = 1000 #int(max_epoch/100)
    
    x = torch.from_numpy(x_train.values).float()
    y = torch.from_numpy(y_train.values).float()
    x, y =Variable(x).to(device), Variable(y).to(device)
    # x_test,y_test = Variable(torch.from_numpy(x_test.values).float()).to(device),Variable(torch.from_numpy(y_test.values).float()).to(device)
    # 训练模型
    train_data = Data.TensorDataset(x,y)
    batch_size = len(x)
    data_loader = Data.DataLoader(train_data, batch_size =batch_size,shuffle = True, drop_last = False )
    model = build_model(num_feature,device, hidden_dim)
    if  os.path.exists(f'{ckpt_path}latest.ckpt'):
        print("load model")
        model.load_state_dict(torch.load(f'{ckpt_path}latest.ckpt', map_location=device))
    criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.1)
    # optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum =0.9, weight_decay=1e-2)
    optimizer = optim.Adam(model.parameters())
    for epoch in range(max_epoch):
        for i, (train_x,train_y) in enumerate(data_loader):
            optimizer.zero_grad()
            # import pdb;pdb.set_trace()
            y_pred = model(train_x)
            loss = criterion(y_pred, train_y)

            loss.backward()

            optimizer.step()
     
        if (epoch+1)%display_step == 0:
            torch.save(model.state_dict(), ckpt_path+f'{epoch+1}.ckpt') 
            model.eval()
            y_test_pred = model(x_test)
            test_loss = criterion(y_test_pred, y_test)
            print('step %d, loss= %.6f, test_loss= %.6f'%(epoch+1, loss.data.cpu(),test_loss.data.cpu() ))
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
        return sublist_1,sublist_2
    train_index, test_index = split(total_ID, shuffle=True, ratio = ratio)
    x_train, x_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], Y.iloc[train_index], Y.iloc[test_index]
    print('index shape:')
    print(np.shape(train_index))
    print(np.shape(test_index))
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
    print("RMSE: ", RMSE)
    print(np.max(diff))
    print(np.min(diff))
    print(np.mean(np.abs(diff)))
    print(np.median(np.abs(diff)))


def eval_result (y_test, y_pred, y_train, y_pred_train):

    print('Evaluating model performance...')
    y_test = np.array(y_test).reshape((-1,1)) 
    y_pred = np.array(y_pred).reshape((-1,1))
    y_train = np.array(y_train).reshape((-1,1)) 
    y_pred_train = np.array(y_pred_train).reshape((-1,1))
   
    print('The std(y_pred_train) is:',np.std(y_pred_train))
    if len(y_test) == 0:
        RMSE = 0
    else:
        RMSE = mean_squared_error(y_test, y_pred) ** 0.5
    RMSE_train = mean_squared_error(y_train, y_pred_train) ** 0.5
    print('The RMSE of prediction is:', RMSE)
    print('The RMSE of prediction of training dataset is:', RMSE_train)
    if len(y_test) == 0:
        R_square = 0
    else:
        R_square = R2(y_test, y_pred)
    R_square_train = R2(y_train, y_pred_train)
    
    print('The R2 is:', R_square)
    print('The R2 of training dataset is:', R_square_train)
    return RMSE, RMSE_train, R_square, R_square_train

if __name__ == '__main__':
    import sys
    import os
    dataset =  sys.argv[1]
    mode = int(sys.argv[2])
    device = torch.device('cuda:0')
    # dataset = 'Kdv'
    ratio = 0.8
    data_noise_level = 0.01
    seed =0 
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    ckpt_path = f'./dso/task/pde/ckpt/noise{data_noise_level}_data_amount{ratio}/{dataset}/'
    os.makedirs(ckpt_path,exist_ok=True)
    max_epoch = 5*10000
    hidden_dim=32
    normalize = True

    u_true,x,t, ut,sym_true, n_input_var,test_list = load_data(dataset, 0)
    # data process
    X,Y,U_noise,std,mean = prepocess_data(x,t,u_true)
    n,m = u_true.shape
    u_raw = u_true.reshape(-1)
    x_train, x_test, y_train, y_test = batch_data(X,Y,ratio=ratio)
    print(y_test.shape)
    x_test, y_test = Variable(torch.from_numpy(x_test.values).float()).to(device), Variable(torch.from_numpy(y_test.values).float().to(device))
    x_all = torch.from_numpy(X.values).float().to(device)
    y_all = torch.from_numpy(u_raw.reshape(-1,1)).float().to(device)
    
    #train
    if mode == 0:
        y_pred_train, model = model_NN(x_train, y_train,x_test,y_test, 2,
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
        # print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
        print('The rmse of training is:', mean_squared_error(y_train, y_pred_train) ** 0.5)
        print('The rmse of raw data  is:', mean_squared_error(u_raw, y_all_pred) ** 0.5)

        # eval
        RMSE, RMSE_train, R_square, R_square_train = eval_result (y_test, y_pred, y_train, y_pred_train)
        result_test_real = y_test*Y.std()[0]+Y.mean()[0]
        result_pred_real = y_pred*Y.std()[0]+Y.mean()[0]
        print('Neural network generated')
    else:
        # device = 'cpu'
        # x_all = x_all.to(device)
        # x_test = x_test.to(device)
        # y_test = y_test.cpu().data.numpy().flatten()
        # x_train = x_train.to(device)
        # y_test = y_test.cpu().data.numpy().flatten()
        # import pdb;pdb.set_trace()
        for i in range(1,100):
            i = i*1000 
        # i='latest'
            ckpt_path2= f'{ckpt_path}{i}.ckpt'
            data_path = f'./dso/task/pde/data/{dataset}_noise{data_noise_level}.npy'
            model = build_model(2,device, hidden_dim)
            model.load_state_dict(torch.load(ckpt_path2, map_location=device))
            y_all_predict = model(x_all)
            y_all_pred = y_all_predict.cpu().data.numpy().flatten()
            if normalize:
                y_all_pred = y_all_pred*std+mean
            print('The rmse of raw data  is:', mean_squared_error(u_raw, y_all_pred) ** 0.5)
                
            y_all_pred = y_all_pred.reshape(n,m)
        
        # plt.imshow(y_all_pred-u_true)
        # plt.colorbar()
        # plt.show()
        # np.save(data_path, y_all_pred)
            # print(mean_squared_error(u_raw, ) ** 0.5)
            # test(model,ckpt_path, x_all, y_test,std,mean, normalize)