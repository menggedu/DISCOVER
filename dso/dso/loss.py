import torch
import numpy as np

def mse_loss(prediction, target):
    '''Loss functions for the MSE loss. Calculates loss for each term in list.'''
    loss = torch.mean((prediction - target)**2, dim=0)
    return loss

def pinn_loss(model,p,x,t, w):
    """_summary_

    Args:
        model (_type_): _description_
        p (_type_): _description_
        x (_type_): _description_
        t (_type_): _description_
        w (_type_): coefficients of each fucntion terms

    Returns:
        _type_: _description_
    """
    u = model.net_u(torch.cat([x, t], axis= 1))
    ut = torch.autograd.grad(outputs=u,inputs = t,
                                grad_outputs = torch.ones_like(u),
                                 create_graph=True)[0]
    p.switch_tokens()
    
    residual = p.STRidge.calculate_RHS(u,[x],ut,w)
       
    loss = torch.mean(torch.pow(residual,2))   
    return loss   

def pinn_multi_loss(model,p,x_list,t_list,w):
    pass
