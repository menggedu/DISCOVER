import torch
import numpy as np

from dso.task.pde.utils_nn import torch_diff

def mse_loss(prediction, target):
    '''Loss functions for the MSE loss. Calculates loss for each term in list.'''
    loss = torch.mean((prediction - target)**2, dim=0)
    return loss

def pinn_loss(model,p,x,t,w, extra_gradient=False):
    """
    embed discoverd equation in the pinn-training manner

    Args:
        model (_type_): NN
        p (_type_): Program of the discovered equation
        x (_type_): x
        t (_type_): t
        w (_type_): coefficients of each fucntion terms

    Returns:
        _type_: pinn loss 
    """
    u = [model.net_u(torch.cat([*x, t], axis= 1))]
    ut = torch.autograd.grad(outputs=u[0][:,0:1],inputs = t,
                                grad_outputs = torch.ones_like(u[0][:,0:1]),
                                 create_graph=True)[0]
    
  
    residual = p.STRidge.calculate_RHS(u,x,ut,w, extra_gradient=extra_gradient)
       
    loss = torch.mean(torch.pow(residual,2))   
    return loss   

def pinn_multi_loss(model,p,x_list,t_list,w):
    pass
