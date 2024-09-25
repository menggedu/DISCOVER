from dso import DeepSymbolicOptimizer_PDE

#outer packages
import collections
import time
import os
import sys
import pickle
import torch
import warnings

from dso.config import load_config
from dso.pinn import PINN_model

if __name__  == "__main__":
    # Create and train the model
    warnings.filterwarnings('ignore', 'Intel MKL ERROR')
    pde = sys.argv[1]
    folder  = sys.argv[2]
    # import pdb;pdb.set_trace()
    output_path = sys.argv[3]
    data_name = sys.argv[4]
    device = torch.device('cpu')
    config_file = f"./dso/config/{folder}/config_pde_{pde}.json"
    config = load_config(config_file)
    config_pinn = config["pinn"]
    # import pdb;pdb.set_trace()
    pinn = PINN_model(
                './dso/log',
                config_pinn,
                data_name,
                device
            )

    pinn.reconstructed_field_evaluation(output_path)

