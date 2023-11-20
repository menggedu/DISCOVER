from dso import DeepSymbolicOptimizer_PDE

#outer packages
import collections
import time
import os
import sys
import pickle

import warnings




if __name__  == "__main__":
    # Create and train the model
    warnings.filterwarnings('ignore', 'Intel MKL ERROR')
    pde = sys.argv[1]
    folder  = sys.argv[2]
    # sample_num = sys.argv[3]
    # pde_config = {

    # print(os.path.abspath(os.curdir))  
    # }
    # if machine == 'windows':
    #     model = DeepSymbolicOptimizer_PDE(f"D:/menggedu/pde_discovery/DISCOVER/dso/dso/config/{folder}/config_pde_{pde}.json")
    # elif machine == 'remote':
    model = DeepSymbolicOptimizer_PDE(f"./dso/config/{folder}/config_pde_{pde}.json")
    
    start = time.time()
    result = model.train()
    cost_time=time.time() - start
    # result.pop("program")
    print("cost time : ",cost_time)
    # save_path = model.config_experiment["save_path"]
    # summary_path = os.path.join(save_path, "summary.csv")

    # with open(f'{pde}.pkl', 'wb') as f:
    #     pickle.dump(result, f)
        
