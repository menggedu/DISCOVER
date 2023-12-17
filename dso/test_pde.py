from dso import DeepSymbolicOptimizer_PDE

#outer packages
import collections
import time
import os
import sys
import pickle

import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

if __name__  == "__main__":
    warnings.filterwarnings('ignore', 'Intel MKL ERROR')
    pde = sys.argv[1]
    folder  = sys.argv[2]

    # build model by passing the path of user-defined config file. 
    model = DeepSymbolicOptimizer_PDE(f"./dso/config/{folder}/config_pde_{pde}.json")
    
    # model training
    start = time.time()
    result = model.train()
    cost_time=time.time() - start

    print("cost time : ",cost_time)

    # save_path = model.config_experiment["save_path"]
    # summary_path = os.path.join(save_path, "summary.csv")

    # with open(f'{pde}.pkl', 'wb') as f:
    #     pickle.dump(result, f)
        
