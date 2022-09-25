from dso import DeepSymbolicOptimizer_PDE

#outer packages
import collections
import time
import os
import sys
import pickle





if __name__  == "__main__":
    # Create and train the model
    machine = sys.argv[1]
    pde = sys.argv[2]
    folder  = sys.argv[3]
    # sample_num = sys.argv[3]
    # pde_config = {

        
    # }
    if machine == 'windows':
        model = DeepSymbolicOptimizer_PDE(f"C:/Users/lthpc/Nutstore/1/code/code/PDE_RL_dso/dso/dso/config/{folder}/config_pde_{pde}.json")
    elif machine == 'mac':

        model = DeepSymbolicOptimizer_PDE(f"/Users/doumengge/NutstoreCloudBridge/code/code/PDE_RL_dso/dso/dso/config/{folder}/config_pde_{pde}.json")
    
    start = time.time()
    result = model.train()
    result["t"] = time.time() - start
    result.pop("program")
    print("cost time : ",result["t"])
    save_path = model.config_experiment["save_path"]
    summary_path = os.path.join(save_path, "summary.csv")

    with open(f'{pde}.pkl', 'wb') as f:
        pickle.dump(result, f)
        
