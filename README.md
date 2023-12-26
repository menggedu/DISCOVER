# DISCOVER: Deep identification of symbolically concise open-form PDEs via enhanced reinforcement-learning
The working mechanisms of complex natural systems tend to abide by concise and profound partial differential equations (PDEs). Methods that directly mine equations from data are called PDE discovery. In this respository, an enhanced deep reinforcement-learning framework is built to uncover symbolically concise
open-form PDEs with little prior knowledge. 



This repository provides the code and data for the following research paper:

DISCOVER: Deep identification of symbolically concise open-form PDEs via enhanced reinforcement-learning. [PDF](https://arxiv.org/pdf/2210.02181.pdf)




# Installation
```
conda create -n env_name python=3.7 # Create a Python 3 virtual environment with conda.
source activate env_name # Activate the virtual environment
```
From the root directory, 
```
pip install --upgrade setuptools pip
export CFLAGS="-I $(python -c "import numpy; print(numpy.get_include())") $CFLAGS" # Needed on Mac to prevent fatal error: 'numpy/arrayobject.h' file not found
pip install -e ./dso # Install  package and core dependencies

```
Extra dependencies,
```
pip install -r requirements.txt # Possible incompatibility may occurs due to the version of tensorboard. Manually installing it may be required.
pip install tensorboard 
```

# Functions
DISCOVER can be utilized to uncover open-form governing equations. When high-quality data are avaiable. Partial derivatives are evaluated by numerical differentiation on regular grids. When meansurements are nosiy, DNN can be optionally utilized to smoothe available data and generate meta data to reduce the impact of noise. The introduction of the whole framework can be found in paper [PDF](https://arxiv.org/pdf/2210.02181.pdf). GPU is not necessary when numerical differentiation is utilized. Note that automatic differentiation is also supported, but more computational resources are required.


# Run
Several benchmark datasets are provided, including Chafee-Infante equation, KdV equations and PDE_divide, etc. Run the script below can repeat the results in the paper.
 ```
 sh ./script_test/MODE1_test.sh
 ```
Data for the phase separation and oceanographic system are provided in [link](https://drive.google.com/drive/folders/1jEK_kYKgzlyVx4U3p2bVyCvq2LWfVZmv?usp=drive_link).
# Procedures for discovering a new dataset

* **Step 1**:  Put the dataset in the specified directory and write the data loading module. The default directory for benchmark datasets is './dso/dso/task/pde/data_new'.  The function of load_data for loading benchmark datasets is located at './dso/dso/task/pde/data_load.py' and it is called by class PDETask (in './dso/dso/task/pde/pde.py') to load relevant dataset.

```python
def load_data(data_path='./dso/task/pde/data_new/Kdv.mat'):
    """ load dataset in class PDETask"""    
    data = scio.loadmat(data_path)
    u=data.get("uu")
    n,m=u.shape
    x=np.squeeze(data.get("x")).reshape(-1,1)
    t=np.squeeze(data.get("tt").reshape(-1,1))
    n,m = u.shape #512, 201
    dt = t[1]-t[0]
    dx = x[1]-x[0]
    # true right-hand-side expressions
    sym_true = 'add,mul,u1,diff,u1,x1,diff3,u1,x1'

    n_input_var = 1 # space dismension
    n_state_var = 1 # number of the state variable 
    X=[] # define the space vector list, inlcuding x,y,...
    test_list =None

    ut = np.zeros((n, m)) # define the left-hand-side of the PDE
    dt = t[1]-t[0]
    X.append(x)
    
    for idx in range(n):
        ut[idx, :] = FiniteDiff(u[idx, :], dt)
    
    return [u],X,t,ut,sym_true, n_input_var,test_list,n_state_var
```

* **Step 2**: Hyperparameter setting. All of hyperparameters are passed to the class DeepSymbolicOptimizer_PDE through a JSON file. The default parameter setting is located at './dso/dso/config/config_pde.json'. Users can define their own parameters according to the example in the benchmark dataset './dso/dso/config/MODE1'.
```json
{
      // Experiment configuration.
   "experiment" : {

         // Root directory to save results.
         "logdir" : "./log/MODE1",
   
         // Random number seed. Don't forget to change this for multiple runs!
         "seed" : 0
      },
   
   "task" : {
      // Deep Symbolic PDE discovery
      "task_type" : "pde",

      // The name of the benchmark dataset (all of the avaiable data provided
      // can be found in ./dso/task/pde/data_new 
      // New dataset can be added according to the application.
      "dataset" : "Kdv",

      // To customize a function set, edit this! See functions.py for a list of
      // supported funcbatch_tions.
      "function_set": ["add", "mul", "div", "diff","diff2", "diff3","n2","n3"],
 
      // supported metrics.
      "metric" : "pde_reward",
      "metric_params" : [0.01],

      // Optional alternate metric to be used at evaluation time.
      "extra_metric_test" : null,
      "extra_metric_test_params" : [],

      // threshold for early stopping.
      "threshold" : 5e-4,
   },

   // Only the key training hyperparameters are listed here. See
   // config_pde.json for the full list.
   "training" : {
      "n_samples" : 50000,
      "batch_size" : 500,
      "epsilon" : 0.02,
      "early_stopping" : false
   },

   // Only the key RNN controller hyperparameters are listed here. See
   // config_pde.json for the full list.
   "controller" : {
      "learning_rate": 0.0025,
      "entropy_weight" : 0.03,
      "entropy_gamma" : 0.7,
      // Priority queue training hyperparameters.
      "pqt" : true,
      "pqt_k" : 10,
      "pqt_batch_size" : 1,
      "pqt_weight" : 0.0,
      "pqt_use_pg" : true,
      "attention": true
   },

```
* **Step 3**: Execute the PDE discovery task. Output and save results. An example is shown in './dso/test_pde.py'.
```python
from dso import DeepSymbolicOptimizer_PDE
import pickle 

data_name = 'KdV'
config_file_path = "./dso/config/MODE1/config_pde_KdV.json"
# build model by passing the path of user-defined config file. 
model = DeepSymbolicOptimizer_PDE(config_file_path)
    
# model training
result = model.train()

#save results
with open(f'{data_name}.pkl', 'wb') as f:
    pickle.dump(result, f)
        
```

# Reference

(1) Petersen et al. 2021 Deep symbolic regression: Recovering mathematical expressions from data via risk-seeking policy gradients. ICLR 2021.  [Paper](https://openreview.net/forum?id=m5Qsh0kBQG)

(2) Mundhenk, T., Landajuela, M., Glatt, R., Santiago, C. P., & Petersen, B. K. (2021). Symbolic Regression via Deep Reinforcement Learning Enhanced Genetic Programming Seeding. Advances in Neural Information Processing Systems, 34, 24912-24923.  [Paper](https://proceedings.neurips.cc/paper/2021/file/d073bb8d0c47f317dd39de9c9f004e9d-Paper.pdf)


# Copyright statement

The code of this repository is developed specifically for PDE discovery tasks based on the framework of [DSO](https://github.com/brendenpetersen/deep-symbolic-optimization). This repository is not available for commercial use.
