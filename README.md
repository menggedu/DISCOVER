# DISCOVER: Deep identification of symbolically concise open-form PDEs via enhanced reinforcement-learning
The working mechanisms of complex natural systems tend to abide by concise and profound partial differential equations (PDEs). Methods that directly mine equations from data are called PDE discovery. In this respoository, an enhanced deep reinforcement-learning framework is built to uncover symbolically concise
open-form PDEs with little prior knowledge. 



This repository provides the code and data for following research papers:

(1) DISCOVER: Deep identification of symbolically concise open-form PDEs via enhanced reinforcement-learning. [PDF](https://arxiv.org/pdf/2210.02181.pdf)

(2) Physics-constrained robust learning of open-formPDEsfrom limited and noisy data. [PDF](https://arxiv.org/ftp/arxiv/papers/2309/2309.07672.pdf)



# Installation
```
conda create -n env_name python=3.7 # Create a Python 3 virtual environment with conda.
source activate env_name # Activate the virtual environment
```
From the root directory, 
```
pip install --upgrade setuptools pip
export CFLAGS="-I $(python -c "import numpy; print(numpy.get_include())") $CFLAGS" # Needed on Mac to prevent fatal error: 'numpy/arrayobject.h' file not found
pip install -e ./dso # Install DSO package and core dependencies
```
The installation detail is similar to the  in [DSO](https://github.com/brendenpetersen/deep-symbolic-optimization), we recommond 

# Mode
There are two executation modes in DISCOVER. The first mode is designed for discoving PDEs from high-quality data. Based on a symbol library of basic operators and
operands, a structure-aware recurrent neural network agent is designed and seamlessly combined with the
sparse regression method to generate concise and open-form PDE expressions. All of the generated PDEs
are evaluated by a meticulously designed reward function by balancing fitness to data and parsimony, and
updated by the model-based reinforcement learning in an efficient way. Customized constraints and regulations are formulated to guarantee the rationality of PDEs in terms of physics and mathematics. Note that derivatives are evaluated by numerical differentiation on regular grids. DNN can be optionally utilized to smoothe available data and generate meta data to reduce the impact of noise. The introduction of the whole framework can be found in the first paper above. GPU is not necessary since the matrix calculation is based on Numpy.

The second mode is baseA robust verison of DISCOVER named R_DISCOVER can be utilized to handle sparse and noisy data.The framework operates through two alternating update processes: discovering and embedding. The discovering phase employs symbolic representation and a novel reinforcement learning (RL)-guided hybrid PDE generator to efficiently produce diverse open-form PDEs with tree structures. A neural network-based predictive model fits the system response and serves as the reward evaluator for the generated PDEs. PDEs with higher rewards are utilized to iteratively optimize the generator via the RL strategy and the best-performing PDE is selected by a parameter-free stability metric. The embedding phase integrates the initially identified PDE from the discovering process as a physical constraint into the predictive model for robust training. The traversal of PDE trees automates the construction of the computational graph and the embedding process without human intervention.
execuated by two alternating updation process: discovering and embedding. A NN is utilized to fit the system response and evaluate the reward by automatic differentiation. It is trained in a PINN manner when effective physical information are discovered. This mode is more suitable for the high-noisy scenarios. GPU resources are required to acclerate the searching process.

# Run
 Take burgers equation as an example. For first mode, run the script below.
 ```
 sh MODE1_test.sh
 ```
For second mode,
```
sh  MODE2_test.sh
```


# Reference

(1) Petersen et al. 2021 Deep symbolic regression: Recovering mathematical expressions from data via risk-seeking policy gradients. ICLR 2021.  [Paper](https://openreview.net/forum?id=m5Qsh0kBQG)

(2) Mundhenk, T., Landajuela, M., Glatt, R., Santiago, C. P., & Petersen, B. K. (2021). Symbolic Regression via Deep Reinforcement Learning Enhanced Genetic Programming Seeding. Advances in Neural Information Processing Systems, 34, 24912-24923.  [Paper](https://proceedings.neurips.cc/paper/2021/file/d073bb8d0c47f317dd39de9c9f004e9d-Paper.pdf)


# Copyright statement

The code of this repository is developed for PDE discovery tasks based on the framework of DSO
, and the copyright of the main code is owned by the original authors of DSO and the organizations to which they belong. This repository is not available for commercial use.
