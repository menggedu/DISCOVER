# DISCOVER: Deep identification of symbolic open-form PDEs via enhanced reinforcement-learning

# Abstract
The working mechanisms of complex natural systems tend to abide by concise and profound partial differential equations (PDEs). Methods that directly mine equations from data are called PDE discovery, which reveals consistent physical laws and facilitates our interaction with the natural world. In this paper, an enhanced deep reinforcement-learning framework is proposed to uncover symbolic open-form PDEs with little prior knowledge. Specifically, (1) we first build a symbol library and define that a PDE can be represented as a tree structure. Then, (2) we design a structure-aware recurrent neural network agent by combining structured inputs and monotonic attention to generate the pre-order traversal of PDE expression trees. The expression trees are then split into function terms, and their coefficients can be calculated by the sparse regression method. (3) All of the generated PDE candidates are first filtered by some physical and mathematical constraints, and then evaluated by a meticulously designed reward function considering the fitness to data and the parsimony of the equation. (4) We adopt the risk-seeking policy gradient to iteratively update the agent to improve the best-case performance. The experiment demonstrates that our framework is capable of mining the governing equations of several canonical systems with great efficiency and scalability.

# Installation


The installation details can be found in [DSO](https://github.com/brendenpetersen/deep-symbolic-optimization), we recommond the Installation of all tasks.

# Run
Current version of the code is a pre-released version and can not be run durectly. To be updated soon.




# Reference

(1) Petersen et al. 2021 Deep symbolic regression: Recovering mathematical expressions from data via risk-seeking policy gradients. ICLR 2021.  [Paper](https://openreview.net/forum?id=m5Qsh0kBQG)

(2) Mundhenk, T., Landajuela, M., Glatt, R., Santiago, C. P., & Petersen, B. K. (2021). Symbolic Regression via Deep Reinforcement Learning Enhanced Genetic Programming Seeding. Advances in Neural Information Processing Systems, 34, 24912-24923.  [Paper](https://proceedings.neurips.cc/paper/2021/file/d073bb8d0c47f317dd39de9c9f004e9d-Paper.pdf)

(3) Code from DSO https://github.com/brendenpetersen/deep-symbolic-optimization

# Copyright statement

The code of this repository is developed for PDE discovery tasks based on the framework of DSO
, and the copyright of the main code is owned by the original authors of DSO and the organizations to which they belong. This repository is not available for commercial use.
