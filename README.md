# DISCOVER: Deep identification of symbolically concise open-form PDEs via enhanced reinforcement-learning [PDF](https://arxiv.org/pdf/2210.02181.pdf)

# Abstract
The working mechanisms of complex natural systems tend to abide by concise and profound partial differential equations (PDEs). Methods that directly mine equations from data are called PDE discovery,
which reveals consistent physical laws and facilitates our adaptive interaction with the natural world. In
this paper, an enhanced deep reinforcement-learning framework is proposed to uncover symbolically concise
open-form PDEs with little prior knowledge. Particularly, based on a symbol library of basic operators and
operands, a structure-aware recurrent neural network agent is designed and seamlessly combined with the
sparse regression method to generate concise and open-form PDE expressions. All of the generated PDEs
are evaluated by a meticulously designed reward function by balancing fitness to data and parsimony, and
updated by the model-based reinforcement learning in an efficient way. Customized constraints and regulations are formulated to guarantee the rationality of PDEs in terms of physics and mathematics. The
experiments demonstrate that our framework is capable of mining open-form governing equations of several
dynamic systems, even with compound equation terms, fractional structure, and high-order derivatives, with
excellent efficiency. Without the need for prior knowledge, this method shows great potential for knowledge
discovery in more complicated circumstances with exceptional efficiency and scalability.

# Installation


The installation details can be found in [DSO](https://github.com/brendenpetersen/deep-symbolic-optimization), we recommond the Installation of all tasks.

# Run
 Example:
 sh train_single.sh
 
 **To be updated soon**.




# Reference

(1) Petersen et al. 2021 Deep symbolic regression: Recovering mathematical expressions from data via risk-seeking policy gradients. ICLR 2021.  [Paper](https://openreview.net/forum?id=m5Qsh0kBQG)

(2) Mundhenk, T., Landajuela, M., Glatt, R., Santiago, C. P., & Petersen, B. K. (2021). Symbolic Regression via Deep Reinforcement Learning Enhanced Genetic Programming Seeding. Advances in Neural Information Processing Systems, 34, 24912-24923.  [Paper](https://proceedings.neurips.cc/paper/2021/file/d073bb8d0c47f317dd39de9c9f004e9d-Paper.pdf)


# Copyright statement

The code of this repository is developed for PDE discovery tasks based on the framework of DSO
, and the copyright of the main code is owned by the original authors of DSO and the organizations to which they belong. This repository is not available for commercial use.
