"""Core deep symbolic optimizer construct."""

import warnings

from dso.utils import safe_merge_dicts
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import zlib
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import random
from time import time
from datetime import datetime
import logging

import numpy as np
import tensorflow as tf
import commentjson as json
import torch

from dso.task import set_task
from dso.controller import Controller
from dso.train import learn
from dso.prior import make_prior
from dso.program import Program,from_str_tokens
from dso.config import load_config
from dso.tf_state_manager import make_state_manager as manager_make_state_manager
from dso.core import DeepSymbolicOptimizer
from dso.pinn import PINN_model


class DeepSymbolicOptimizer_PDE(DeepSymbolicOptimizer):
    """
    Deep symbolic optimization model. Includes model hyperparameters and
    training configuration.

    Parameters
    ----------
    config : dict or str
        Config dictionary or path to JSON.

    Attributes
    ----------
    config : dict
        Configuration parameters for training.

    Methods
    -------
    train
        Builds and trains the model according to config.
    """

    def __init__(self, config=None, pde_config=None):
        self.set_config(config, pde_config)
        self.sess = None

    def setup(self, ):

        # Clear the cache and reset the compute graph
        Program.clear_cache()
        tf.reset_default_graph()

        # Generate objects needed for training and set seeds
        self.pool = self.make_pool_and_set_task()
        self.set_seeds() # Must be called _after_ resetting graph and _after_ setting task
        self.sess = tf.Session()

        # Save complete configuration file
        self.output_file = self.make_output_file()
        self.save_config()

        # Prepare training parameters
        self.prior = self.make_prior()
        self.state_manager = self.make_state_manager()
        self.controller = self.make_controller()
        self.gp_controller = self.make_gp_controller()
        self.denoise_pinn = self.make_pinn_model()
        
    def reset_up(self):
                # Clear the cache and reset the compute graph
        Program.clear_cache()
        tf.reset_default_graph()
        self.set_seeds() # Must be called _after_ resetting graph and _after_ setting task
        self.sess = tf.Session()
        self.controller = self.make_controller()
        self.gp_controller = self.make_gp_controller()
        
    def pretrain(self):
        # pretrain for pinn
        self.denoise_pinn.pretrain()
        
        # path = './log/debug/Burgers2_2023-02-19-235657/dso_Burgers2_0_pretrain.ckpt'
        # self.denoise_pinn.load_model(path)
        Program.reset_task(self.denoise_pinn)
        
        
    def pinn_train(self, best_p,count,coef=0.1,same_threshold=0,last=False):
        # sym = 'add,sub,diff,mul,div,u,u,n2,u,x1,add,u,mul,u,u,diff2,u,x1'
        # best_p = from_str_tokens(sym)
        # r = best_p.r_ridge
        # print(f"reward is {r}")
        self.denoise_pinn.train_pinn(best_p,count,coef=coef,same_threshold=same_threshold,last=last)
        Program.reset_task(self.denoise_pinn)
        
    def callLearn(self):
        result = learn(self.sess,
                    self.controller,
                    self.pool,
                    self.gp_controller,
                    self.denoise_pinn,
                    self.output_file,
                    **self.config_training) #early_stop  
        return result
    
    def callIterPINN(self):
        """iterative pinn and pde discovery
        """
        self.pretrain()
        cur_best_traversal=""
        same_threshold = 0 # same program
        prefix, _ = os.path.splitext(self.output_file)
        iter_num = self.config_pinn["iter_num"]
        for i in range(iter_num):
            if i>0:
                self.reset_up()
            print(f"The No.{i} pde discovery process")   
            result = self.callLearn()
            self.output_file = f"{prefix}_{i+1}.csv"
            best_p = result['program']
            
            # judgement for the last program
            if cur_best_traversal ==  repr(best_p):
                same_threshold +=1
                if same_threshold>=3:
                    last=True
            else:
                same_threshold = 0
                last=False
                
            if i+1==iter_num:
                last=True
                
            self.pinn_train(best_p,count = i+1,coef = self.config_pinn['coef_pde'],
                            same_threshold = same_threshold,
                            last=last)
            
            if last:
                break
        print(f"The No.{iter_num} pde discovery process")
        self.reset_up()    
        return self.callLearn()
        
    def callPINN_var(self):
        """iterative pinn and pde discovery
        """
        self.pretrain()

        prefix, _ = os.path.splitext(self.output_file)

        result = self.callLearn()
        self.output_file = f"{prefix}_1.csv"
        best_p = result['program']
        

        self.denoise_pinn.train_pinn_cv(best_p, coef = self.config_pinn['coef_pde']) 
        Program.reset_task(self.denoise_pinn)
        self.reset_up()    
        result = self.callLearn()
        return result
             
    def train(self):
        # Setup the model
        self.setup()

        #pretrain
        if self.denoise_pinn is not None:
            if self.config_pinn['use_variance']:
                result = self.callPINN_var()
            else:
                result = self.callIterPINN()
            return result
        else:
            # conventional procedures   
            result = {"seed" : self.config_experiment["seed"]} # Seed listed first
            result.update(self.callLearn())
            return result

    def set_config(self, config, pde_config):
        config = load_config(config)
        if  pde_config is not None:
            config = safe_merge_dicts(config, pde_config)

        self.config = defaultdict(dict, config)
        self.config_task = self.config["task"]
        self.config_prior = self.config["prior"]
        self.config_training = self.config["training"]
        self.config_state_manager = self.config["state_manager"]
        self.config_controller = self.config["controller"]
        self.config_gp_meld = self.config["gp_meld"]
        self.config_experiment = self.config["experiment"]
        self.config_pinn = self.config["pinn"]

    def save_config(self):
        # Save the config file
        if self.output_file is not None:
            path = os.path.join(self.config_experiment["save_path"],
                                "config.json")
            # With run.py, config.json may already exist. To avoid race
            # conditions, only record the starting seed. Use a backup seed
            # in case this worker's seed differs.
            backup_seed = self.config_experiment["seed"]
            if not os.path.exists(path):
                if "starting_seed" in self.config_experiment:
                    self.config_experiment["seed"] = self.config_experiment["starting_seed"]
                    del self.config_experiment["starting_seed"]
                with open(path, 'w') as f:
                    json.dump(self.config, f, indent=3)
            self.config_experiment["seed"] = backup_seed

    def set_seeds(self):
        """
        Set the tensorflow, numpy, and random module seeds based on the seed
        specified in config. If there is no seed or it is None, a time-based
        seed is used instead and is written to config.
        """

        seed = self.config_experiment.get("seed")

        # Default uses current time in milliseconds, modulo 1e9
        if seed is None:
            seed = round(time() * 1000) % int(1e9)
            self.config_experiment["seed"] = seed

        # Shift the seed based on task name
        # This ensures a specified seed doesn't have similarities across different task names
        task_name = Program.task.name
        shifted_seed = seed + zlib.adler32(task_name.encode("utf-8"))

        # Set the seeds using the shifted seed
        tf.set_random_seed(shifted_seed)
        np.random.seed(shifted_seed)
        random.seed(shifted_seed)
        torch.random.manual_seed(shifted_seed)
        torch.cuda.manual_seed_all(shifted_seed)

    def make_prior(self):
        prior = make_prior(Program.library, self.config_prior)
        return prior

    def make_state_manager(self):
        return manager_make_state_manager(self.config_state_manager)

    def make_pinn_model(self):
        device = torch.device('cuda:0')
        if not self.config_pinn['use_pinn']:
            model = None
        else:
            model = PINN_model(
                self.output_file,
                self.config_pinn,
                self.config_task['dataset'],
                device
            )
        return model
    
    def make_controller(self):
        # import pdb;pdb.set_trace()
        controller = Controller(self.sess,
                                self.prior,
                                self.state_manager,
                                **self.config_controller)
        return controller

    def make_gp_controller(self):
        if self.config_gp_meld.pop("run_gp_meld", False):
            from dso.gp.gp_controller import GPController
            gp_controller = GPController(self.prior,
                                         self.pool,
                                         **self.config_gp_meld)
        else:
            gp_controller = None
        return gp_controller


    def make_output_file(self):
        """Generates an output filename"""

        # If logdir is not provided (e.g. for pytest), results are not saved
        if self.config_experiment.get("logdir") is None:
            print("WARNING: logdir not provided. Results will not be saved to file.")
            return None

        # When using run.py, timestamp is already generated
        timestamp = self.config_experiment.get("timestamp")
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
            self.config_experiment["timestamp"] = timestamp

        # Generate save path
        task_name = Program.task.name
        save_path = os.path.join(
            self.config_experiment["logdir"],
            '_'.join([task_name, timestamp]))
        self.config_experiment["task_name"] = task_name
        self.config_experiment["save_path"] = save_path
        os.makedirs(save_path, exist_ok=True)

        seed = self.config_experiment["seed"]
        output_file = os.path.join(save_path,
                                   "dso_{}_{}.csv".format(task_name, seed))

        return output_file

    def save(self, save_path):

        saver = tf.train.Saver()
        saver.save(self.sess, save_path)

    def load(self, load_path):

        if self.sess is None:
            self.setup()
        saver = tf.train.Saver()
        saver.restore(self.sess, load_path)