
"""Core deep PDE symbolic optimizer construct."""

import warnings
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
from scipy.stats import pearsonr

from dso.task import set_task
from dso.controller import Controller
from dso.train import learn
from dso.prior import make_prior
from dso.program import Program,from_str_tokens,from_tokens
from dso.config import load_config
from dso.tf_state_manager import make_state_manager as manager_make_state_manager
from dso.core import DeepSymbolicOptimizer
from dso.pinn import PINN_model
from dso.utils import safe_merge_dicts
warnings.filterwarnings('ignore', category=FutureWarning)


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
        _, file_name = os.path.split(config)
        self.job_name = file_name.split('.j')[0]
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
        self.gp_aggregator = self.make_gp_aggregator()
        
        
    def reset_up(self,clear_cache = True,reset_controller=True,new_seed=None):
        # Clear the cache and reset the compute graph
        if clear_cache:
            Program.clear_cache()
        
        tf.reset_default_graph()

        if new_seed is not None:
            self.set_seeds(new_seed) # Must be called _after_ resetting graph and _after_ setting task
            
        self.sess = tf.Session()
        if reset_controller:
            self.controller = self.make_controller()
        self.gp_controller = self.make_gp_controller()
        
    def pretrain(self):
        # pretrain for pinn with only obeservations
        self.denoise_pinn.pretrain()
        Program.reset_task(self.denoise_pinn, self.config_pinn['generation_type'])
        
        
    def pinn_train(self, best_p,count,coef=0.1,local_sample=False,last=False):
        """
        emedding process with discovered equation constraint
        """
        # sym = "add_t,mul_t,u1,u1,add_t,diff_t,u1,x1,diff3_t,u1,x1"
        # best_p = from_str_tokens(sym)
        # r = best_p.r_ridge
        # print(f"reward is {r}")
        self.denoise_pinn.train_pinn(best_p,count,coef=coef,local_sample=local_sample,last=last)
        Program.reset_task(self.denoise_pinn, self.config_pinn['generation_type'])
        
    def callLearn(self,eq_num = 1):
        """
        discovering process
        """
        if eq_num == 1:
            result = learn(self.sess,
                        self.controller,
                        self.pool,
                        self.gp_controller,
                        self.gp_aggregator,
                        self.denoise_pinn,
                        self.output_file,
                        self.best_p[0],
                        **self.config_training) #early_stop  
            return [result]
        else:
            #multi_eq
            func_repre = ['u','v','q']
            file = self.output_file[:-4]
            results = []
            for i in range(eq_num):
                Program.task.reset_ut(i)
                self.output_file = file+f"_{func_repre[i]}"+".csv"
                result = learn(self.sess,
                            self.controller,
                            self.pool,
                            self.gp_controller,
                            self.gp_aggregator,
                            self.denoise_pinn,
                            self.output_file,
                            self.best_p[i],
                            **self.config_training) #early_stop  
                
                results.append(result)
                self.reset_up(clear_cache=False)

        return results
    

    def callIterPINN(self):
        """iterative pinn and pde discovery
        """
        self.pretrain()
        # self.pinn_train(None,count = 1,coef = self.config_pinn['coef_pde'],
        #                 local_sample = self.config_pinn['local_sample'],
        #                 last=True)
        #  import pdb;pdb.set_trace()
        last=False    
        last_best_p = None
        best_tokens= []
        prefix, _ = os.path.splitext(self.output_file)
        iter_num = self.config_pinn["iter_num"]
        eq_num = self.config_task.get('eq_num',1)
        
        for i in range(iter_num):
            if i>0:
                self.reset_up(reset_controller=False)
                # change search epoch
                bsz = self.config_training['batch_size']
                self.config_training['n_samples'] = 10*bsz

            print(f"The No.{i} pde discovery process")   
            results = self.callLearn(eq_num)
            self.output_file = f"{prefix}_{i+1}.csv"
            best_p = [results[j]['program'] for j in range(len(results))]

            if len(best_tokens)>0:
                # keep the best expressions from last iteration
                new_best_p = []
                last_best_p = [from_tokens(best_tokens[t]) for t in range(len(results))]
                # assert len(last_best_p)==len(best_p)
                for j in range(len(results)):
                    if last_best_p[j].r_ridge>best_p[j].r_ridge:
                        new_best_p.append(last_best_p[j])
                    else:
                        new_best_p.append(best_p[j])
                best_p = new_best_p
               
            if i+1==iter_num:
                last=True
            
            self.pinn_train(best_p,count = i+1,coef = self.config_pinn['coef_pde'],
                            local_sample = self.config_pinn['local_sample'],
                            last=last)
            
            best_tokens = [best_p[j].tokens  for j in range(len(results))]
            self.best_p = best_p


        print(f"The No.{iter_num} pde discovery process")
        self.reset_up(reset_controller=False)    
        return self.callLearn(eq_num)
        
    def callPINN_var(self):
        """iterative pinn and pde discovery with variable coef
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

        # initiate the best expressions
        eq_num = self.config_task['eq_num'] 
        self.best_p = [None for _ in range(eq_num)]

        if self.denoise_pinn is not None:
            if self.config_pinn['use_variance']:
                result = self.callPINN_var()
            else:
                result = self.callIterPINN()
            return result
        else: 
            if self.config_param['on']:
                self.residual_training()
            else:
                # conventional procedures  
                
                result = {"seed" : self.config_experiment["seed"]} # Seed listed first
                result.update(*self.callLearn(eq_num))
                return result
        
    def residual_training(self):
        reset = True
        for i in range(self.config_param['iter']):
            print(f"***********The {i}th iteration*********** ")
            result = self.callLearn()
            p = result["program"]

            # Program.clear_cache()
            if reset: 
                self.reset_up(new_seed=i)
            prefix, _ = os.path.splitext(self.output_file)
            self.output_file = f"{prefix}_{i}.csv"
            
            # save_result
            
            reset = Program.task.process_results(p)
            
        terms,_, w_best = Program.task.sum_results()
        print([repr(t) for t in terms])
        print(w_best)
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
        self.config_param = self.config['parameterized']
        self.config_gp_agg = self.config["gp_agg"]

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

    def set_seeds(self, new_seed=None):
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
        if new_seed is not None:
            shifted_seed+=new_seed
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

    def make_gp_aggregator(self):
        if self.config_gp_agg.pop("run_gp_agg", False):
            from dso.aggregate import gpAggregator
            gp_aggregator = gpAggregator(self.prior,
                                         self.pool,
                                         self.config_gp_agg)
        else:
            gp_aggregator = None
        return gp_aggregator
            
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
            '_'.join([self.job_name, timestamp]))
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