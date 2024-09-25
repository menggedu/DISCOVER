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
from dso.searcher import Searcher

warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class SymEqOptimizer():
    """

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

    def __init__(self,
        n_iterations= 100,
        n_samples_per_batch = 100,
        binary_operators = [],
        unary_operators = [],
        out_path = '',
        core_num = 1,
        dataset=None,
        noise=False,
        config_out=None,
        seed=0,
        ):
        """
        
        
        """

        self.n_iterations = n_iterations
        self.n_samples_per_batch= n_samples_per_batch
        self.operator = binary_operators+unary_operators
        self.out_path = out_path
        self.dataset = dataset
        self.seed = seed
        self.core_num = core_num
        self.set_config(config_out)
        # Setup the model
        self.setup()
        

    def setup(self, ):
        # Clear the cache and reset the compute graph
        Program.clear_cache()
        tf.reset_default_graph()

        # Generate objects needed for training and set seeds
        self.pool = self.make_pool_and_set_task()
        self.sess = tf.Session()

        # Save complete configuration file
        if self.out_path is not None:
            self.output_file = os.path.join(self.out_path,
                                   "dso_{}_{}.csv".format(self.dataset, self.seed))

        # Prepare training parameters
        self.prior = self.make_prior()
        self.state_manager = self.make_state_manager()
        self.controller = self.make_controller()
        self.gp_aggregator = self.make_gp_aggregator()
        self.pool = self.make_pool_and_set_task()
        self.searcher = self.make_searcher()
        
    
    def info(self,):
        print("Library: ", Program.task.library)
        self.controller.prior.report_constraint_counts()
        

    def fit(x,y):
        pass


    def train_one_step(self, epoch = 0, verbose = True):
        return self.searcher.search_one_step(epoch = epoch, verbose =verbose)


    def train(self, n_epochs = 100, verbose = True):
        """ full training procedure"""

        return self.searcher.search(n_epochs = n_epochs, verbose= verbose)

        
    def set_config(self, config):
        if config is not None:
            config = load_config(config)

        base_config_file = "config/config_pde.json"
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), base_config_file), encoding='utf-8') as f:    
            base_config = json.load(f)
        config_update = safe_merge_dicts(base_config, config)
        self.config = defaultdict(dict, config_update)
        self.config_task = self.config["task"]
        self.config_prior = self.config["prior"]
        self.config_state_manager = self.config["state_manager"]
        self.config_controller = self.config["controller"]
        self.config_gp_agg = self.config["gp_agg"]
        self.config_training = self.config["training"]

    def make_prior(self):
        prior = make_prior(Program.library, self.config_prior)
        return prior

    def make_state_manager(self):
        return manager_make_state_manager(self.config_state_manager)

    def make_controller(self):
        controller = Controller(self.sess,
                                self.prior,
                                self.state_manager,
                                **self.config_controller)
        return controller


    def make_gp_aggregator(self):
        if self.config_gp_agg.pop("run_gp_agg", False):
            from dso.aggregate import gpAggregator
            gp_aggregator = gpAggregator(self.prior,
                                         self.pool,
                                         self.config_gp_agg)
        else:
            gp_aggregator = None
        return gp_aggregator
    
    def make_searcher(self):
        self.config_training['n_iterations'] = self.n_iterations
        self.config_training['n_samples_per_batch'] = self.n_samples_per_batch

        searcher = Searcher(
                            sess = self.sess,
                            controller=self.controller,
                            args = self.config_training,
                            gp_aggregator=self.gp_aggregator
                            )
        return searcher

    def save(self, save_path):
        saver = tf.train.Saver()
        saver.save(self.sess, save_path)

    def load(self, load_path):

        if self.sess is None:
            self.setup()
        saver = tf.train.Saver()
        saver.restore(self.sess, load_path)

    def make_pool_and_set_task(self):
        # Create the pool and set the Task for each worker
        
        # Set the constant optimizer
        const_optimizer = self.config_training["const_optimizer"]
        const_params = self.config_training["const_params"]
        const_params = const_params if const_params is not None else {}
        Program.set_const_optimizer(const_optimizer, **const_params)

        pool = None
        n_cores_batch = self.config_training.get("n_cores_batch")
        
        # Set the Task for the parent process
        if self.dataset is not None:
            self.config_task['dataset'] = self.dataset
        if len(self.operator)>0:
            self.config_task['function_set'] = self.operator
        set_task(self.config_task)

        return pool
    
    
    def print_pq(self):
        self.searcher.print_pq()

    def plot(self, fig_type, **kwargs):
        return self.searcher.plot(fig_type, **kwargs)

class Deep_SymEqOptimizer(SymEqOptimizer):


    # def __init__(self):
    #     pass

    
    def set_config(self,config):
        super().set_config(config)
        self.config_pinn = self.config["pinn"]

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
    
    def set_up(self):
        super().setup()

        self.denoise_pinn = self.make_pinn_model()


    def fit():
        pass

