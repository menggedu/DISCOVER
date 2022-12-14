import numpy as np
import pandas as pd
import scipy.io as scio

from dso.task import HierarchicalTask
from dso.library import Library
from dso.functions import create_tokens
from dso.task.pde.utils import *


class PDETask(HierarchicalTask):
    """
    Class for the symbolic regression task. Discrete objects are expressions,
    which are evaluated based on their fitness to a specified dataset.
    """

    task_type = "pde"

    def __init__(self, function_set, dataset, metric="residual",
                 metric_params=(1.0,), extra_metric_test=None,
                 extra_metric_test_params=(), reward_noise=0.0,
                 reward_noise_type="r", threshold=1e-12,
                 normalize_variance=False, protected=False,
                 decision_tree_threshold_set=None):
        """
        Parameters
        ----------
        function_set : list or None
            List of allowable functions. If None, uses function_set according to
            benchmark dataset.

        dataset : dict, str, or tuple
            If dict: .dataset.BenchmarkDataset kwargs.
            If str ending with .csv: filename of dataset.
            If other str: name of benchmark dataset.
            If tuple: (X, y) data

        metric : str
            Name of reward function metric to use.

        metric_params : list
            List of metric-specific parameters.

        extra_metric_test : str
            Name of extra function metric to use for testing.

        extra_metric_test_params : list
            List of metric-specific parameters for extra test metric.

        reward_noise : float
            Noise level to use when computing reward.

        reward_noise_type : "y_hat" or "r"
            "y_hat" : N(0, reward_noise * y_rms_train) is added to y_hat values.
            "r" : N(0, reward_noise) is added to r.

        threshold : float
            Threshold of NMSE on noiseless data used to determine success.

        normalize_variance : bool
            If True and reward_noise_type=="r", reward is multiplied by
            1 / sqrt(1 + 12*reward_noise**2) (We assume r is U[0,1]).

        protected : bool
            Whether to use protected functions.

        decision_tree_threshold_set : list
            A set of constants {tj} for constructing nodes (xi < tj) in decision trees.
        """

        super(HierarchicalTask).__init__()

    
        self.X_test = self.y_test = self.y_test_noiseless = None
        self.name = dataset
        load_class = load_data_2D if '2D' in dataset else load_data

        self.u,self.x,ut,sym_true, n_input_var = load_class(dataset)
        self.ut=ut
        self.sym_true = sym_true
        self.ut = self.ut.reshape(-1,1)
        # Save time by only computing data variances once
        self.var_y_test = np.var(self.ut)
        # import pdb;pdb.set_trace()
        self.var_y_test_noiseless = np.var(self.ut)

        """
        Configure train/test reward metrics.
        """
        self.threshold = threshold
        self.metric, self.invalid_reward, self.max_reward = make_pde_metric(metric, self.ut, *metric_params)
        self.extra_metric_test = extra_metric_test
        if extra_metric_test is not None:
            self.metric_test, _, _ = make_pde_metric(extra_metric_test, self.y_test, *extra_metric_test_params)
        else:
            self.metric_test = None

        """
        Configure reward noise.
        """
        self.reward_noise = reward_noise
        self.reward_noise_type = reward_noise_type
        self.normalize_variance = normalize_variance
        assert reward_noise >= 0.0, "Reward noise must be non-negative."
        if reward_noise > 0:
            assert reward_noise_type in ["y_hat", "r"], "Reward noise type not recognized."
            self.rng = np.random.RandomState(0)
            y_rms_train = np.sqrt(np.mean(self.y_train ** 2))
            if reward_noise_type == "y_hat":
                self.scale = reward_noise * y_rms_train
            elif reward_noise_type == "r":
                self.scale = reward_noise
        else:
            self.rng = None
            self.scale = None

        # Set the Library 
        
        tokens = create_tokens(n_input_var=n_input_var,
                               function_set=function_set,
                               protected=protected,
                               decision_tree_threshold_set=decision_tree_threshold_set,
                               task_type='pde')
        self.library = Library(tokens)

        # Set stochastic flag
        self.stochastic = reward_noise > 0.0

    def reward_function(self, p):

        
        # Compute estimated values
        y_hat = p.execute(self.u, self.x)

        # For invalid expressions, return invalid_reward
        # import  pdb;pdb.set_trace()
        if p.invalid:
            print(p.tokens)
            # import pdb;pdb.set_trace()
            return self.invalid_reward

        # Observation noise
        # For reward_noise_type == "y_hat", success must always be checked to
        # ensure success cases aren't overlooked due to noise. If successful,
        # return max_reward.
        if self.reward_noise and self.reward_noise_type == "y_hat":
            if p.evaluate.get("success"):
                return self.max_reward
            y_hat += self.rng.normal(loc=0, scale=self.scale, size=y_hat.shape)

        # Compute metric
        r = self.metric(self.ut, y_hat)

        # Direct reward noise
        # For reward_noise_type == "r", success can for ~max_reward metrics be
        # confirmed before adding noise. If successful, must return np.inf to
        # avoid overlooking success cases.
        if self.reward_noise and self.reward_noise_type == "r":
            if r >= self.max_reward - 1e-5 and p.evaluate.get("success"):
                return np.inf
            r += self.rng.normal(loc=0, scale=self.scale)
            if self.normalize_variance:
                r /= np.sqrt(1 + 12 * self.scale ** 2)

        return r

    def reward_functionSTR(self,p):
        y_hat, w = p.execute_STR(self.u, self.x, self.ut)
        n = len(w)
        if p.invalid:
            # print(p.tokens)
            # import pdb;pdb.set_trace()
            return self.invalid_reward, [0]

        # Compute metric
        r = self.metric(self.ut, y_hat,n)

        return r, w

    def evaluate(self, p):

        # Compute predictions on test data
        y_hat, w = p.execute_STR(self.u, self.x, self.ut)
        n = len(w)
        # y_hat = p.execute(self.X_test)
        if p.invalid:
            nmse_test = None
            nmse_test_noiseless = None
            success = False

        else:
            # NMSE on test data (used to report final error)
            nmse_test = np.mean((self.ut - y_hat) ** 2)

            # NMSE on noiseless test data (used to determine recovery)
            # nmse_test_noiseless = np.mean((self.y_test_noiseless - y_hat) ** 2) / self.var_y_test_noiseless

            # Success is defined by NMSE on noiseless test data below a threshold
            success = nmse_test < self.threshold

        info = {
            "nmse_test" : nmse_test,
            "success" : success,

        }

        if self.metric_test is not None:
            if p.invalid:
                m_test = None
                m_test_noiseless = None
            else:
                m_test = self.metric_test(self.y_test, y_hat,n)
                m_test_noiseless = self.metric_test(self.y_test_noiseless, y_hat,n)

            info.update({
                self.extra_metric_test : m_test,
                self.extra_metric_test + '_noiseless' : m_test_noiseless
            })

        return info


def make_pde_metric(name, ut, *args):
    """
    Factory function for a regression metric. This includes a closures for
    metric parameters and the variance of the training data.

    Parameters
    ----------

    name : str
        Name of metric. See all_metrics for supported metrics.

    args : args
        Metric-specific parameters

    Returns
    -------

    metric : function
        Regression metric mapping true and estimated values to a scalar.

    invalid_reward: float or None
        Reward value to use for invalid expression. If None, the training
        algorithm must handle it, e.g. by rejecting the sample.

    max_reward: float
        Maximum possible reward under this metric.
    """

    var_y = np.var(ut)
    
    all_metrics = {

        "inv_nrmse" :    (lambda y, y_hat : 1/(1 + args[0]*np.sqrt(np.mean((y - y_hat)**2)/var_y)),
                        1),

        "pde_reward":  (lambda y, y_hat,n : (1-0.01*n)/(1 + args[0]*np.sqrt(np.mean((y - y_hat)**2)/var_y)),
                        1),

      
    }

    assert name in all_metrics, "Unrecognized reward function name."
    assert len(args) == all_metrics[name][1], "For {}, expected {} reward function parameters; received {}.".format(name,all_metrics[name][1], len(args))
    metric = all_metrics[name][0]

    # For negative MSE-based rewards, invalid reward is the value of the reward function when y_hat = mean(y)
    # For inverse MSE-based rewards, invalid reward is 0.0
    # For non-MSE-based rewards, invalid reward is the minimum value of the reward function's range
    all_invalid_rewards = {
       
        "inv_nrmse" : 0.0, #1/(1 + args[0]),
        'pde_reward':0.0

    }
    invalid_reward = all_invalid_rewards[name]

    all_max_rewards = {
        "inv_nrmse" : 1.0,
        "pde_reward":1.0

    }
    max_reward = all_max_rewards[name]

    return metric, invalid_reward, max_reward

