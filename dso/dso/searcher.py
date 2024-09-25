"""Defines main training loop for deep symbolic optimization."""

import os
import time
from itertools import compress,chain
import dill
from multiprocessing import cpu_count, Pool
import tensorflow as tf
import numpy as np
import logging
from tqdm import tqdm


from dso.program import Program, from_tokens,from_str_tokens
from dso.utils import empirical_entropy, get_duration, weighted_quantile, draw_criterion, filter_same,criterion
from dso.ga_utils import drop_duplicates
from dso.memory import Batch, make_queue
from dso.variance import quantile_variance
from dso.train_stats import StatsLogger
from dso.plotter import Plotter
# Ignore TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Searcher:
    def __init__(self, sess,
                 controller,
                 args,
                 gp_aggregator = None,
                 ):
        
        self.plotter = Plotter()
        self.controller = controller
        self.sess = sess
        self.gp_aggregator = gp_aggregator
        self.sess.run(tf.global_variables_initializer())
        self.args = args
        
        self.set_up()
    
    def set_up(self):
        # cached properties
        self.n_iterations = self.args['n_iterations']
        self.batch_size = self.args['n_samples_per_batch']
        self.r_best= -np.inf
        self.prev_r_best = None
        self.best_p = None
        self.pq = make_queue(priority=True, capacity=self.controller.pqt_k)
        self.r_history = []
        self.r_train = []
        self.total_epoch = 0
        self.nevals = 0 # Total number of sampled expressions (from RL or GP)
        self.top_samples_per_batch = list()


    @property
    def best_expression(self):
        return self.pq.get_max()[1]

    @property
    def best_reward(self):
        return self.r_best
    
    def search_one_step(self,epoch = 0, verbose = False):
        # Sample batch of Programs from the Controller
        # Shape of actions: (batch_size, max_length)
        # Shape of obs: [(batch_size, max_length)] * 3
        # Shape of priors: (batch_size, max_length, n_choices)
        self.total_epoch +=1
        start_time = time.time()
        actions, obs, priors, lenghts, finished = self.controller.debug(self.batch_size)
        
        programs = [from_tokens(a) for a in actions]
        
        self.nevals += self.batch_size
        # Run GP seeded with the current batch, returning elite samples

                    
        # Compute rewards (or retrieve cached rewards)
        r = np.array([p.r_ridge for p in programs])
        
        # delete nan
        r[np.isnan(r)]=0.
        r_train = r

        # Need for Vanilla Policy Gradient (epsilon = null)
        p_train     = programs

        l           = np.array([len(p.traversal) for p in programs])
        s           = [p.str for p in programs] # Str representations of Programs
        s_str = [p.str_expression for p in programs] 
        
        on_policy   = np.array([p.originally_on_policy for p in programs])
        invalid     = np.array([p.invalid for p in programs], dtype=bool)


        # Store in variables the values for the whole batch (those variables will be modified below)
        r_full = r
        l_full = l
        s_full = s
        actions_full = actions
        invalid_full = invalid
        quantile = np.min(r)
        valid_full1 = invalid_full ==False
        valid_full2 = r_full >0
        valid_full = np.logical_and(valid_full1, valid_full2)
        r_full_valid = r_full[valid_full]
        r_full = r_full[valid_full2]
        l_full= l_full[valid_full]
        
        r_max = np.max(r_full_valid)
        self.r_best = max(r_max, self.r_best)
       
        """
        Apply risk-seeking policy gradient: compute the empirical quantile of
        rewards and filter out programs with less reward.
        """

        quantile = np.quantile(r, 1 - self.args['epsilon'], interpolation="higher")

        keep        = r >= quantile
        l           = l[keep]
        s           = list(compress(s, keep))
        s_str           = list(compress(s_str, keep))
        invalid     = invalid[keep]
        
        r_train = r         = r[keep]
        p_train = programs  = list(compress(programs, keep))

        '''
            get the action, observation, priors and on_policy status of all programs returned to the controller.
        '''
        actions     = actions[keep, :]
        obs         = obs[keep, :, :]
        priors      = priors[keep, :, :]
        on_policy   = on_policy[keep]
        
        #invalid process
        train_index= np.arange(len(actions))
        invalid_index = train_index[invalid== True]
        valid = invalid ==False
            
        # keep valid expression
        actions     = actions[valid, :]
        obs         = obs[valid, :, :]
        priors      = priors[valid, :, :]
        on_policy   = on_policy[valid]
        r_train = r         = r[valid]
        p_train = programs  = list(compress(programs, valid))
        l           = l[valid]
        s           = list(compress(s, valid))
        s_str           = list(compress(s_str, valid))
        invalid     = invalid[valid]
        
        if len(actions)<1:
            print("no valid training samples")
            # continue when no valid or high-rewards samples
            return None
                
        # Clip bounds of rewards to prevent NaNs in gradient descent
        r       = np.clip(r,        -1e6, 1e6)
        r_train = np.clip(r_train,  -1e6, 1e6)
        r_full_valid = np.clip(r_full_valid, -1e6, 1e6)

        if self.gp_aggregator is not None:   
            p_agg, action_agg, ob_agg, prior_agg = self.gp_aggregator( p_train, actions.shape[1])

            if len(p_agg)>0:        
                self.nevals += self.gp_aggregator.num
                # Combine RNN and deap programs, actions, obs, and priors
                programs = programs + p_agg
                actions = np.append(actions, action_agg, axis=0)
                obs = np.append(obs, ob_agg, axis=0)
                priors = np.append(priors, prior_agg, axis=0)
                
                r = r_train = np.append(r_train, [p_agg[i].r_ridge for i in range(len(p_agg))])
                r_full_valid = np.append(r_full_valid, [p_agg[i].r_ridge for i in range(len(p_agg))])
                p_train = p_train + p_agg
                on_policy = np.append(on_policy, [False]*len(p_agg))
                r_max = np.max(r)
                self.r_best = max(r_max, self.r_best)
        
        self.r_history.append(r_full_valid)
        self.r_train.append(r_train)
        # sort in descending order: larger rewards -> better solutions
        sorted_idx = np.argsort(r)[::-1]
        one_perc = int(len(programs) * float(self.args['epsilon']))
        cur_best_program = programs[sorted_idx[0]]
        for idx in sorted_idx[:one_perc]:
            self.top_samples_per_batch.append([epoch, r[idx], repr(programs[idx])])

        b_train = quantile

        # Compute sequence lengths
        lengths = np.array([min(len(p.traversal), self.controller.max_length)
                            for p in p_train], dtype=np.int32)

        # Create the Batch
        sampled_batch = Batch(actions=actions, obs=obs, priors=priors,
                              lengths=lengths, rewards=r_train, on_policy=on_policy)

        # Update and sample from the priority queue

        # priority_queue.push_best(sampled_batch, programs)
        self.pq.push_batch(sampled_batch, programs)
        pqt_batch = self.pq.sample_batch(self.controller.pqt_batch_size)


        # Train the controller
        summaries = self.controller.train_step(b_train, sampled_batch, pqt_batch)

        # time calculation for the epoch
        epoch_walltime = time.time() - start_time

        # Collect sub-batch statistics and write output
        # logger.save_stats(r_full_valid, r_full, l_full, actions_full, s_full, invalid_full, r,
        #                   l, actions, s, invalid, , r_max, ewma, summaries, epoch,
        #                   s_history, b_train, epoch_walltime, controller_programs)


        # Update new best expression
        new_r_best = False

        if self.prev_r_best is None or r_max > self.prev_r_best:
            new_r_best = True
            self.best_p = cur_best_program
            
        self.prev_r_best = self.r_best
        # Print new best expression
        if verbose and new_r_best:
            print("Training epoch {}/{}, current best R: {:.4f}, cost time: {:.2f}".format( epoch + 1, self.n_iterations, self.prev_r_best,epoch_walltime ))
            print("\n\t** New best")
            self.best_p.print_stats()

        if  (epoch + 1) % 10 == 0:
            print("Training epoch {}/{}, current best R: {:.4f}, cost time: {:.2f}".format( epoch + 1, self.n_iterations, self.r_best, epoch_walltime))

        result_info = {

            'program': cur_best_program,
            'r': cur_best_program.r_ridge,
            "expression": cur_best_program.str_expression

        }
        
        return result_info
    
    def search(self,n_epochs = None, verbose=True):
        if n_epochs == None:
            n_epochs = self.n_iterations
        else:
            self.n_iterations=n_epochs
        pbar = tqdm(range(n_epochs), desc='Progress', ncols=100)
        for i in pbar:

            cur_info = self.search_one_step(epoch=i+1, verbose = verbose)
        
        # Print the priority queue at the end of training
        final_p_list = []

        for i, item in enumerate(self.pq.iter_in_order()):
            # print("\nTop {} result:".format(i))
            # p.print_stats()
            p = Program.cache[item[0]]
            final_p_list.append(p)


            # Return statistics of best Program
            result = {
                "r" : self.best_p.r_ridge,
            }
            # result.update(self.best_p.evaluate)
            result.update({
                "expression" : self.best_p.str_expression,
                "traversal" : repr(self.best_p),
                "program" : self.best_p,
                # "pqt_list":final_p_list
                })

        if verbose:
            print("-- Searching End ------------------")
        return result 

    def print_pq(self):
        for i, item in enumerate(self.pq.iter_in_order()):
            print("\nTop {} result:".format(i))
            p = Program.cache[item[0]]
            p.print_stats()
        
    def plot(self,fig_type, **kwargs):
        if fig_type == 'tree':
            return self.plotter.tree_plot(self.best_p)
        elif fig_type == 'evolution':
            self.plotter.evolution_plot(self.r_train)
        elif fig_type == 'density':
           self.plotter.density_plot(self.r_history, **kwargs)
        else:
            assert False, "not supported figure type"





 