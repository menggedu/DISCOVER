"""Performs computations and file manipulations for train statistics logging purposes"""
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
import pandas as pd
from dso.program import Program, from_tokens
from dso.utils import is_pareto_efficient, empirical_entropy
from itertools import compress
from io import StringIO
import shutil
from collections import defaultdict

#These functions are defined globally so they are pickleable and can be used by Pool.map
def hof_work(p):
    return [p.r_ridge, p.on_policy_count, p.off_policy_count, p.str_expression, repr(p), p.evaluate]

def pf_work(p):
    return [p.complexity, p.r_ridge, p.on_policy_count, p.off_policy_count, p.str_expression, repr(p), p.evaluate]


class StatsLogger():
    """ Class responsible for dealing with output files of training statistics. It encapsulates all outputs to files."""

    def __init__(self, sess, output_file, save_summary=False, save_all_epoch=False, hof=100,
                 save_pareto_front=True, save_positional_entropy=False, save_top_samples_per_batch=0,
                 save_cache=False, save_cache_r_min=0.9, save_freq=1, save_token_count=False,
                 save_rewards = False, save_all_rewards = False
                 ):

        """"
        sess : tf.Session
            TenorFlow Session object (used for generating summary files)

        output_file : str
            Filename to write results for each iteration.

        save_summary : bool, optional
            Whether to write TensorFlow summaries.

        save_all_epoch : bool, optional
            Whether to save statistics for all programs for each iteration.

        hof : int or None, optional
            Number of top Programs to evaluate after training.

        save_pareto_front : bool, optional
            If True, compute and save the Pareto front at the end of training.

        save_positional_entropy : bool, optional
            Whether to save evolution of positional entropy for each iteration.

        save_top_samples_per_batch : float, optional
            Whether to store the X% (float) top-performer samples in every batch to a csv file.

        save_cache : bool
            Whether to save the str, count, and r of each Program in the cache.

        save_cache_r_min : float or None
            If not None, only keep Programs with r >= r_min when saving cache.

        save_freq : int or None
            Statistics are flushed to file every save_freq epochs (default == 1). If < 0, uses save_freq = inf

        save_token_count : bool
            Wether to count used tokens in each epoch
        """
        self.sess = sess
        self.output_file = output_file
        self.save_summary = save_summary
        self.save_all_epoch = save_all_epoch
        self.hof = hof
        self.save_pareto_front = save_pareto_front
        self.save_positional_entropy = save_positional_entropy
        self.save_top_samples_per_batch = save_top_samples_per_batch
        self.save_cache = save_cache
        self.save_cache_r_min = save_cache_r_min
        self.save_token_count = save_token_count
        self.save_rewards = save_rewards
        self.save_all_rewards = save_all_rewards
        # self.save_function_terms = save_function_terms
        self.all_r = []   # save all R separately to keep backward compatibility with a generated file.
        self.all_r_valid = []
        
        if save_freq is None:
            self.buffer_frequency = 1
        elif save_freq < 1:
            self.buffer_frequency = float('inf')
        else:
            self.buffer_frequency = save_freq

        self.buffer_epoch_stats = StringIO()  #Buffer for epoch statistics
        self.buffer_all_programs = StringIO()  #Buffer for the statistics for all programs.
        self.buffer_token_stats = StringIO()  #Buffer for epoch statistics

        self.setup_output_files()

    def setup_output_files(self):
        """
        Opens and prepares all output log files controlled by this class.
        """
        if self.output_file is not None:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            prefix, _ = os.path.splitext(self.output_file)
            self.all_r_output_file = "{}_all_r.csv".format(prefix)
            self.all_r_output_valid_file = "{}_all_valid_r.csv".format(prefix)
            self.all_info_output_file = "{}_all_info.csv".format(prefix)
            self.hof_output_file = "{}_hof.csv".format(prefix)
            self.pf_output_file = "{}_pf.csv".format(prefix)
            self.positional_entropy_output_file = "{}_positional_entropy.npy".format(prefix)
            self.top_samples_per_batch_output_file = "{}_top_samples_per_batch.csv".format(prefix)
            self.cache_output_file = "{}_cache.csv".format(prefix)
            self.token_counter_output_file = "{}_token_count.csv".format(prefix)
            self.save_function_term_out_file = "{}_function_terms_per_batch.csv".format(prefix)
            with open(self.output_file, 'w') as f:
                # r_best : Maximum across all iterations so far
                # r_max : Maximum across this iteration's batch
                # r_avg_full : Average across this iteration's full batch (before taking epsilon subset)
                # r_avg_sub : Average across this iteration's epsilon-subset batch
                # n_unique_* : Number of unique Programs in batch
                # n_novel_* : Number of never-before-seen Programs per batch
                # a_ent_* : Empirical positional entropy across sequences averaged over positions
                # invalid_avg_* : Fraction of invalid Programs per batch
                # baseline: Baseline value used for training
                # time: time used to learn in this epoch (in seconds)
                headers = ["r_best",
                           "r_max",
                           "r_avg_full",
                           "r_avg_sub",
                           "l_avg_full",
                           "l_avg_sub",
                           "ewma",
                           "n_unique_full",
                           "n_unique_sub",
                           "n_novel_full",
                           "n_novel_sub",
                           "a_ent_full",
                           "a_ent_sub",
                           "invalid_avg_full",
                           "invalid_avg_sub",
                           "baseline",
                           "time"]
                f.write("{}\n".format(",".join(headers)))
            if self.save_all_epoch:
                with open(self.all_info_output_file, 'w') as f:
                    # epoch : The epoch in which this line was saved
                    # r : reward for this program
                    # l : length of the program
                    # invalid : if the program is invalid
                    headers = ["epoch",
                                "r",
                                "l",
                                ]
                    f.write("{}\n".format(",".join(headers)))
            if self.save_token_count:
                with open(self.token_counter_output_file, 'w') as f:
                    headers = [str(token) for token in Program.library.tokens]
                    f.write("{}\n".format(",".join(headers)))

        else:
            self.all_r_output_file = None
            self.all_info_output_file = None
            self.hof_output_file = None
            self.pf_output_file = None
            self.positional_entropy_output_file = None
            self.top_samples_per_batch_output_file = None
            self.cache_output_file = None
            self.token_counter_output_file = None

        # Create summary writer
        if self.save_summary:
            if self.output_file is not None:
                summary_dir = "{}_summary".format(prefix)
            else:
                timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
                summary_dir = os.path.join("summary", timestamp)
            self.summary_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)
        else:
            self.summary_writer = None

    def save_stats(self, r_full_valid, r_full, l_full, actions_full, s_full, invalid_full, r, l,
                   actions, s, invalid, r_best, r_max, ewma, summaries, epoch, s_history,
                   baseline, epoch_walltime, programs):
        """
        Computes and saves all statistics that are computed for every time step. Depending on the value of
            self.buffer_frequency, the statistics might be instead saved in a buffer before going to disk.
        :param r_full: The reward of all programs
        :param l_full: The length of all programs
        :param actions_full: all actions sampled this step
        :param s_full: String representation of all programs sampled this step.
        :param invalid_full: boolean for all programs sampled showing if they are invalid
        :param r: r_full excluding the ones where keep=false
        :param l: l_full excluding the ones where keep=false
        :param actions: actions_full excluding the ones where keep=false
        :param s: s_full excluding the ones where keep=false
        :param invalid: invalid_full excluding the ones where keep=false
        :param r_best: reward from the all time best program so far
        :param r_max: reward from the best program in this epoch
        :param ewma: Exponentially Weighted Moving Average weight that might be used for baseline computation
        :param summaries: Sumarries returned by the Controller this step
        :param epoch: This epoch id
        :param s_history: all programs ever seen in string format.
        :param baseline: baseline value used for training
        :param epoch_walltime: time taken to process this epoch
        :param programs: A batch of controller programs
        """
        epoch = epoch + 1 #changing from 0-based index to 1-based
        if self.output_file is not None:
            r_avg_full = np.mean(r_full)

            l_avg_full = np.mean(l_full)
            a_ent_full = np.mean(np.apply_along_axis(empirical_entropy, 0, actions_full))
            n_unique_full = len(set(s_full))
            n_novel_full = len(set(s_full).difference(s_history))
            invalid_avg_full = np.mean(invalid_full)

            r_avg_sub = np.mean(r)
            l_avg_sub = np.mean(l)
            a_ent_sub = np.mean(np.apply_along_axis(empirical_entropy, 0, actions))
            n_unique_sub = len(set(s))
            n_novel_sub = len(set(s).difference(s_history))
            invalid_avg_sub = np.mean(invalid)
            stats = np.array([[
                r_best,
                r_max,
                r_avg_full,
                r_avg_sub,
                l_avg_full,
                l_avg_sub,
                ewma,
                n_unique_full,
                n_unique_sub,
                n_novel_full,
                n_novel_sub,
                a_ent_full,
                a_ent_sub,
                invalid_avg_full,
                invalid_avg_sub,
                baseline,
                epoch_walltime
            ]], dtype=np.float32)
            np.savetxt(self.buffer_epoch_stats, stats, delimiter=',')
        if self.save_all_epoch:
            all_epoch_stats = np.array([
                              [epoch]*len(r_full),
                              r_full,
                              l_full,
                              ]).transpose()
            df = pd.DataFrame(all_epoch_stats)
            df.to_csv(self.buffer_all_programs, mode='a', header=False, index=False, line_terminator='\n')

        # Collect stats about used tokens and write to buffer
        if self.save_token_count:
            self.write_token_count(programs)

        # summary writers have their own buffer
        if self.save_summary:
            self.summary_writer.add_summary(summaries, epoch)

        # Should the buffer be saved now?
        if epoch % self.buffer_frequency == 0:
            self.flush_buffers()

        #Backwards compatibility of all_r numpy file
        if self.save_all_rewards:
            self.all_r.append(r_full)
            
        if self.save_rewards:
            self.all_r_valid.append(r_full_valid)
        
    def save_results(self, positional_entropy, top_samples_per_batch,funcion_info_per_batch, r_history, pool, n_epochs, n_samples):
        """
        Saves stats that are available only after all epochs are finished
        :param positional_entropy: evolution of positional_entropy for all epochs
        :param top_samples_per_batch: list containing top solutions on every batch
        :param r_history: reward for each unique program found during training
        :param pool: Pool used to parallelize reward computation
        :param n_epochs: index of last epoch
        :param n_samples: Total number of samples
        """
        n_epochs = n_epochs + 1
        # First of all, saves any pending buffer
        self.flush_buffers()

        if self.save_all_epoch:
            # Kept all_r numpy file for backwards compatibility.
            with open(self.all_r_output_file, 'ab') as f:
                all_r = np.array(self.all_r, dtype=np.float32)
                np.save(f, all_r)
        if self.save_rewards:
            all_r_valid = pd.DataFrame(self.all_r_valid)                            
            all_r_valid.to_csv(self.all_r_output_valid_file)  
        if self.save_all_rewards:     
            all_r = pd.DataFrame(self.all_r)                            
            all_r.to_csv(self.all_r_output_file)
        
        if self.save_positional_entropy:
            with open(self.positional_entropy_output_file, 'ab') as f:
                np.save(f, positional_entropy)

        if self.save_top_samples_per_batch > 0:
            df_topsamples = pd.DataFrame(top_samples_per_batch,
                                         columns=['Epoch', 'Reward', 'Sequence'])
            df_topsamples.to_csv(self.top_samples_per_batch_output_file)

            function_samples = pd.DataFrame(funcion_info_per_batch, 
                                            columns = ['Epoch', 'Functions', 'Reward', 'coefficients'])
            function_samples.to_csv(self.save_function_term_out_file)
        # Save the hall of fame
        if self.hof is not None and self.hof > 0:
            # For stochastic Tasks, average each unique Program's r_history,
            if Program.task.stochastic:

                # Define a helper function to generate a Program from its tostring() value
                def from_token_string(str_tokens):
                    tokens = np.fromstring(str_tokens, dtype=np.int32)
                    return from_tokens(tokens)

                # Generate each unique Program and manually set its reward to the average of its r_history
                keys = r_history.keys()  # str_tokens for each unique Program
                vals = r_history.values()  # reward histories for each unique Program
                programs = [from_token_string(str_tokens) for str_tokens in keys]
                for p, r in zip(programs, vals):
                    p.r_ridge = np.mean(r)
                    #It is not possible to tell if each program was sampled on- or off-policy at this point.
                    # -1 on off_policy_count signals that we can't distinguish the counters in this task.
                    p.on_policy_count = len(r)
                    p.off_policy_count = -1

            # For deterministic Programs, just use the cache
            else:
                programs = list(Program.cache.values())  # All unique Programs found during training

            r = [p.r_ridge for p in programs]
            i_hof = np.argsort(r)[-self.hof:][::-1]  # Indices of top hof Programs
            hof = [programs[i] for i in i_hof]

            if pool is not None:
                results = pool.map(hof_work, hof)
            else:
                results = list(map(hof_work, hof))

            eval_keys = list(results[0][-1].keys())
            columns = ["r", "count_on_policy", "count_off_policy", "expression", "traversal"] + eval_keys
            hof_results = [result[:-1] + [result[-1][k] for k in eval_keys] for result in results]
            df = pd.DataFrame(hof_results, columns=columns)
            if self.hof_output_file is not None:
                print("Saving Hall of Fame to {}".format(self.hof_output_file))
                df.to_csv(self.hof_output_file, header=True, index=False)

            #save cache
            if self.save_cache and Program.cache:
                print("Saving cache to {}".format(self.cache_output_file))
                cache_data = [(repr(p), p.on_policy_count, p.off_policy_count, p.r_ridge) for p in Program.cache.values()]
                df_cache = pd.DataFrame(cache_data)
                df_cache.columns = ["str", "count_on_policy", "count_off_policy", "r"]
                if self.save_cache_r_min is not None:
                    df_cache = df_cache[df_cache["r"] >= self.save_cache_r_min]
                df_cache.to_csv(self.cache_output_file, header=True, index=False)

            # Compute the pareto front
            if self.save_pareto_front:
                all_programs = list(Program.cache.values())
                costs = np.array([(p.complexity, -p.r_ridge) for p in all_programs])
                pareto_efficient_mask = is_pareto_efficient(costs)  # List of bool
                pf = list(compress(all_programs, pareto_efficient_mask))
                pf.sort(key=lambda p: p.complexity)  # Sort by complexity

                if pool is not None:
                    results = pool.map(pf_work, pf)
                else:
                    results = list(map(pf_work, pf))

                eval_keys = list(results[0][-1].keys())
                columns = ["complexity", "r", "count_on_policy", "count_off_policy", "expression", "traversal"] + eval_keys
                pf_results = [result[:-1] + [result[-1][k] for k in eval_keys] for result in results]
                df = pd.DataFrame(pf_results, columns=columns)
                if self.pf_output_file is not None:
                    print("Saving Pareto Front to {}".format(self.pf_output_file))
                    df.to_csv(self.pf_output_file, header=True, index=False)

                # Look for a success=True case within the Pareto front
                for p in pf:
                    if p.evaluate.get("success"):
                        p_final = p
                        break
                    
            # Save error summaries
            # Print error statistics of the cache
            n_invalid = 0
            error_types = defaultdict(lambda: 0)
            error_nodes = defaultdict(lambda: 0)

            result = {}
            for p in Program.cache.values():
                if p.invalid:
                    count = p.off_policy_count + p.on_policy_count
                    n_invalid += count
                    error_types[p.error_type] += count
                    error_nodes[p.error_node] += count

            if n_invalid > 0:
                print("Invalid expressions: {} of {} ({:.1%}).".format(n_invalid, n_samples,
                                                                       n_invalid / n_samples))
                print("Error type counts:")
                for error_type, count in error_types.items():
                    print("  {}: {} ({:.1%})".format(error_type, count, count / n_invalid))
                    result["error_"+str(error_type)] = count
                print("Error node counts:")
                for error_node, count in error_nodes.items():
                    print("  {}: {} ({:.1%})".format(error_node, count, count / n_invalid))
                    result["error_node_" + str(error_node)] = count

            result['n_epochs'] = n_epochs
            result['n_samples'] = n_samples
            result['n_cached'] = len(Program.cache)
            return result

    def write_token_count(self, programs):
        token_counter = {token: 0 for token in Program.library.names}
        for program in programs:
            for token in program.traversal:
                token_counter[token.name] += 1
        stats = np.array([[
            token_counter[token] for token in token_counter.keys()
        ]], dtype=np.int)
        np.savetxt(self.buffer_token_stats, stats, fmt='%i', delimiter=',')
        
    def save_function_terms(self,functions):
        df_functions = pd.DataFrame(functions,
                                        columns=['Epoch', 'terms', 'rewards'])
        df_functions.to_csv(self.top_samples_per_batch_output_file)
        
        function_id = {}
        start_id = 0
        function_locs = []
        for function_terms in functions:
            function_loc = []
            for i in range(len(function_terms)):
                function = function_terms[i]
                if function not in function_id:
                    function_id[function] = start_id
                    start_id+=1
                function_loc.append(function_id[function])
                
            function_locs.append(function_loc)
        function_arrays = np.zeros((len(function_terms),start_id) )
        for i in range(len(functions)):
            for loc in function_locs[i]:
                function_arrays[i, loc]=1
        np.save('best_function_arrary.npy',function_arrays )    
        
    def flush_buffers(self):
        """Write all available buffers to file."""
        if self.output_file is not None:
            self.buffer_epoch_stats = self.flush_buffer(
                self.buffer_epoch_stats, self.output_file)
        if self.save_all_epoch:
            self.buffer_all_programs = self.flush_buffer(
                self.buffer_all_programs, self.all_info_output_file)
        if self.save_token_count:
            self.buffer_token_stats = self.flush_buffer(
                self.buffer_token_stats, self.token_counter_output_file)
        if self.summary_writer:
            self.summary_writer.flush()

    def flush_buffer(self, buffer, output_file):
        """
        Write specific buffer to corresponding output file.
        :param buffer: Buffer that will be flushed
        :param output_file: File to which the buffer will be flushed
        """
        with open(output_file, 'a') as f:
            buffer.seek(0)
            shutil.copyfileobj(buffer, f, -1)
        # clear buffer
        return StringIO()