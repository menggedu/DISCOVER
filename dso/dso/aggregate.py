import numpy as np
import time
import random
import copy
from itertools import compress

from dso.subroutines import jit_parents_siblings_at_once
from dso.program import Program, from_tokens
from dso.ga_utils import TrainSTRidge, tokens2index,mutate_single,drop_duplicates,cross_over_p,evaluate_terms,spatial_complete

class gpAggregator:
    def __init__(self, prior, pool, config):
        
        self.prior = prior
        self.pool = pool
        self.gp_config = config['gp']
        self.STR_config = config['STRidge']
        self.expressions = [] # token list [ [token1, token2, token3]...]
        self.num=0
        self.l0_penalty = self.STR_config.get("l0_penalty", 0.001)
        self.l0_coef = self.STR_config.get("l0_coef", 0.1)
        self.max_terms = self.STR_config.get("max_terms", 10)
        # keep best
        self.best_r = 0
        self.best_p = None
        self.best_tokens = []
        self.best_values = []

    def convert_programs(self,terms_list):
        programs = []
        for terms in terms_list:
            index = tokens2index(terms,self.max_length)
            programs.append(from_tokens(index, on_policy=False))
        return programs
     
    def get_agent_programs(self):
        hof = self.expressions
        L = Program.library.L

        # Init all as numpy arrays of integers with default values or empty
        actions = np.empty((len(hof), self.max_length), dtype=np.int32)
        obs_action = np.empty((len(hof), self.max_length), dtype=np.int32)
        obs_parent = np.zeros((len(hof), self.max_length), dtype=np.int32)
        obs_sibling = np.zeros((len(hof), self.max_length), dtype=np.int32)
        obs_dangling = np.ones((len(hof), self.max_length), dtype=np.int32)

        obs_action[:, 0] = L 
        programs = []

        # Compute actions, obs, and programs
        for i, ind in enumerate(hof):
            index = tokens2index(ind, self.max_length)
            programs.append(from_tokens(index, on_policy=False))
            
            actions[i, :] = index
            obs_action[i, 1:] = index[:-1]
            obs_parent[i, :], obs_sibling[i, :] = jit_parents_siblings_at_once(np.expand_dims(index, axis=0),
                                                                               arities=Program.library.arities,
                                                                               parent_adjust=Program.library.parent_adjust)  
            arities = np.array([Program.library.arities[t] for t in index])
            obs_dangling[i, :] = 1 + np.cumsum(arities - 1)

        # Compute priors   
        priors = self.prior.at_once(actions, obs_parent, obs_sibling)
        obs = np.stack([obs_action, obs_parent, obs_sibling, obs_dangling], axis=1)
        
        self.expressions = [] # clean out cache
        return programs, actions, obs, priors
    
    def cross_over(self, programs):
        p_cv = self.gp_config.get("p_cross_over", 0)
        length = len(programs)
        programs2 = copy.deepcopy(programs)
        new_terms_list = []
        random.shuffle(programs2)
        for p1, p2 in zip(programs, programs2):
            terms1, terms2 = cross_over_p(p1,p2)
            new_terms_list.append(terms1)
            new_terms_list.append(terms2)
        new_terms_cro = [self.reconstruct_tokens(new_terms_list[i]) for i in range(len(new_terms_list))]
        new_p = self.convert_programs(new_terms_cro)
        return new_p 
        # self.expression.extend(new_terms_list)

    def mutate_p(self,programs):
        """
            mutatation of programs
        """
        p_mutate = self.gp_config.get('p_mutate', 0)
        lib = programs[0].library
        new_programs = []
        for p in programs:
            # PDE level
            terms_list, _ = p.terms_values
            new_terms_list =[]
            for term in terms_list:
                # term level 
                if len(term)>=3:
                    new_term = []
                    for token in term:
                        # token level
                        not_mutate = np.random.choice([True, False], p=([1 - p_mutate, p_mutate]))
                        if not not_mutate:# mutate
                            new_token = mutate_single(lib, token)
                            new_term.append(new_token)
                        else:
                            new_term.append(token) 
                    new_terms_list.append(new_term)
                else:
                    new_terms_list.append(term)
            tokens_expand = self.reconstruct_tokens(new_terms_list)
            new_programs.append(tokens_expand)
            
        new_p = self.convert_programs(new_programs)
        return new_p


    def __call__(self, programs, max_length):
        
        t1 = time.perf_counter()
        #
        self.expressions = []
        # add best_tokens in history
        if len(self.best_tokens)>0:
            best_token_expend = self.reconstruct_tokens(self.best_tokens)
            self.expressions.append(best_token_expend)
        self.ut,self.u, self.x = programs[0].task.ut,programs[0].task.u,programs[0].task.x
        self.max_length = max_length

        # drop duplicates
        rp = [p.r_ridge for p in programs]
        r_quantile = np.min(rp)
        programs,rp = drop_duplicates(programs,rp)
        r_max = np.max(rp)
        if self.best_p is not None:
            programs.append(self.best_p)

        if len(self.gp_config)>0  and self.gp_config.get('p_mutate', 0)>0:
            p_cor = self.cross_over(programs)
            mutate_p = self.mutate_p(programs)
            p_gp = p_cor + mutate_p
            p_gp_high = [p for p in p_gp if p.r_ridge > r_quantile]
            r_high = [p.r_ridge for p in p_gp_high]
            # import pdb;pdb.set_trace()
            p_gp_high,r_high = drop_duplicates(p_gp_high, r_high)
            programs = programs+ p_gp_high
            p_gp_high = sorted(p_gp_high, key = lambda x: x.r_ridge, reverse=True)
            p_gp_terms =  [p.terms_values[0] for p in p_gp_high]
            p_gp_tokens = [self.reconstruct_tokens(p_gp_terms[i]) for i in range(len(p_gp_terms))]
            self.expressions.extend(p_gp_tokens)
            if len(r_high)>0:
                print(f"gp maximum reward: ",p_gp_high[0].r_ridge)
                print(f"gp exression: ", p_gp_high[0].str_expression)
            
        programs = sorted(programs, key = lambda x: x.r_ridge, reverse=True)  
                                 
        self.values,self.tokens = self.aggregate(programs[:self.STR_config['agg_num']] , p_mutate = self.STR_config['p_mutate_STR'])
        self.values_np = np.array(self.values).T # n, d (d is the number of library)
        # get coefficients
        try:
            # import pdb;pdb.set_trace()
            w,mse= TrainSTRidge(self.values_np, self.ut, l0_penalty=self.l0_penalty)
            self.l0_penalty = self.l0_coef*mse
            self.tokens = [self.tokens[i] for i in range(len(w)) if w[i]!=0]
            self.values = [self.values[i] for i in range(len(w)) if w[i]!=0]
            # add tokens 
            # import pdb;pdb.set_trace()
            # self.tokens, self.values = spatial_complete(self.tokens, self.values,Program.execute_function, self.u, self.x)
            tokens_expand = self.reconstruct_tokens(self.tokens) # add token added
            self.expressions.append(tokens_expand)
        except Exception as e:
            print(e)
            print(self.tokens)
            # import pdb;pdb.set_trace()
        

        # convert terms to agent programs
        ps, actions, obs, priors = self.get_agent_programs()

        t2 = time.perf_counter() - t1
        print(f"time cost : {t2}" )
        print(f"library length: {len(self.tokens)}")
        print(f"stridge reward: ",ps[-1].r_ridge, "mse: ", mse)
        print(f"stridge expression: {ps[-1].str_expression}")
        
        #save output whose rewards are larger than quantile_r
        r = np.array([p.r_ridge for p in ps])
        best_p = programs[0] if programs[0].r_ridge>ps[-1].r_ridge else ps[-1]
        if best_p.r_ridge>self.best_r:
            self.best_tokens, self.best_values  = best_p.terms_values
            self.best_r = best_p.r_ridge
            self.best_p = copy.copy(best_p)
        # only better examples saved
        keep = r>r_max
        if len(tokens_expand)>self.max_terms:
            keep[-1] = False
        ps = list(compress(ps,keep ))
        actions     = actions[keep, :]
        obs         = obs[keep, :, :]
        priors      = priors[keep, :, :]
        self.num=len(ps)
        return ps, actions, obs,priors
    
        
    def aggregate(self, programs, p_mutate = 0):
        """
        aggregate tokens and values with repeated terms deleted

        Args:
            programs (_type_): _description_

        Returns:
            _type_: _description_
        """
        # extract all the function terms and its tokens
        values, tokens = [],[]
        for p in programs:
            token, value = p.terms_values 
            tokens.extend(token)
            values.extend(value)
        
        tokens.extend(self.best_tokens)
        values.extend(self.best_values)
        
        new_tokens,new_values = drop_duplicates(tokens, values)

        if p_mutate>0:
            mutate_terms, mutate_values = [],[]
            for term in new_tokens:
                if len(term)>=3: # only long sequence 
                    new_term = []
                    for token in term:
                        not_mutate = np.random.choice([True, False], p=([1 - p_mutate, p_mutate]))
                        if not not_mutate:# mutate
                            new_token = mutate_single(Program.library, token)
                            new_term.append(new_token)
                        else:
                            new_term.append(token)
                    result = evaluate_terms(Program.execute_function,new_term, self.u, self.x) 
                    if not isinstance(result, tuple):
                        mutate_terms.append(new_term)
                        mutate_values.append(result) 
                    # p_list = self.convert_programs([new_term])
                    # mu_term,mu_value = p_list[0].terms_values
            new_tokens.extend(mutate_terms)
            new_values.extend(mutate_values)
            new_tokens,new_values = drop_duplicates(new_tokens,new_values)    
        return new_values, new_tokens
    
    
    def reconstruct_tokens(self, token_list):

        add_token = Program.library['add_t'] if "add_t" in Program.library.names else Program.library['add']  
        new_tokens = []
        for i in range(len(token_list)):
        
            new_tokens.append(add_token)
            new_tokens.append(token_list[i])
            
        del new_tokens[-2]
        # expand list
        tokens_expand =[]
        for i in new_tokens:
            if isinstance(i,list):
                tokens_expand.extend(i)
            else:
                tokens_expand.append(i)
        
        return tokens_expand      
        
            

