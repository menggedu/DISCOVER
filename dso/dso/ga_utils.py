import numpy as np
import copy
import torch
import collections
import random
import itertools

from dso.subroutines import jit_parents_siblings_at_once
from dso.program import Program, from_tokens
from dso.task.pde.utils_noise import tensor2np

def TrainSTRidge(R, Ut, lam=1e-5, d_tol=1, maxit=100, STR_iters = 10, l0_penalty = None, normalize = 2, split = 0, 
                    print_best_tol = False):            
    """
    Sparse regression with STRidge

    Args:
        R (_type_): _description_
        Ut (_type_): _description_
        lam (_type_, optional): _description_. Defaults to 1e-5.
        d_tol (int, optional): _description_. Defaults to 1.
        maxit (int, optional): _description_. Defaults to 100.
        STR_iters (int, optional): _description_. Defaults to 10.
        l0_penalty (_type_, optional): _description_. Defaults to None.
        normalize (int, optional): _description_. Defaults to 2.
        split (int, optional): _description_. Defaults to 0.
        print_best_tol (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # Split data into 80% training and 20% test, then search for the best tolderance.
    if split != 0:
        np.random.seed(0) # for consistancy
        n,_ = R.shape
        train = np.random.choice(n, int(n*split), replace = False)
        test = [i for i in np.arange(n) if i not in train]
        TrainR = R[train,:]
        TestR = R[test,:]
        TrainY = Ut[train,:]
        TestY = Ut[test,:]
    else:
        TrainR = R
        TestR = R
        TrainY = Ut
        TestY = Ut
        
    # Set up the initial tolerance and l0 penalty
    d_tol = float(d_tol)
    tol = d_tol
    
    if l0_penalty == None:    
        l0_penalty = 0.001
        
    D = TrainR.shape[1]        
    w = np.zeros((D,1))
    w_best = np.linalg.lstsq(TrainR, TrainY,rcond=None)[0]
    err_best = np.mean((TestY - TestR.dot(w_best))**2)  + l0_penalty*np.count_nonzero(w_best)
    tol_best = 0
    
                
    # Now increase tolerance until test performance decreases
    for iter in range(maxit):

        # Get a set of coefficients and error
        w = STRidge(TrainR, TrainY, lam, STR_iters, tol, normalize =normalize )
        
        err = np.mean((TestY - TestR.dot(w))**2)  + l0_penalty*np.count_nonzero(w)

        # Has the accuracy improved?
        if err <= err_best:
            err_best = err
            w_best = w
            tol_best = tol
            tol = tol + d_tol
        else:
            tol = max([0,tol - 2*d_tol])
            d_tol =2*d_tol / (maxit - iter)# d_tol/1.618
            tol = tol + d_tol

    if print_best_tol: print ("Optimal tolerance:", tol_best)
    test_err =  np.mean((TestY - TestR.dot(w_best))**2)   
    return w_best, test_err


def STRidge(X0, y, lam, maxit, tol, normalize=0, print_results=False):
    """
    Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse
    approximation to X^{-1}y.  The idea is that this may do better with correlated observables.
    This assumes y is only one column
    """
    n, d = X0.shape
    X = np.zeros((n, d))
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d, 1))
        for i in range(0, d):
            Mreg[i] = 1.0 / (np.linalg.norm(X0[:, i], normalize))
            X[:, i] = Mreg[i] * X0[:, i]
    else:
        X = X0
    # Get the standard ridge estimate
    if lam != 0:
        w = np.linalg.lstsq(X.T.dot(X) + lam * np.eye(d), X.T.dot(y))[0]
    else:
        w = np.linalg.lstsq(X, y)[0]
    num_relevant = d
    biginds = np.where(abs(w) > tol)[0]
    # Threshold and continue
    for j in range(maxit):
        # Figure out which items to cut out
        smallinds = np.where(abs(w) < tol)[0]
        new_biginds = [i for i in range(d) if i not in smallinds]
        # If nothing changes then stop
        if num_relevant == len(new_biginds):
            break
        else:
            num_relevant = len(new_biginds)
        # Also make sure we didn't just lose all the coefficients
        if len(new_biginds) == 0:
            if j == 0:
                return w
            else:
                break
        biginds = new_biginds
        # Otherwise get a new guess
        w[smallinds] = 0
        if lam != 0:
            w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam * np.eye(len(biginds)), X[:, biginds].T.dot(y))[0]
        else:
            w[biginds] = np.linalg.lstsq(X[:, biginds], y)[0]
    # Now that we have the sparsity pattern, use standard least squares to get w
    if biginds != []: 
        w[biginds] = np.linalg.lstsq(X[:, biginds], y)[0]
    if normalize != 0:
        return np.multiply(Mreg, w)
    else:
        return w
####

def tokens2index(tokens, max_length):
    """convert tokens to actions and padding

    Args:
        tokens (_type_): _description_
        max_length (_type_): _description_

    Returns:
        _type_: _description_
    """
    actions = Program.library.actionize(tokens)
    actions_padded = np.zeros(max_length, dtype=np.int32)
    actions_padded[:len(actions)] = actions
    return actions_padded

def mutate_single(lib, token,terminal_only =False):
    '''
    mutation on a single node
    '''
    action = lib.actionize(token)[0]
    if token.arity == 0:    
        if token.state_var is not None:
            candidate = lib.state_tokens

        elif token.input_var is not None:
            candidate = lib.input_tokens
    else:
        if terminal_only :
            return token
        
        candidate = lib.tokens_of_arity[token.arity]
        # import pdb;pdb.set_trace()
        # return token
    if len(candidate) == 1:
        # print("only one token, mutation invalid")
        return token
    new_op = np.random.choice(candidate)
    while new_op==action:
        new_op = np.random.choice(candidate)
    return lib[new_op]

def cross_over_p(pde1, pde2, p_cv):
    '''
    pde refers to two programs
    cross over terms
    '''
    # new_pde1, new_pde2 = copy.deepcopy(pde1), copy.deepcopy(pde2)
    terms1, terms2 = copy.copy(pde1.terms_values[0]), copy.copy(pde2.terms_values[0])
    w1, w2 = len(terms1), len(terms2)
    ix1, ix2 = np.random.randint(w1), np.random.randint(w2)
    if np.random.random() <p_cv:
        terms1[ix1], terms2[ix2] = terms2[ix2],terms1[ix1]
    else:
        #add and delete
        terms1.append(terms2[ix2])
        if len(terms2)>1:
            del terms2[ix2]


    return terms1, terms2

def drop_duplicates( tokens, values):
    '''
    
    '''
    # import pdb;pdb.set_trace()
    unique_tokens = []
    unique_values = []
    token_len = [len(t) if isinstance(t, list) else t.len_traversal for t in tokens]
    # import pdb;pdb.set_trace()

    for i, (arr_current, str_current) in enumerate(zip(values, token_len)):
        duplicate_found = False
        for arr_compare, str_compare in zip(values[i+1:], token_len[i+1:]):
            # if the sum of differences is less than 1e-5, consider them the same
            if np.abs(np.sum(arr_compare - arr_current)) < 1e-5:
                duplicate_found = True
                # if the current string is shorter, replace the compared one
                if str_current < str_compare:
                    idx = token_len.index(str_compare)
                    token_len[idx] = str_current
                    tokens[idx]=tokens[i]
                    values[idx]=values[i]

        if not duplicate_found:
            unique_values.append(arr_current)
            unique_tokens.append(tokens[i])

    return unique_tokens, unique_values

def drop_duplicates_v2(tokens, values):
    """
    drop terms/programs/tokens with same values/rewards

    Args:
        tokens (_type_): _description_ [[add,u,u], ... ]
        values (_type_): _description_
    """
    new_values, new_tokens = [],[]
    while len(values)>0:
        value, token= values[0], tokens[0]
        if np.abs(np.sum(value)) <1e-5: # delete values with diffrence equal to 0
            tokens.pop(0)
            values.pop(0)
            continue
        deleteID  = []
        t_len = len(token) if isinstance(token, list) else token.len_traversal
        
        keep_id = 0 
        for i,(t,v) in enumerate(zip(tokens, values)): 
            if i == 0:
                continue
            if np.abs(np.sum(value-v))<1e-5:
                cur_len = len(t) if isinstance(t, list) else t.len_traversal
                if t_len>cur_len:
                    t_len = cur_len
                    deleteID.append(keep_id)
                    keep_id = i
                else:
                    deleteID.append(i)

        new_values.append(values[keep_id])
        new_tokens.append(tokens[keep_id])
        deleteID.append(keep_id)
        values,tokens = [v for i, v in enumerate(values) if i not in deleteID],[t for i, t in enumerate(tokens) if i not in deleteID]            

    return new_tokens, new_values

def evaluate_terms(execution_func,traversal, u, x):
    #original
    result, invalid, error_node, error_type = execution_func(traversal, u, x)

    #KS example
    # result, invalid, error_node, error_type = execution_func(traversal, u[0], x[0])
    # result2, invalid, error_node, error_type = execution_func(traversal, u[1], x[1])
    # result = result-result2

    if invalid:
        return 0,[0],invalid,error_node,error_type,None
    if torch.is_tensor(result):
        result = tensor2np(result) 

    return result.reshape(-1)


def spatial_complete(tokens, values, execution_func, u, x, p_del = 1, max_term =10, supplement=False ):
    """
    only 
    """
    num = len(Program.library.input_tokens)
    if num == 1:
        return tokens, values
    new_terms, new_values = [],[]
    p = np.random.random()

    if p>p_del and not supplement:
        # delete
        # TODO complete operations need to select two pair of tokens into a group.s
        for i, term in enumerate(tokens):
            if not include_input_tokens(term):
                new_terms.append(term)
                new_values.append(values[i])
                continue
            complete_terms = complete(term)
            complete_values = [evaluate_terms(execution_func,complete_term, u, x) for complete_term in complete_terms]
            if include_term(complete_values, values)[0]:
                new_terms.append(term)
                new_values.append(values[i])
        if len(new_terms) > 0:
            return new_terms,new_values

    # supplement
    for term in tokens:
        if not include_input_tokens(term):
            continue
        complete_terms = complete(term)
        complete_values = [evaluate_terms(execution_func,complete_term, u, x) for complete_term in complete_terms]
        # import pdb;pdb.set_trace()
        # if not include_term(complete_values, values)[0]:
        #     # sample from new_terms
        #     sample_id = random.choice([i for i in range(len(complete_terms))])
        for i in range(len(complete_terms)):
            new_terms.append(complete_terms[i])
            new_values.append(complete_values[i])
        if len(new_terms)>10:
            return tokens+new_terms, values+new_values
    tokens.extend(new_terms)
    values.extend(new_values)

    return tokens, values
            

def complete(term):

    """
    multi state variables and 2-3 dimensional input tokens
    generate related dimensions, for example 3d: u_xx,-> (u_yy,u_zz,u_xx)
    can not deal with u_xy
    """
    # new_terms = []
    input_tokens = Program.library.input_tokens
    state_tokens = Program.library.state_tokens
    
    new_terms_list = []
    # obtain the occurence of state tokens
    count_input = 0
    count_state = 0
    state_group = [] 
    input_group = []
    for token in term:
        if token.state_var is not None:
            count_state+=1
            state_group.append(Program.library.actionize(token)[0])
        if token.input_var is not None:
            count_input+=1
            input_group.append(Program.library.actionize(token)[0])
    # generate state token permutations if more than one state token is utilized
    state_permutations = list(itertools.permutations(state_tokens,count_state)) if count_state >1 else []
    input_permutations = list(itertools.permutations(input_tokens,count_input))
    
    for state_permu in [tuple(state_group)] + state_permutations:
        for input_permu in input_permutations:
            if tuple(input_group) == input_permu:
                continue
            state_ind = 0
            input_ind = 0
            new_terms = []
            
            for token in term:
                if token.input_var is not None:
                    new_terms.append(Program.library[input_permu[input_ind]])
                    input_ind +=1
                elif token.state_var is not None:
                        new_terms.append(Program.library[state_permu[state_ind]])
                        state_ind +=1
                else:
                    new_terms.append(token)
                
            new_terms_list.append(new_terms)
    return new_terms_list

def check_equal(tokens):
    # convert tokens
    num = len(Program.library.input_tokens)
    num_dicts = {i:0 for i in range(num)}
    for term in tokens:
        indices = Program.library.actionize(term).tolist()
        for i in range(num):
            ind_num = indices.count(Program.library.input_tokens[i])
            num_dicts[Program.library.input_tokens[i]] += ind_num
    
    sorted_dicts = sorted(num_dicts.items, key = lambda x: x[1])
    if sorted_dicts[0][1]-sorted_dicts[-1][1] == 0:
        return True
    else:
        return False

def include_input_tokens(tokens):
    for token in tokens:
        if token.input_var is not None:
            return True
    return False

def include_term(values_gen, values):
    for vg in values_gen:
        for i, v in enumerate(values):
            if np.abs(np.sum(vg-v))<1e-5:
                return True,i
    return False,None 
    # assert len(num) <=3, "three dimension"

