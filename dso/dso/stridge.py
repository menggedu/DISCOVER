import numpy as np
import math
import torch

from dso.task.pde.utils_noise import tensor2np,cut_bound
from dso.execute import python_execute, python_execute_torch
from dso.task.pde.utils_nn import torch_diff
class InvalidLog():
    """Log class to catch and record numpy warning messages"""

    def __init__(self):
        self.error_type = None # One of ['divide', 'overflow', 'underflow', 'invalid']
        self.error_node = None # E.g. 'exp', 'log', 'true_divide'
        self.new_entry = False # Flag for whether a warning has been encountered during a call to Program.execute()

    def write(self, message):
        """This is called by numpy when encountering a warning"""

        if not self.new_entry: # Only record the first warning encounter
            message = message.strip().split(' ')
            self.error_type = message[1]
            self.error_node = message[-1]
        self.new_entry = True

    def update(self):
        """If a floating-point error was encountered, set Program.invalid
        to True and record the error type and error node."""

        if self.new_entry:
            self.new_entry = False
            return True, self.error_type, self.error_node
        else:
            return False, None, None


invalid_log = InvalidLog()
np.seterrcall(invalid_log) # Tells numpy to call InvalidLog.write() when encountering a warning

# Define closure for execute function
def unsafe_execute(traversal, u, x):
    """This is a wrapper for execute_function. If a floating-point error
    would be hit, a warning is logged instead, p.invalid is set to True,
    and the appropriate nan/inf value is returned. It's up to the task's
    reward function to decide how to handle nans/infs."""

    with np.errstate(all='log'):
        y = python_execute(traversal, u,x)
        invalid, error_node, error_type = invalid_log.update()
        return y, invalid, error_node, error_type
    
def unsafe_execute_torch(traversal, u, x):
    
    with np.errstate(all='log'):
        y = python_execute_torch(traversal, u,x)
        if y is None:
            invalid_log.write("bad_diff bad_diff bad_diff")
            return 0, True,'bad_diff','bad_diff'
        
        error_node,error_type=None,None
        invalid=False
        return y, invalid, error_node, error_type


# Possible library elements that sympy capitalizes
capital = ["add", "mul", "pow"]


      
class Node(object):
    """Basic tree class supporting printing"""

    def __init__(self, val):
        self.val = val.name
        self.children = []
        self.token = val
        self.symbol = 1
        

    def __repr__(self):
        children_repr = ",".join(repr(child) for child in self.children)
        if len(self.children) == 0:
            return self.val # Avoids unnecessary parantheses, e.g. x1()
        return "{}({})".format(self.val, children_repr)


def build_tree(traversal):
    """Recursively builds tree from pre-order traversal"""

    op = traversal.pop(0)
    n_children = op.arity
    val = repr(op)
    if val in capital:
        val = val.capitalize()

    node = Node(val)

    for _ in range(n_children):
        node.children.append(build_tree(traversal))

    return node

def build_tree_new(traversal):
    """Recursively builds tree from pre-order traversal"""

    op = traversal.pop(0)
    n_children = op.arity


    node = Node(op)

    for _ in range(n_children):
        node.children.append(build_tree(traversal))

    return node

illegal_type = ['no_u', 'spatial_error', 'depth_limit']

class Regulations(object):
    """
    set regulations to doudble check invalid expressions

    """
    def __init__(self, max_depth= 3, spatial_error = True):
        self.max_depth = max_depth
        self.spatial_error = spatial_error
        
    def apply_regulations(self, x, traversal, terms_token, depth):
        # symmetric regulations
        if self.spatial_error:
            omit_list1,error_list1 = self.check_spatial_regulation(x, traversal)
        else:
            omit_list1,error_list1 = [],[]
        omit_list2,error_list2 = self.check_single_term(terms_token, depth)
        
        return omit_list1+omit_list2,error_list1+error_list2
    
    def check_spatial_regulation(self,x,traversal):
        dim = len(x)
        if dim>0 and type(x[-1]) == str: #prevent the last input is t-related
            dim -=1
        num = repr(traversal).count('x1')
        omit_list = []
        error_list= []
        for i in range(1,dim):
            new_num = repr(traversal).count(f'x{i+1}')
            if i == dim-1 and type(x[-1]) == str :
                if new_num == 0:
                    error_list.append('spatial_error')  
                continue
            if new_num!= num:
                error_list.append('spatial_error') 
        return  omit_list,error_list
      
    def check_single_term(self, terms_token,depth):
        omit_list = []
        error_list= []
        for i, traversal in enumerate(terms_token):
            if ('diff' in repr(traversal) or 'Diff' in repr(traversal)) and ', u' not in repr(traversal):
                error_list.append('no_u')  
                omit_list.append(i)

            if depth[i] > self.max_depth:
                error_list.append('depth_limit')
                omit_list.append(i)
            # import pdb;pdb.set_trace()
            if 'u' not in repr(traversal): #in ['x1', 'x2','x3']:
                error_list.append('violate physical constraint')
                omit_list.append(i)
        return omit_list,error_list
    
        # omit_list, err_list = self.regulation.apply_regulations(x,self.traversal_copy, self.terms_token, self.depth)
        # if len(err_list)>0:
        #     if len(err_list) == 1 and 'spatial_error' in err_list and self.spatial_error:
        #         invalid=True
        #         return 0,[0],invalid,'spatial_error','spatial_error',None
        #     else:
        #         invalid = True
        #         return 0,[0],invalid,'+'.join(err_list),'+'.join(err_list),None

class STRidge(object):
    execute_function = None
    cache = {}
    """
    conduct sparse regression and calculate the coefficients
    """
    def __init__(self, traversal, default_terms =[], noise_level=0,
                 max_depth=4,
                 cut_ratio = 0.02,
                 spatial_error = False,
                 const = False):
        self.traversal = traversal
        self.traversal_copy = traversal.copy()
        # self.set_execute_function()
        self.term_values = []
        self.terms=[]
        self.invalid = False
        self.w_sym = []
        self.default_terms = [build_tree_new(dt) for dt in default_terms]
        self.split_forest()
        self.regulation = Regulations(max_depth=max_depth, spatial_error = spatial_error) 
        self.omit_terms = []
        self.noise_level = noise_level
        self.add_const = const
        self.cut_ratio = cut_ratio
        # self.results = []
        
    @classmethod
    def clear_cache(cls):
        """Clears the class' cache"""

        cls.cache = {}

    def split_forest(self):
        root = self.rebuild_tree()
    
        def split_sum(root):
            """ split the traversal node according to the '+', '-'  """
            # import pdb;pdb.set_trace()
            if root.val not in ['add','sub', 'add_t', 'sub_t']:
                
                return [root]

            if 'sub' in root.val:

                if root.symbol == 1:
                    root.children[1].symbol *=-1
                else:
                    #
                    if 'sub' in root.val:    
                        root.children[0].symbol*=-1
                        root.children[1].symbol*=-1
            
            return [split_sum(root.children[0]), split_sum(root.children[1])]

        def expand_list(input_list):
            """ expand multiple lists to one list"""
            while len(input_list)!=0:
                cur = input_list.pop(0)
                if isinstance(cur, list):
                    expand_list(cur)
                else:
                    self.terms.append(cur)

        def preorder_traverse(node):
            
            traversals = []
            nodes = [node]
            while len(nodes)!=0:
                cur = nodes.pop(0)
                traversals.append(cur.token)
                if len(cur.children) ==0:
                    pass
                elif len(cur.children) == 2:
                    nodes.insert(0, cur.children[0])
                    nodes.insert(1, cur.children[1])
                elif len(cur.children) ==1:
                    nodes.insert(0,cur.children[0])
                else:
                    print("wrong children number")
                    assert False
            return traversals  
        # import  pdb;pdb.set_trace()
        term_list = split_sum(root)
        #initial symbols for each terms (+,-)
        expand_list(term_list)
        
        self.terms.extend(self.default_terms)
        self.terms_token = [preorder_traverse(node) for node in self.terms]
        self.terms_len = [len(tokens) for tokens in self.terms_token]
 
        def max_depth(root):
            """calculate the max depth of subtreees

            Args:
                root (_type_): root node

            Returns:
                _type_: max depth of subtrees (function terms)
            """
            max_num = 0
          
            for children in root.children:
                max_num = max(max_depth(children), max_num)
            
            return max_num+1
       
        # self.depth = [max_depth(node) for node in self.terms]
        self.depth = [np.ceil(math.log(length, 2)) for length in self.terms_len]
        
    def build_tree(self,traversal):
        stack = []
        leaf_node = None
        while len(traversal) != 0:
            
            if leaf_node is None:
                op = traversal.pop(0) 
                node_arity=op.arity
                node = Node(op)
            else:
                node = leaf_node
                node_arity = 0
            if node_arity>0:
                stack.append((node, node_arity))
            else:
                leaf_node=node
                if stack!=[]:
                    last_op,last_arity = stack.pop(-1)
                    last_op.children.append(node)
                    last_arity-=1
                    if last_arity > 0:
                        stack.append((last_op,last_arity))
                        leaf_node=None  
                    else:
                        leaf_node = last_op
                
        return leaf_node
    
    def rebuild_tree(self):
        """Recursively builds tree from pre-order traversal"""

        op = self.traversal.pop(0)
        n_children = op.arity
        # import pdb;pdb.set_trace()
        node = Node(op)

        for _ in range(n_children):
            node.children.append(self.rebuild_tree())

        return node

    def wf_calculate(self, u,x,ut, test=False, execute_function = unsafe_execute, cached=None):
        results1 = self.evaluate_terms([u[0]],[x[0]], test=False, execute_function = execute_function)
        if isinstance(results1, tuple):
            return results1
        results2 = self.evaluate_terms([u[1]],[x[1]], test=False, execute_function = execute_function)
        results = [results1[i]-results2[i] for i in range(len(results1))]
        results = np.array(results).T
        return self.coef_calculate(results,ut)

    def calculate(self, u,x,ut, test=False, execute_function = unsafe_execute, cached=None):
        #evaluate function terms
        
        results = self.evaluate_terms(u,x, test=False, execute_function = execute_function)
        if isinstance(results, tuple):
            return results
        
        #coefficient calculation
        if not isinstance(ut, list):
            # self.results = results
            results = np.array(results).T
            if self.add_const:
                results = np.concatenate((results, np.ones((results.shape[0], 1))), axis = 1)
            return self.coef_calculate(results,ut)
        else:
            #stridge
            results_reshape = [res.reshape(u[0].shape) for res in results]
            t_shape,lev_shape, x_shape, y_shape = u[0].shape
            results_new = [[res[:,i,:,:].reshape(-1) for res in results_reshape ] for i in range(lev_shape)]
            # self.results = results_new
            return self.multi_coef_calculate(results_new,ut, cached_terms = cached)
            # multi state
            # pass

    def calculate_RHS_terms(self, u, x, execute_func = unsafe_execute_torch, extra_gradient=False):
        
        results = []
        
        for i,traversal in enumerate(self.terms_token):
            try:
               
                result,_,_,_ = execute_func(traversal, u,x)
            except Exception as e:
                print("bad program")
                return torch.tensor(float('nan'))
            if extra_gradient:
                for j in range(len(x)):
                    result = torch_diff(result, x[j])
            results.append(result)

        return results
        
    def evaluate_terms(self,u,x, test=False, execute_function = unsafe_execute):
        results = []
        
        omit_list, err_list = self.regulation.apply_regulations(x,\
                                self.traversal_copy, self.terms_token, self.depth)
        if len(err_list)>0:
            invalid = True
            return 0,[0],invalid,'+'.join(err_list),'+'.join(err_list),None
        
        for i,traversal in enumerate(self.terms_token):
           
            result, invalid, error_node, error_type = execute_function(traversal, u, x)
            # result = result[2:-2,1:] #RRE
            if invalid:
                return 0,[0],invalid,error_node,error_type,None
            # import pdb;pdb.set_trace()
            if torch.is_tensor(result):
                result = tensor2np(result)
            else:
                if self.noise_level>0:
                    result = cut_bound(result, percent=self.cut_ratio,test = test)
            results.append(result.reshape(-1))

        # empty results
        if len(results) ==0:
            invalid =True
            return 0,[0],invalid,'dim_error','dim_error',None
        else:
            result_shape = results[0].shape
            for res in results[1:]:
                if res.shape!=result_shape:
                    invalid =True
                    return 0,[0],invalid,'dim_error','dim_error',None

        # coefficients filter
        omit_terms = []
        omit_terms.extend(omit_list)
        omit_terms = set(omit_terms)
        
        if len(omit_terms)>0:
            terms_token = [self.terms_token[i] for i in range(len(self.terms_token)) if i not in omit_terms]
            terms = [self.terms[i] for i in range(len(self.terms)) if i not in omit_terms]
            results_left = [results[i] for i in range(len(results)) if i not in omit_terms]
            self.terms = terms
            self.terms_token = terms_token
            invalid = True   
            self.omit_terms.extend(omit_terms)
        else:
            results_left = results
            #n d

        return results_left

    def coef_calculate(self,rhs, lhs):
        # from sklearn.linear_model import LinearRegression
        # lr = LinearRegression(fit_intercept=False).fit(rhs,lhs)
        try:
            w_best = np.linalg.lstsq(rhs, lhs)[0]
        except Exception as e:
            # print(e)
            invalid = True
            return 0, [0], invalid, "bad_str", 'bad_str',None
        
        y_hat = rhs.dot(w_best)
        # y_hat_const = rhs_const.dot(w_best_const)
        r = np.sum(np.abs(y_hat-rhs))
        # r_const = np.sum(np.abs(y_hat_const))
        # if r_const<r:
        #     for i in range(len(w_best_const)):
        #         if np.abs(w_best_const[i])<1e-5 or np.abs(w_best_const[i])>1e4:
        #             return 0, [0], True,"small_coe", 'small_coe',None
        #     return y_hat_const, w_best_const, False,None,None,rhs_const
        # old
        for i in range(len(w_best)):
            

            if np.abs(w_best[i])<5e-5:
                if self.add_const and i== len(w_best)-1:
                    rhs= rhs[:,:-1]
                    w_best = np.linalg.lstsq(rhs, lhs)[0]  
                    continue
                return 0, [0], True,"small_coe", 'small_coe',None
            if  np.abs(w_best[i])>1e4:
                return 0, [0], True,"large_coe", 'large_coe',None
            
        y_hat = rhs.dot(w_best)
        w_best = w_best.reshape(-1).tolist()
        return y_hat, w_best, False,None,None,rhs
    
    def multi_coef_calculate(self, results, uts, cached_terms = (None,None)):
        """
        multi-dataset for subgrid force term prediction
        

        Args:
            results (_type_):  list =  lev [(txy), ...]
            ut (_type_): q subgrid force shape = (t, lev, x, y)
        """
        #subgrid
        
        assert len(uts) == len(results)
        n = len(results[0])
        y_hat_list, w_list, y_rhs_list = [],[],[]
        _, cached_vals = cached_terms
        for i in range(len(results)):
            result = np.array(results[i]).reshape(n, -1).T
            if cached_vals is not None:
                result = np.concatenate((result, cached_vals[i]), axis = 1)
            ut = uts[i].reshape(-1,1)
            y_hat, w, invalid, error_node, error_type, y_rhs = self.coef_calculate(result, ut)
            if invalid:
                return 0, [0], invalid, error_node, error_type, y_rhs
            else:
                y_hat_list.append(y_hat)
                w_list.append(w)
                y_rhs_list.append(y_rhs)
        return y_hat_list, w_list,invalid,error_node, error_type, results

    def calculate_RHS(self, u, x, ut, coefs, execute_func = unsafe_execute_torch,extra_gradient=False ):
        results = self.calculate_RHS_terms(u,x, execute_func,extra_gradient=extra_gradient)
        RHS = 0
        if isinstance(coefs, list):
            for i in range(len(coefs)):
                RHS += results[i]*coefs[i]        
        else:
            RHS = coefs(results)
            
        residual = ut-RHS
        return residual
    
    def calculate_RHS_terms(self, u, x, execute_func = unsafe_execute_torch, extra_gradient=False):
        
        results = []
        
        for i,traversal in enumerate(self.terms_token):
            try:
               
                result,_,_,_ = execute_func(traversal, u,x)
            except Exception as e:
                print("bad program")
                return torch.tensor(float('nan'))
            if extra_gradient:
                for j in range(len(x)):
                    result = torch_diff(result, x[j])
            results.append(result)

        return results
    
    def evaluate(self,traversal,u,x,ut, execute_function = unsafe_execute):
       
        omit_list, err_list = self.regulation.apply_regulations(x,traversal, self.terms_token, self.depth)
        if len(err_list)>0:
            if len(err_list) == 1 and 'spatial_error' in err_list and self.spatial_error:
                invalid=True
                return 0,[0],invalid,'spatial_error','spatial_error',None
            else:
                invalid = True
                return 0,[0],invalid,'+'.join(err_list),'+'.join(err_list),None
        
        result, invalid, error_node, error_type = execute_function(traversal, u, x)
                    
        if invalid:
            return 0,[0],invalid,error_node,error_type,None
        
        if torch.is_tensor(result):
            result = tensor2np(result)
        else:   
            if self.noise_level>0:
                result = cut_bound(result, percent=self.cut_ratio)
        result = result.reshape(-1,1)
        if result.shape != ut.shape:
            invalid = True
            error_node, error_type = 'dim_error','dim_error'

        return result,[1 for _ in range(len(self.terms_token))],invalid,error_node, error_type,result
