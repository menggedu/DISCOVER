import numpy as np
import math
import torch

from dso.task.pde.utils_noise import tensor2np,cut_bound
from dso.execute import python_execute, python_execute_torch

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
    def __init__(self, max_depth= 4 ):
        self.max_depth = max_depth
        
    def apply_regulations(self, x, traversal, terms_token, depth):
        # symmetric regulations
        dim = len(x)
        num = repr(traversal).count('x1')
        omit_list = []
        error_list= []
        # import pdb;pdb.set_trace()
        for i in range(1,dim):
            new_num = repr(traversal).count(f'x{i+1}')
            if new_num!= num:
                error_list.append('spatial_error')

        for i, traversal in enumerate(terms_token):
            if ('diff' in repr(traversal) or 'Diff' in repr(traversal)) and ', u' not in repr(traversal):
                error_list.append('no_u')  
                omit_list.append(i)

            if depth[i] > self.max_depth:
                error_list.append('depth_limit')
                omit_list.append(i)
        return omit_list,error_list

class STRidge(object):
    execute_function = None
    cache = {}
    def __init__(self, traversal, default_terms =[], noise_level=0,
                 max_depth=4,
                 cut_ratio = 0.02,
                 spatial_error = False):
        self.traversal = traversal
        self.traversal_copy = traversal.copy()
        # self.set_execute_function()
        self.term_values = []
        self.terms=[]
        self.invalid = False
        self.w_sym = []
        self.default_terms = [build_tree_new(dt) for dt in default_terms]
        self.split_forest()
        self.regulation = Regulations(max_depth=max_depth)
        self.omit_terms = []
        self.noise_level = noise_level
        self.spatial_error = spatial_error
        self.cut_ratio = cut_ratio
        
    def set_execute_function(self):
        pass
        
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

    def calculate(self,u,x,ut, test=False, execute_function = unsafe_execute):
        results = []
        # import pdb;pdb.set_trace()
        omit_list, err_list = self.regulation.apply_regulations(x,self.traversal_copy, self.terms_token, self.depth)
        if len(err_list)>0:
            if len(err_list) == 1 and 'spatial_error' in err_list and self.spatial_error:
                invalid=True
                return 0,[0],invalid,'spatial_error','spatial_error',None
            else:
                invalid = True
                return 0,[0],invalid,'+'.join(err_list),'+'.join(err_list),None
        
        
        for i,traversal in enumerate(self.terms_token):
            
            result, invalid, error_node, error_type = execute_function(traversal, u, x)
            # result = result[2:-2,1:] #RRE
            
            if invalid:
                return 0,[0],invalid,error_node,error_type,None
            
            if torch.is_tensor(result):
                result = tensor2np(result)
            else:
                
                if self.noise_level>0:
                    result = cut_bound(result, percent=self.cut_ratio,test = test)
            results.append(result.reshape(-1))
        # import pdb;pdb.set_trace()
        
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
            results = np.array(results_left).T 
            invalid = True   
            self.omit_terms.extend(omit_terms)
        else:
            results_left = results
            
            results = np.array(results).T
        # coefficients calculation

        try:
            
            w_best = np.linalg.lstsq(results, ut)[0]
        except Exception as e:
            invalid = True
            return 0, [0], invalid, "bad_str", 'bad_str',None
        
        omit_terms2 = []
        for i in range(len(w_best)):
            if np.abs(w_best[i])<1e-5 or np.abs(w_best[i])>1e4:
                return 0, [0], invalid,"small_coe", 'small_coe',None
                omit_terms2.append(i)
           
        # if len(omit_terms2)>0:
        #     w_best_left = [w_best[i] for i in range(len(w_best)) if i not in omit_terms2]
        #     terms_token = [self.terms_token[i] for i in range(len(self.terms_token)) if i not in omit_terms2]
        #     terms = [self.terms[i] for i in range(len(self.terms)) if i not in omit_terms2]
        #     results_left2 = [results_left[i] for i in range(len(results_left)) if i not in omit_terms2]
        #     self.terms = terms
        #     self.terms_token = terms_token
        #     w_best = np.array(w_best_left)
        #     results = np.array(results_left2).T 
        #     invalid = True

        #     for n in omit_terms2:
        #         bias = 0
        #         for i in omit_terms:
        #             if n<=i:
        #                 bias+=1
        #         self.omit_terms.append(n+bias)
        
        y_hat = results.dot(w_best)
        w_best = w_best.reshape(-1).tolist()

        return y_hat, w_best, invalid,error_node,error_type,results
    
    
    def calculate_RHS(self, u, x, ut, coefs):
        # automatic differentiation
        
        # assert len(coefs) ==  len(self.terms_token)
        # w = np.array(w).reshape(-1,1)
        results = []
        RHS = 0
        for i,traversal in enumerate(self.terms_token):
            try:
                result,_,_,_ = unsafe_execute_torch(traversal, u,x)
            except Exception as e:
                print("bad program")
                return torch.tensor(float('nan'))
                # import pdb;pdb.set_trace()
            # import pdb;pdb.set_trace()
            # result = tensor2np(result)
            # results.append(result.reshape(-1))
            results.append(result)
            
            # RHS += result*w[i]
        if isinstance(coefs, list):
            for i in range(len(coefs)):
                RHS += results[i]*coefs[i]
                
        else:
            RHS = coefs(results)
            
        residual = ut-RHS
        return residual
           
           
