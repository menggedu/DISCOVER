"""Class for symbolic expression object or program."""

import array
# from msilib.schema import Error
import warnings
from textwrap import indent

import numpy as np
from sympy.parsing.sympy_parser import parse_expr
from sympy import pretty
import time

from dso.functions import PlaceholderConstant
from dso.const import make_const_optimizer
from dso.utils import cached_property
import dso.utils as U



def _finish_tokens(tokens):

    """
    Complete a possibly unfinished string of tokens.

    Parameters
    ----------
    tokens : list of integers
        A list of integers corresponding to tokens in the library. The list
        defines an expression's pre-order traversal.

    Returns
    _______
    tokens : list of ints
        A list of integers corresponding to tokens in the library. The list
        defines an expression's pre-order traversal. "Dangling" programs are
        completed with repeated "x1" until the expression completes.

    """

    n_objects = Program.n_objects
    # import pdb;pdb.set_trace()
    arities = np.array([Program.library.arities[t] for t in tokens])
    # Number of dangling nodes, returns the cumsum up to each point
    # Note that terminal nodes are -1 while functions will be >= 0 since arities - 1
    dangling = 1 + np.cumsum(arities - 1)

    if -n_objects in (dangling - 1):
        # Chop off tokens once the cumsum reaches 0, This is the last valid point in the tokens
        expr_length = 1 + np.argmax((dangling - 1) == -n_objects)
        tokens = tokens[:expr_length]
    else:
        # Extend with valid variables until string is valid
        # NOTE: This only appends onto the end of a set of tokens, even in the multi-object case!
        assert n_objects == 1, "Is max length constraint turned on? Max length constraint required when n_objects > 1."
        tokens = np.append(tokens, np.random.choice(Program.library.input_tokens, size=dangling[-1]))
        
    return tokens


def from_str_tokens(str_tokens, skip_cache=False):
    """
    Memoized function to generate a Program from a list of str and/or float.
    See from_tokens() for details.

    Parameters
    ----------
    str_tokens : str | list of (str | float)
        Either a comma-separated string of tokens and/or floats, or a list of
        str and/or floats.

    skip_cache : bool
        See from_tokens().

    Returns
    -------
    program : Program
        See from_tokens().
    """

    # Convert str to list of str
    if isinstance(str_tokens, str):
        str_tokens = str_tokens.split(",")

    # Convert list of str|float to list of tokens
    if isinstance(str_tokens, list):
        traversal = []
        constants = []
        for s in str_tokens:
            if s in Program.library.names:
                t = Program.library.names.index(s.lower())
            elif U.is_float(s):
                assert "const" not in str_tokens, "Currently does not support both placeholder and hard-coded constants."
                t = Program.library.const_token
                constants.append(float(s))
            else:
                raise ValueError("Did not recognize token {}.".format(s))
            traversal.append(t)
        traversal = np.array(traversal, dtype=np.int32)
    else:
        raise ValueError("Input must be list or string.")

    # Generate base Program (with "const" for constants)
    p = from_tokens(traversal, skip_cache=skip_cache)

    # Replace any constants
    p.set_constants(constants)

    return p


def from_tokens(tokens, skip_cache=False, on_policy=True, finish_tokens=True):

    """
    Memoized function to generate a Program from a list of tokens.

    Since some tokens are nonfunctional, this first computes the corresponding
    traversal. If that traversal exists in the cache, the corresponding Program
    is returned. Otherwise, a new Program is returned.

    Parameters
    ----------
    tokens : list of integers
        A list of integers corresponding to tokens in the library. The list
        defines an expression's pre-order traversal. "Dangling" programs are
        completed with repeated "x1" until the expression completes.

    skip_cache : bool
        Whether to bypass the cache when creating the program (used for
        previously learned symbolic actions in DSP).
        
    finish_tokens: bool
        Do we need to finish this token. There are instances where we have
        already done this. Most likely you will want this to be True. 

    Returns
    _______
    program : Program
        The Program corresponding to the tokens, either pulled from memoization
        or generated from scratch.
    """

    '''
        Truncate expressions that complete early; extend ones that don't complete
    '''
  
    if finish_tokens:
        tokens = _finish_tokens(tokens)

    # For stochastic Tasks, there is no cache; always generate a new Program.
    # For deterministic Programs, if the Program is in the cache, return it;
    # otherwise, create a new one and add it to the cache.
    # import pdb;pdb.set_trace()
    if skip_cache or Program.task.stochastic:
        p = Program(tokens, on_policy=on_policy)
    else:
        key = tokens.tostring() 
        try:
            p = Program.cache[key]
            if on_policy:
                p.on_policy_count += 1
            else:
                p.off_policy_count += 1
        except KeyError:
            p = Program(tokens, on_policy=on_policy)
            Program.cache[key] = p

    return p


class Program(object):
    """
    The executable program representing the symbolic expression.

    The program comprises unary/binary operators, constant placeholders
    (to-be-optimized), input variables, and hard-coded constants.

    Parameters
    ----------
    tokens : list of integers
        A list of integers corresponding to tokens in the library. "Dangling"
        programs are completed with repeated "x1" until the expression
        completes.

    Attributes
    ----------
    traversal : list
        List of operators (type: Function) and terminals (type: int, float, or
        str ("const")) encoding the pre-order traversal of the expression tree.

    tokens : np.ndarry (dtype: int)
        Array of integers whose values correspond to indices

    const_pos : list of int
        A list of indicies of constant placeholders along the traversal.

    float_pos : list of float
        A list of indices of constants placeholders or floating-point constants
        along the traversal.

    sympy_expr : str
        The (lazily calculated) SymPy expression corresponding to the program.
        Used for pretty printing _only_.

    complexity : float
        The (lazily calcualted) complexity of the program.

    r : float
        The (lazily calculated) reward of the program.

    count : int
        The number of times this Program has been sampled.

    str : str
        String representation of tokens. Useful as unique identifier.
    """

    # Static variables
    task = None             # Task
    library = None          # Library
    const_optimizer = None  # Function to optimize constants
    cache = {}
    n_objects = 1          # Number of executable objects per Program instance

    default_terms=[]
    # Cython-related static variables
    have_cython = None      # Do we have cython installed
    execute = None          # Link to execute. Either cython or python
    cyfunc = None           # Link to cyfunc lib since we do an include inline

    def __init__(self, tokens=None, on_policy=True):
        """
        Builds the Program from a list of of integers corresponding to Tokens.
        """
        
        # Can be empty if we are unpickling 
        if tokens is not None:
            self._init(tokens, on_policy)
            
    def _init(self, tokens, on_policy=True):

        self.traversal = [Program.library[t] for t in tokens]
        default_terms = [[Program.library[t]] for term in Program.default_terms for t in term]
        # import pdb;pdb.set_trace()
        self.STRidge = STRidge(self.traversal.copy(),default_terms)
        
        self.const_pos = [i for i, t in enumerate(self.traversal) if isinstance(t, PlaceholderConstant)]
        self.len_traversal = len(self.traversal)

        if self.have_cython and self.len_traversal > 1:
            self.is_input_var = array.array('i', [t.input_var is not None for t in self.traversal])

        self.invalid = False
        self.str = tokens.tostring()
        self.tokens = tokens

        self.on_policy_count = 1 if on_policy else 0
        self.off_policy_count = 0 if on_policy else 1
        self.originally_on_policy = on_policy # Note if a program was created on policy

        if Program.n_objects > 1:
            # Fill list of multi-traversals
            danglings = -1 * np.arange(1, Program.n_objects + 1)
            self.traversals = [] # list to keep track of each multi-traversal
            i_prev = 0
            arity_list = [] # list of arities for each node in the overall traversal
            for i, token in enumerate(self.traversal):
                arities = token.arity
                arity_list.append(arities)
                dangling = 1 + np.cumsum(np.array(arity_list) - 1)[-1]
                if (dangling - 1) in danglings:
                    trav_object = self.traversal[i_prev:i+1]
                    self.traversals.append(trav_object)
                    i_prev = i+1
                    """
                    Keep only what dangling values have not yet been calculated. Don't want dangling to go down and up (e.g hits -1, goes back up to 0 before hitting -2)
                    and trigger the end of a traversal at the wrong time
                    """
                    danglings = danglings[danglings != dangling - 1]
                    
    def __getstate__(self):
        
        have_r = "r" in self.__dict__
        have_evaluate = "evaluate" in self.__dict__
        possible_const = have_r or have_evaluate
        
        state_dict = {'tokens' : self.tokens, # string rep comes out different if we cast to array, so we can get cache misses.
                      'have_r' : bool(have_r),
                      'r' : float(self.r) if have_r else float(-np.inf), 
                      'have_evaluate' : bool(have_evaluate),
                      'evaluate' : self.evaluate if have_evaluate else float(-np.inf), 
                      'const' : array.array('d', self.get_constants()) if possible_const else float(-np.inf), 
                      'on_policy_count' : bool(self.on_policy_count),
                      'off_policy_count' : bool(self.off_policy_count),
                      'originally_on_policy' : bool(self.originally_on_policy),
                      'invalid' : bool(self.invalid), 
                      'error_node' : array.array('u', "" if not self.invalid else self.error_node), 
                      'error_type' : array.array('u', "" if not self.invalid else self.error_type)}    
        
        # In the future we might also return sympy_expr and complexity if we ever need to compute in parallel 

        return state_dict
                
    def __setstate__(self, state_dict):
        
        # Question, do we need to init everything when we have already run, or just some things?
        self._init(state_dict['tokens'], state_dict['originally_on_policy'])
        
        have_run = False
        
        if state_dict['have_r']:
            setattr(self, 'r', state_dict['r'])
            have_run = True
            
        if state_dict['have_evaluate']:
            setattr(self, 'evaluate', state_dict['evaluate'])
            have_run = True 
        
        if have_run:
            self.set_constants(state_dict['const'].tolist())
            self.invalid = state_dict['invalid']
            self.error_node = state_dict['error_node'].tounicode()
            self.error_type = state_dict['error_type'].tounicode()
            self.on_policy_count = state_dict['on_policy_count']
            self.off_policy_count = state_dict['off_policy_count']

    def execute_STR(self,u, x,ut):
        import time
        st = time.time()
        y_hat, w_best, self.invalid,self.error_node,self.error_type= self.STRidge.calculate(u,x,ut)
        et  = time.time()
        # if et-st>1:
        # print('long calculate: ', et-st)
            # import pdb;pdb.set_trace()
        return y_hat, w_best

    def execute(self, u, x):
        """
        Execute program on input u, x.

        Parameters
        ==========

        u : np.array
            main variable to execute the Program over.

        Returns
        =======

        result : np.array or list of np.array
            In a single-object Program, returns just an array. In a multi-object Program, returns a list of arrays.
        """
        if Program.n_objects > 1:
            if not Program.protected:
                result = []
                invalids = []
                for trav in self.traversals:
                    val, invalid, self.error_node, self.error_type = Program.execute_function(trav, X)
                    result.append(val)
                    invalids.append(invalid)
                self.invalid = any(invalids)
            else:
                result = [Program.execute_function(trav, u, x) for trav in self.traversals]
            return result
        else:
            if not Program.protected:
                result, self.invalid, self.error_node, self.error_type = Program.execute_function(self.traversal, u, x)
            else:
                result = Program.execute_function(self.traversal, u, x)
            return result


    def optimize(self):
        """
        Optimizes PlaceholderConstant tokens against the reward function. The
        optimized values are stored in the traversal.
        """

        if len(self.const_pos) == 0:
            return
        
        # Define the objective function: negative reward
        def f(consts):
            self.set_constants(consts)
            r = self.task.reward_function(self)
            obj = -r # Constant optimizer minimizes the objective function

            # Need to reset to False so that a single invalid call during
            # constant optimization doesn't render the whole Program invalid.
            self.invalid = False

            return obj

        # Do the optimization
        x0 = np.ones(len(self.const_pos)) # Initial guess
        optimized_constants = Program.const_optimizer(f, x0)

        # Set the optimized constants
        self.set_constants(optimized_constants)

    def get_constants(self):
        """Returns the values of a Program's constants."""

        return [t.value for t in self.traversal if isinstance(t, PlaceholderConstant)]

    def set_constants(self, consts):
        """Sets the program's constants to the given values"""

        for i, const in enumerate(consts):
            assert U.is_float, "Input to program constants must be of a floating point type"
            # Create a new instance of PlaceholderConstant instead of changing
            # the "values" attribute, otherwise all Programs will have the same
            # instance and just overwrite each other's value.
            self.traversal[self.const_pos[i]] = PlaceholderConstant(const)


    @classmethod
    def set_n_objects(cls, n_objects):
        Program.n_objects = n_objects

    @classmethod
    def clear_cache(cls):
        """Clears the class' cache"""

        cls.cache = {}


    @classmethod
    def set_task(cls, task):
        """Sets the class' Task"""

        Program.task = task
        Program.library = task.library
    @classmethod
    def set_default_terms(cls, default_terms):
        Program.default_terms = default_terms

    @classmethod
    def set_const_optimizer(cls, name, **kwargs):
        """Sets the class' constant optimizer"""

        const_optimizer = make_const_optimizer(name, **kwargs)
        Program.const_optimizer = const_optimizer


    @classmethod
    def set_complexity(cls, name):
        """Sets the class' complexity function"""

        all_functions = {
            # No complexity
            None : lambda p : 0.0,

            # Length of sequence
            "length" : lambda p : len(p.traversal),

            # Sum of token-wise complexities
            "token" : lambda p : sum([t.complexity for t in p.traversal]),

        }

        assert name in all_functions, "Unrecognzied complexity function name."

        Program.complexity_function = lambda p : all_functions[name](p)

    @classmethod
    def set_execute(cls, protected):
        """Sets which execute method to use"""

        # Check if cython_execute can be imported; if not, fall back to python_execute
        # try:
        #     from dso import cyfunc
        #     from dso.execute import cython_execute
        #     execute_function        = cython_execute
        #     Program.have_cython     = True
        # except ImportError:
        from dso.execute import python_execute
        execute_function        = python_execute
        Program.have_cython     = False

        Program.protected = False
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
                y = execute_function(traversal, u,x)
                invalid, error_node, error_type = invalid_log.update()
                return y, invalid, error_node, error_type

        Program.execute_function = unsafe_execute

    @cached_property
    def r_ridge(self):
        """Evaluates and returns the reward of the program"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import time
            st = time.time()

            # Return final reward after optimizing
            
            result, self.w = self.task.reward_function(self)
            
            dur =time.time()-st
            # print(dur)
            return result

    @cached_property
    def r(self):
        """Evaluates and returns the reward of the program"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # import time
            # st = time.time()
            # Optimize any PlaceholderConstants
            self.optimize()

            # Return final reward after optimizing
            result = self.task.reward_function(self)
            # dur =time.time()-st
            # print(dur)
            # if dur>2:
            #     import pdb;pdb.set_trace()
            return result

    @cached_property
    def complexity(self):
        """Evaluates and returns the complexity of the program"""

        return Program.complexity_function(self)

    @cached_property
    def evaluate(self):
        """Evaluates and returns the evaluation metrics of the program."""

        # Program must be optimized before computing evaluate
        # if "r" not in self.__dict__:
        #     print("WARNING: Evaluating Program before computing its reward." \
        #           "Program will be optimized first.")
        #     self.optimize()

        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")

        return self.task.evaluate(self)

    @cached_property
    def sympy_expr(self):
        """
        Returns the attribute self.sympy_expr.

        This is actually a bit complicated because we have to go: traversal -->
        tree --> serialized tree --> SymPy expression
        """

        if Program.n_objects == 1:
            tree = self.traversal.copy()
            tree = build_tree(tree)
            tree = convert_to_sympy(tree)
            try:
                expr = parse_expr(tree.__repr__()) # SymPy expression
            except:
                expr = tree.__repr__()
            return [expr]
        else:
            exprs = []
            for i in range(len(self.traversals)):
                tree = self.traversals[i].copy()
                tree = build_tree(tree)
                tree = convert_to_sympy(tree)
                try:
                    expr = parse_expr(tree.__repr__()) # SymPy expression
                except:
                    expr = tree.__repr__()
                exprs.append(expr)
            return exprs

    def pretty(self):
        """Returns pretty printed string of the program"""
        return [pretty(self.sympy_expr[i]) for i in range(Program.n_objects)] 


    def print_stats(self):
        """Prints the statistics of the program
        
            We will print the most honest reward possible when using validation.
        """
        
        print("\tReward: {}".format(self.r_ridge))
        print("\tMSE:{}".format(self.evaluate['nmse_test']))
        print("\tCount Off-policy: {}".format(self.off_policy_count))
        print("\tCount On-policy: {}".format(self.on_policy_count))
        print("\tOriginally on Policy: {}".format(self.originally_on_policy))
        print("\tInvalid: {}".format(self.invalid))
        print("\tTraversal: {}".format(self))

        if Program.n_objects == 1:
            print("\tExpression:")
            # print("{}\n".format(indent(self.pretty()[0], '\t  ')))
            print(self.str_expression)
        else:
            for i in range(Program.n_objects):
                print("\tExpression {}:".format(i))
                print("{}\n".format(indent(self.pretty()[i], '\t  ')))

    def __repr__(self):
        """Prints the program's traversal"""
        
        return ','.join([repr(t) for t in self.traversal])

    @property   
    def str_expression(self):
        out = ""
        for number, traversal in zip(self.w, self.STRidge.terms):
            out+=str(number)+" * "+ repr(traversal) +' + '
        
        return out[:-3]


###############################################################################
# Everything below this line is currently only being used for pretty printing #
###############################################################################


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



class STRidge(object):
    def __init__(self, traversal,default_terms =[]):
        self.traversal = traversal
        self.traversal_copy = traversal.copy()
        self.term_values = []
        self.terms=[]
        self.invalid = False
        self.w_sym = []
        self.default_terms = [build_tree_new(dt) for dt in default_terms]
        self.split_forest()

    def split_forest(self):
        root = self.rebuild_tree()
        
        def split_sum(root):
            """ split the traversal node according to the '+‘, '-'  """
            # import pdb;pdb.set_trace()
            if root.val not in ['add','sub']:
                
                return [root]

            if root.val == 'sub':

                if root.symbol == 1:
                    root.children[1].symbol *=-1
                else:
                    #
                    if root.val == 'sub':    
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
        # self.terms_token.extend(self.default_terms)

    def rebuild_tree(self):
        """Recursively builds tree from pre-order traversal"""

        op = self.traversal.pop(0)
        n_children = op.arity
        # import pdb;pdb.set_trace()
        node = Node(op)

        for _ in range(n_children):
            node.children.append(self.rebuild_tree())

        return node

    def calculate(self,u,x,ut):
        results = []
        # sum over spatial coordinates
        dim = len(x)
        num = repr(self.traversal_copy).count('x1')
        
        # import pdb;pdb.set_trace()
        for i in range(1,dim):
            new_num = repr(self.traversal_copy).count(f'x{i+1}')
            if new_num!= num:
                return 0,[0],True, 'spatial_error', 'spatial_error'
        omit_terms = []
        num = 0
        for traversal in self.terms_token:
            
            if 'diff' in repr(traversal) and 'u,' not in repr(traversal):
                omit_terms.append(num)
                num+=1
                
                return 0,[0],True,'no_u','no_u'
            try:
                result, invalid, error_node, error_type = Program.execute_function(traversal, u, x)
            except:
                omit_terms.append(num)
                num+=1
                # continue
                return 0,[0], True, 'wrong_diff', 'wrong_diff'

            if invalid:
                return 0,[0],invalid,error_node,error_type
            
            num+=1
            results.append(result.reshape(-1))
        # empty results
        if len(results) ==0:
            invalid =True
            return 0,[0],invalid,'dim_error','dim_error'
        else:
            result_shape = results[0].shape
            for res in results[1:]:
                if res.shape!=result_shape:
                    invalid =True
                    return 0,[0],invalid,'dim_error','dim_error'
                
        results = np.array(results).T
        self.terms_token_new = [self.terms_token[i] for i in range(len(self.terms_token)) if i not in omit_terms]
        self.terms_new = [self.terms[i] for i in range(len(self.terms)) if i not in omit_terms]
        try:
            
            w_best = np.linalg.lstsq(results, ut)[0]
     
        except Exception as e:
            # print(e)
            # import pdb;pdb.set_trace()
            # invalid = True
            return 0, [0], True,"bad_str", 'bad_str'
        # import pdb;pdb.set_trace()
        y_hat = results.dot(w_best)
        w_best = w_best.reshape(-1).tolist()
        omit_terms = []
        # for i in range(len(w_best)):
            # if np.abs(w_best[i])<5e-5: #ac 5e-5
            #     # omit_terms.append(i)
            #     # continue
            #     return 0, [0],True, 'small_coef','small_coef'
        self.terms_token = [self.terms_token_new[i] for i in range(len(self.terms_token_new)) if i not in omit_terms]
        w_best_left = [w_best[i] for i in range(len(w_best)) if i not in omit_terms]
        self.terms = [self.terms_new[i] for i in range(len(self.terms_new)) if i not in omit_terms]
        return y_hat, w_best_left, False,error_node,error_type

    
    # @property
    # def terms(self,):
    #     return self.terms  

           



def convert_to_sympy(node):
    """Adjusts trees to only use node values supported by sympy"""

    if node.val == "div":
        node.val = "Mul"
        new_right = Node("Pow")
        new_right.children.append(node.children[1])
        new_right.children.append(Node("-1"))
        node.children[1] = new_right

    elif node.val == "sub":
        node.val = "Add"
        new_right = Node("Mul")
        new_right.children.append(node.children[1])
        new_right.children.append(Node("-1"))
        node.children[1] = new_right

    elif node.val == "inv":
        node.val = Node("Pow")
        node.children.append(Node("-1"))

    elif node.val == "neg":
        node.val = Node("Mul")
        node.children.append(Node("-1"))

    elif node.val == "n2":
        node.val = "Pow"
        node.children.append(Node("2"))

    elif node.val == "n3":
        node.val = "Pow"
        node.children.append(Node("3"))

    elif node.val == "n4":
        node.val = "Pow"
        node.children.append(Node("4"))

    for child in node.children:
        convert_to_sympy(child)

    return node
