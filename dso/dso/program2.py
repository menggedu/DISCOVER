
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
            
            result, self.w = self.task.reward_functionSTR(self)
            
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

    def calculate(self,u,x,ut):
        results = []
        # sum over spatial coordinates
        dim = len(x)
        num = repr(self.traversal_copy).count('x1')
        for i in range(1,dim):
            new_num = repr(self.traversal_copy).count(f'x{i+1}')
            if new_num!= num:
                return 0,[0],True, 'spatial_error', 'spatial_error'
        omit_terms = []
        num = 0
        for traversal in self.terms_token:
            
            if 'diff' in repr(traversal) and 'u,' not in repr(traversal):
                # omit_terms.append(num)
                # num+=1
                # continue
                return 0,[0],True,'no_u','no_u'
            try:
                result, invalid, error_node, error_type = Program.execute_function(traversal, u, x)
            except:
                # omit_terms.append(num)
                # num+=1
                # continue
                return 0,[0], True, 'wrong_diff', 'wrong_diff'

            if invalid:
                return 0,[0],invalid,error_node,error_type
            
            num+=1
            results.append(result.reshape(-1))

        results = np.array(results).T
        self.terms_token_new = [self.terms_token[i] for i in range(len(self.terms_token)) if i not in omit_terms]
        self.terms_new = [self.terms[i] for i in range(len(self.terms)) if i not in omit_terms]
        try:
            
            w_best = np.linalg.lstsq(results, ut)[0]
     
        except Exception as e:
            
            invalid = True
            return 0, [0], True,"bad_str", 'bad_str'
        # import pdb;pdb.set_trace()
        y_hat = results.dot(w_best)
        w_best = w_best.reshape(-1).tolist()
        omit_terms = []
        for i in range(len(w_best)):
            if np.abs(w_best[i])<5e-5 or np.abs(w_best[i])>1e4: #ac 5e-5
                # omit_terms.append(i)
                # continue
                return 0, [0],True, 'small_coef','small_coef'
        self.terms_token = [self.terms_token_new[i] for i in range(len(self.terms_token_new)) if i not in omit_terms]
        w_best_left = [w_best[i] for i in range(len(w_best)) if i not in omit_terms]
        self.terms = [self.terms_new[i] for i in range(len(self.terms_new)) if i not in omit_terms]
        return y_hat, w_best_left, False,error_node,error_type