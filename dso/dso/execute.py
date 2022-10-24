try:
    from dso import cyfunc
except ImportError:
    cyfunc = None
import array


def python_execute(traversal, u, x):
    apply_stack = []
  
    for node in traversal:
        apply_stack.append([node])

        while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
            token = apply_stack[-1][0]
            terminals = apply_stack[-1][1:]

            if token.input_var is not None:
                # import pdb;pdb.set_trace()
                intermediate_result =  x[token.input_var]
            elif token.name ==  'u':
                intermediate_result = u
            else:

                intermediate_result = token(*terminals)
            if len(apply_stack) != 1:
                apply_stack.pop()
                apply_stack[-1].append(intermediate_result)
            else:
                return intermediate_result  

def cython_execute(traversal, X):
    """
    Execute cython function using given traversal over input X.

    Parameters
    ----------

    traversal : list
        A list of nodes representing the traversal over a Program.
    X : np.array
        The input values to execute the traversal over.

    Returns
    -------

    result : float
        The result of executing the traversal.
    """
    if len(traversal) > 1:
        is_input_var = array.array('i', [t.input_var is not None for t in traversal])
        return cyfunc.execute(X, len(traversal), traversal, is_input_var)
    else:
        return python_execute(traversal, X)


