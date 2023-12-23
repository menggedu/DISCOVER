try:
    from dso import cyfunc
except ImportError:
    cyfunc = None
import array
import torch.autograd as autograd
import torch
"""
evaluate PDE traversal through predefined operators and operands.
"""
def python_execute(traversal, u, x):
    apply_stack = []
    dim_flag = None
    i = 0
    
    for node in traversal:
        apply_stack.append([node])
        
        while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
            token = apply_stack[-1][0]
            terminals = apply_stack[-1][1:]
            
            if token.input_var is not None:
                # import pdb;pdb.set_trace()
                
                intermediate_result =  x[token.input_var] 
                dim_flag = token.input_var+1
            elif token.state_var is not None:
                intermediate_result = u[token.state_var]
            
            else:
                try:
                    if 'diff' in token.name or 'Diff' in token.name:
                        intermediate_result = token(*[*terminals,dim_flag])
                    elif 'lap' == token.name:
                        intermediate_result = token(*[*terminals, x])
                    else:
                        # import pdb;pdb.set_trace()
                        intermediate_result = token(*terminals)
                except Exception as e:
                    print(e)
                    print(f"illegal token {token.name} utilization")
                    import pdb;pdb.set_trace()
                # import pdb;pdb.set_trace()
            if len(apply_stack) != 1:
                apply_stack.pop()
                apply_stack[-1].append(intermediate_result)
            else:
                return intermediate_result  
         
def python_execute_torch(traversal, u, x):
    apply_stack = []
    dim_flag = None
    i=0
    # import pdb;pdb.set_trace()
    for node in traversal:
        apply_stack.append([node])
        i+=1
        while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
            token = apply_stack[-1][0]
            terminals = apply_stack[-1][1:]
           
            if token.input_var is not None:
                
                intermediate_result =  x[token.input_var] 
                dim_flag = token.input_var+1
                
            elif token.state_var is not None:
                if isinstance(u, list):
                    
                    intermediate_result = u[0][:,token.state_var:token.state_var+1]
                else:
                    
                    intermediate_result = u
            else:
                
                if 'diff' in token.name or 'Diff' in token.name:
                    
                    try:
                        # with autograd.detect_anomaly():
                        intermediate_result = token(*[*terminals])
                    except Exception as e:
                        # print(e)
                        return None
                      
                elif 'lap' in token.name:
                    intermediate_result = token(*[*terminals, x])
                else:  
                    intermediate_result = token(*terminals)
            try:       
                if torch.isnan(intermediate_result).all() or torch.isinf(intermediate_result).all():
                    # print("nan or inf")
                    return None
            except:
                import pdb;pdb.set_trace()
                
            if len(apply_stack) != 1:
                apply_stack.pop()
                apply_stack[-1].append(intermediate_result)
            else:
                return intermediate_result  
          
def python_execute_old(traversal, u, x):
    apply_stack = []
  
    for node in traversal:
        apply_stack.append([node])
       
        while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
            token = apply_stack[-1][0]
            terminals = apply_stack[-1][1:]
            
            if token.input_var is not None:
                
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


