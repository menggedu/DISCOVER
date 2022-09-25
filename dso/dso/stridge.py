import numpy as np


from dso.execute import python_execute

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
        # import pdb;pdb.set_trace()
        for traversal in self.terms_token:
            # import pdb;pdb.set_trace()
            if 'diff' in repr(traversal) and 'u' not in repr(traversal):
                return 0,[0],True,'no_u','no_u'
            # st = time.time()
            result, invalid, error_node, error_type = python_execute(traversal, u, x)
            # et = time.time()
            # print(f"{traversal} cost time :{et-st}")
            if invalid:
                return 0,[0],invalid,error_node,error_type
            results.append(result.reshape(-1))

        results = np.array(results).T
        try:
            
 
            w_best = np.linalg.lstsq(results, ut)[0]
   
        except Exception as e:
            # print(e)
            # import pdb;pdb.set_trace()
            invalid = True
            return 0, [0], invalid,"bad_str", 'bad_str'

        y_hat = results.dot(w_best)
        w_best = w_best.reshape(-1).tolist()
        for i in range(len(w_best)):
            if np.abs(w_best[i])<1e-5:
                return 0, [0],True, 'small_coef','small_coef'
        return y_hat, w_best, False,error_node,error_type

           