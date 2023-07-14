from dso.subroutines import jit_parents_siblings_at_once
from dso.gp import base as gp_base
from dso.program import Program, from_tokens
import dso.gp.utils as U


def reorganize(input_action):

    
    L = Program.library.L
    length = input_action.shape[0]
    max_length = input_action.shape[1]
    # Init all as numpy arrays of integers with default values or empty
    actions = np.empty((len(length), max_length), dtype=np.int32)
    obs_action = np.empty((len(length), max_length), dtype=np.int32)
    obs_parent = np.zeros((len(length), max_length), dtype=np.int32)
    obs_sibling = np.zeros((len(length), max_length), dtype=np.int32)
    obs_dangling = np.ones((len(length), max_length), dtype=np.int32)

    obs_action[:, 0] = L 
    programs = []

  
    for i, ind in enumerate(input_action):

        actions[i, :] = tokens
        obs_action[i, 1:] = tokens[:-1]
        obs_parent[i, :], obs_sibling[i, :] = jit_parents_siblings_at_once(np.expand_dims(tokens, axis=0),
                                                                            arities=Program.library.arities,
                                                                            parent_adjust=Program.library.parent_adjust)
        
        arities = np.array([Program.library.arities[t] for t in tokens])
        obs_dangling[i, :] = 1 + np.cumsum(arities - 1)

        programs.append(from_tokens(tokens, on_policy=False))

    # Compute priors
    if self.train_n > 0:
        priors = self.prior.at_once(actions, obs_parent, obs_sibling)
    else:
        priors = np.zeros((len(programs), self.max_length, L), dtype=np.float32)

    obs = np.stack([obs_action, obs_parent, obs_sibling, obs_dangling], axis=1)

    return programs, actions, obs, priors