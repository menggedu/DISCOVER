from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dso.program import Program


class StateManager(ABC):
    """
    An interface for handling the tf.Tensor inputs to the Controller.
    """

    def setup_manager(self, controller):
        """
        Function called inside the controller to perform the needed initializations (e.g., if the tf context is needed)
        :param controller the controller class
        """
        self.controller = controller
        self.max_length = controller.max_length

    @abstractmethod
    def get_tensor_input(self, obs):
        """
        Convert an observation from a Task into a Tesnor input for the
        Controller, e.g. by performing one-hot encoding or embedding lookup.

        Parameters
        ----------
        obs : np.ndarray (dtype=np.float32)
            Observation coming from the Task.

        Returns
        --------
        input_ : tf.Tensor (dtype=tf.float32)
            Tensor to be used as input to the Controller.
        """
        return

    def process_state(self, obs):
        """
        Entry point for adding information to the state tuple.
        If not overwritten, this functions does nothing
        """
        return obs


def make_state_manager(config):
    """
    Parameters
    ----------
    config : dict
        Parameters for this StateManager.

    Returns
    -------
    state_manager : StateManager
        The StateManager to be used by the Controller.
    """
    manager_dict = {
        "hierarchical": HierarchicalStateManager
    }

    if config is None:
        config = {}

    # Use HierarchicalStateManager by default
    manager_type = config.pop("type", "hierarchical")

    manager_class = manager_dict[manager_type]
    state_manager = manager_class(**config)

    return state_manager


class HierarchicalStateManager(StateManager):
    """
    Class that uses the previous action, parent, sibling, and/or dangling as
    observations.
    """

    def __init__(self, observe_parent=True, observe_sibling=True,
                 observe_action=False, observe_dangling=False, embedding=False,
                 embedding_size=8):
        """
        Parameters
        ----------
        observe_parent : bool
            Observe the parent of the Token being selected?

        observe_sibling : bool
            Observe the sibling of the Token being selected?

        observe_action : bool
            Observe the previously selected Token?

        observe_dangling : bool
            Observe the number of dangling nodes?

        embedding : bool
            Use embeddings for categorical inputs?

        embedding_size : int
            Size of embeddings for each categorical input if embedding=True.
        """
        self.observe_parent = observe_parent#true
        self.observe_sibling = observe_sibling#true
        self.observe_action = observe_action#false
        self.observe_dangling = observe_dangling# false
        self.library = Program.library

        # Parameter assertions/warnings
        assert self.observe_action + self.observe_parent + self.observe_sibling + self.observe_dangling > 0, \
            "Must include at least one observation."

        self.embedding = embedding
        self.embedding_size = embedding_size

    def setup_manager(self, controller):
        super().setup_manager(controller)
        # Create embeddings if needed
        # import pdb;pdb.set_trace()
        if self.embedding:
            self.initializer = nn.init.uniform_
            if self.observe_action:
                self.action_embeddings = nn.Parameter(
                    torch.empty(self.library.n_action_inputs, self.embedding_size)
                    )
                self.initializer(self.action_embeddings, a=-1.0, b=1.0)
            if self.observe_parent:
                self.parent_embeddings = nn.Parameter(
                    torch.empty(self.library.n_parent_inputs, self.embedding_size)
                    )
                self.initializer(self.parent_embeddings, a=-1.0, b=1.0)
            if self.observe_sibling:
                self.sibling_embeddings = nn.Parameter(
                    torch.empty(self.library.n_sibling_inputs, self.embedding_size)
                    )
                self.initializer(self.sibling_embeddings, a=-1.0, b=1.0)

    def get_tensor_input(self, obs):
        observations = []
        action, parent, sibling, dangling = np.split(obs, obs.shape[1], axis=1)
        # Cast action, parent, sibling to int for embedding_lookup or one_hot


        action = torch.Tensor(action).to(torch.int64)
        parent = torch.Tensor(parent).to(torch.int64)
        sibling = torch.Tensor(sibling).to(torch.int64)
        # Action, parent, and sibling inputs are either one-hot or embeddings
        if self.observe_action:
            if self.embedding:
                x = self.action_embeddings(action)
            else:
                x = F.one_hot(action, num_classes=self.library.n_action_inputs)
            observations.append(x)
        if self.observe_parent:
            if self.embedding:
                x = self.parent_embeddings(parent)
            else:
                # import pdb;pdb.set_trace()
                x = F.one_hot(parent, num_classes=self.library.n_parent_inputs)
            observations.append(x)
        if self.observe_sibling:
            if self.embedding:
                x = self.sibling_embeddings(sibling)
            else:
                x = F.one_hot(sibling, num_classes=self.library.n_sibling_inputs)
            observations.append(x)

        # Dangling input is just the value of dangling
        if self.observe_dangling:
            x = dangling.unsqueeze(-1)
            observations.append(x)
        input_ = torch.cat(observations, dim=-1)
        return input_
