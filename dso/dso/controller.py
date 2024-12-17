"""Controller used to generate distribution over hierarchical, variable-length objects."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from dso.memory import Batch
from dso.program import Program
from dso.prior import LengthConstraint

class StructureAwareLSTM(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 input_dim,
                 hidden_dim, 
                 cell_type = 'lstm',
                 initializer='zeros',
                 num_layers = 1,
                 attention = False
        ):
        super(StructureAwareLSTM, self).__init__()
        self.attention = attention
        if cell_type == 'lstm':
            self.cell = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers = num_layers, batch_first=True)
        elif cell_type == 'gru':
            self.cell = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        #else:
        #    raise NotImplementedError
        self.initializer = initializer
        self.apply_initializer()
        #self.attention = MonotonicAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def apply_initializer(self):
        init_func = self.make_initializer()
        for name, param in self.cell.named_parameters():
            if 'weight' in name:
                init_func(param)
            elif 'bias' in name:
                nn.init.zeros_(param) # Biases are typically initialized to zeros

    def make_initializer(self):
        # Return the correct initialization function based on the name
        if self.initializer == 'zeros':
            return lambda x:nn.init.zeros_(x)
        elif self.initializer == 'var_scale':
            return lambda x: nn.init.kaiming_uniform_(x, a =math.sqrt(5), mode='fan_avg') 
        #else:
        #    rasie NotImplementedError

    def forward(self, x, hidden_state = None):
        """
        x: Input tensor of shape (seq_len, batch_size, input_size)
        hidden_states: hidden state passed between layers (optional)
        """

        x = x.float()
        cell_out, hidden_state = self.cell(x, hidden_state)
        # Apply monotic attention
        #if self.attention:
        #    cell_out = self.attention(cell_out)
        output = self.fc(cell_out)

        return output.squeeze(), hidden_state

class Controller(object):
    """
    Recurrent neural network (RNN) controller used to generate expressions.

    Specifically, the RNN outputs a distribution over pre-order traversals of
    symbolic expression trees. It is trained using REINFORCE with baseline.

    Parameters
    ----------

    prior : dso.prior.JointPrior
        JointPrior object used to adjust probabilities during sampling.

    state_manager: dso.tf_state_manager.StateManager
        Object that handles the state features to be used

    summary : bool
        Write tensorboard summaries?

    debug : int
        Debug level, also used in learn(). 0: No debug. 1: Print shapes and
        number of parameters for each variable.

    cell : str
        Recurrent cell to use. Supports 'lstm' and 'gru'.

    num_layers : int
        Number of RNN layers.

    num_units : int or list of ints
        Number of RNN cell units in each of the RNN's layers. If int, the value
        is repeated for each layer.

    initiailizer : str
        Initializer for the recurrent cell. Supports 'zeros' and 'var_scale'.

    optimizer : str
        Optimizer to use. Supports 'adam', 'rmsprop', and 'sgd'.

    learning_rate : float
        Learning rate for optimizer.

    entropy_weight : float
        Coefficient for entropy bonus.

    entropy_gamma : float or None
        Gamma in entropy decay. None (or
        equivalently, 1.0) turns off entropy decay.

    pqt : bool
        Train with priority queue training (PQT)?

    pqt_k : int
        Size of priority queue.

    pqt_batch_size : int
        Size of batch to sample (with replacement) from priority queue.

    pqt_weight : float
        Coefficient for PQT loss function.

    pqt_use_pg : bool
        Use policy gradient loss when using PQT?

    max_length : int or None
        Maximum sequence length. This will be overridden if a LengthConstraint
        with a maximum length is part of the prior.
    """

    def __init__(self, prior, state_manager, debug=0, summary=False,
                 # RNN cell hyperparameters
                 cell='lstm',
                 num_layers=1,
                 num_units=32,
                 initializer='zeros',
                 # Optimizer hyperparameters
                 batch_size=500,
                 optimizer='adam',
                 learning_rate=0.001,
                 # Loss hyperparameters
                 entropy_weight=0.005,
                 entropy_gamma=1.0,
                 # PQT hyperparameters
                 pqt=False,
                 pqt_k=10,
                 pqt_batch_size=1,
                 pqt_weight=200.0,
                 pqt_use_pg=False,
                 # Other hyperparameters
                 max_length=30,
                 attention=False,
                 atten_len=10):

        self.prior = prior
        self.summary = summary
        self.n_objects = Program.n_objects
        self.baseline = 0
        self.batch_size = 0
        self.num_units = num_units
        self.num_layers = num_layers
        self.lib = Program.library
        # Find max_length from the LengthConstraint prior, if it exists
        # Both priors will never happen in the same experiment
        prior_max_length = None
        for single_prior in self.prior.priors:
            if isinstance(single_prior, LengthConstraint):
                if single_prior.max is not None:
                    prior_max_length = single_prior.max
                    self.max_length = prior_max_length
                break

        if prior_max_length is None:
            assert max_length is not None, "max_length must be specified if "\
                "there is no LengthConstraint."
            self.max_length = max_length
            print("WARNING: Maximum length not constrained. Sequences will "
                  "stop at {} and complete by repeating the first input "
                  "variable.".format(self.max_length))
        elif max_length is not None and max_length != self.max_length:
            print("WARNING: max_length ({}) will be overridden by value from "
                  "LengthConstraint ({}).".format(max_length, self.max_length))
        self.max_length *= self.n_objects
        max_length = self.max_length

        # Hyperparameters
        self.entropy_weight = entropy_weight #0.03
        self.pqt = pqt
        self.pqt_use_pg = pqt_use_pg
        self.pqt_k = pqt_k
        self.pqt_batch_size = pqt_batch_size
        self.pqt_weight = pqt_weight

        self.n_choices = self.lib.L
        # Entropy decay vector
        if entropy_gamma is None:
            entropy_gamma = 1.0
        self.entropy_gamma_decay = np.array([entropy_gamma**t for t in range(max_length)])
        # Build controller RNN


        self.task = Program.task
        
        self.initial_obs = self.task.reset_task(prior)
        self.state_manager = state_manager
        self.state_manager.setup_manager(self)
        # import pdb;pdb.set_trace()
        input_dim = self.state_manager.get_tensor_input(self.initial_obs.reshape(1,-1)).shape[-1]
        self.cell = StructureAwareLSTM(self.n_choices, input_dim, num_units,
                               cell_type=cell,
                               initializer=initializer,
                               num_layers=num_layers,
                               attention=attention
                               )
        # Get initial prior
        initial_prior = prior.initial_prior()
        self.initial_prior = torch.tensor(initial_prior, dtype=torch.float32)


        self.optimizer = optimizer
        self.learning_rate = learning_rate
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.cell.parameters(), lr = self.learning_rate)
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.cell.parameters(), lr = self.learning_rate)

    def loop_fn(self, time, cell_output, cell_state, loop_state):

            
        if cell_output is None: # time == 0
            finished = torch.zeros(self.batch_size, dtype=torch.bool)
            obs = self.initial_obs
            next_input = self.state_manager.get_tensor_input(obs)
            obs = torch.tensor(obs, dtype=torch.float32)
            h_0 = torch.zeros(1, self.batch_size, self.num_units)  # hidden state
            c_0 = torch.zeros(1, self.batch_size,self.num_units)  # cell state
            next_cell_state = (h_0, c_0)
            emit_output = None
            actions_ta = []
            obs_ta = []
            priors_ta = []
            prior = self.initial_prior
            lengths = torch.ones(self.batch_size)
            next_loop_state = (
                        actions_ta,
                        obs_ta,
                        priors_ta,
                        obs,
                        prior,
                        lengths, # Unused until implementing variable length
                        finished)
        else:
            actions_ta, obs_ta, priors_ta, obs, prior, lengths, finished = loop_state
            # import pdb;pdb.set_trace()
            logits = cell_output + prior
            next_cell_state = cell_state
            emit_output = logits
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)

                    # When implementing variable length:
                    # action = tf.where(
                    #     tf.logical_not(finished),
                    #     tf.multinomial(logits=logits, num_samples=1, output_dtype=tf.int32)[:, 0],
                    #     tf.zeros(shape=[self.batch_size], dtype=tf.int32))
            actions_ta.append(action) # Write chosen actions
                    # Get current action batch
            actions = torch.stack(actions_ta, dim=0).transpose(0,1) # Shape: (?, time)

                    # Compute obs and prior
            next_obs, next_prior = self.task.get_next_obs(actions.cpu().numpy(), obs.numpy())
            next_obs = torch.tensor(next_obs, dtype=torch.float32)
            next_prior = torch.tensor(next_prior, dtype = torch.float32)

            next_prior = next_prior.view(-1,self.lib.L)
            next_obs = next_obs.view(-1, self.task.OBS_DIM)
            next_obs = self.state_manager.process_state(next_obs)
            next_input = self.state_manager.get_tensor_input(next_obs)
            obs_ta.append(obs)
            priors_ta.append(prior)
            finished = next_finished = torch.logical_or(
                            finished,
                            torch.tensor(time >= self.max_length))
                    # When implementing variable length:
                    # finished = next_finished = tf.logical_or(tf.logical_or(
                    #     finished, # Already finished
                    #     next_dangling == 0), # Currently, this will be 0 not just the first time, but also at max_length
                    #     time >= max_length)
            next_lengths = torch.where(
                    finished, # Ever finished
                    lengths,
                    torch.tensor((time+1)).unsqueeze(0).repeat(self.batch_size))

            next_loop_state = (actions_ta,
                                    obs_ta,
                                    priors_ta,
                                    next_obs,
                                    next_prior,
                                    next_lengths,
                                    next_finished)

        return (finished, next_input, next_cell_state, emit_output, next_loop_state)


         # Generates dictionary containing placeholders needed for a batch of sequences
    def make_batch_ph(self, name):
        batch_ph = {
                "actions": torch.zeros(self.batch_size, self.max_length, dtype=torch.int32),
                "obs": torch.zeros(self.batch_size, self.task.OBS_DIM, self.max_length, dtype=torch.float32),
                "priors": torch.zeros(self.batch_size, self.max_length, self.n_choices, dtype=torch.float32) ,
                "lengths": torch.zeros(self.batch_size, dtype=torch.int32),
                "rewards": torch.zeros(self.batch_size, dtype=torch.float32),
                "on_policy": torch.zeros(self.batch_size, dtype=torch.int32)
        }
        batch_ph = Batch(**batch_ph)

        return batch_ph

    def safe_cross_entropy(self, p, logq, axis=-1):
        safe_logq = torch.where(p==0, torch.ones_like(logq), logq)
        return -torch.sum(p * safe_logq, dim=axis)

        # Generates tensor for neglogp of a given batch
    def make_neglogp_and_entropy(self, B):
                # import pdb;pdb.set_trace()
        inputs = self.state_manager.get_tensor_input(B.obs)
        logits, _ = self.cell(inputs.reshape(-1,inputs.shape[-2], inputs.shape[-1]))
        if len(logits.shape) == 2:
            logits = logits.unsqueeze(0)
        logits += B.priors
        probs = torch.softmax(logits,dim=-1)
        logprobs = F.log_softmax(logits,dim=-1)

                # Generate mask from sequence lengths
                # NOTE: Using this mask for neglogp and entropy actually does NOT
                # affect training because gradients are zero outside the lengths.
                # However, the mask makes tensorflow summaries accurate.

        mask = (torch.arange(self.max_length).unsqueeze(0) < B.lengths.unsqueeze(1)).float()
                # Negative log probabilities of sequences
        actions_one_hot = F.one_hot(B.actions.to(torch.int64), num_classes=self.n_choices).float()
                # import pdb;pdb.set_trace()
        neglogp_per_step = self.safe_cross_entropy(actions_one_hot, logprobs, axis=2) # Sum over action dim

        neglogp = torch.sum(neglogp_per_step * mask, dim=1) # Sum over time dim

                # NOTE 1: The above implementation is the same as the one below:
                # neglogp_per_step = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=actions)
                # neglogp = tf.reduce_sum(neglogp_per_step, axis=1) # Sum over time
                # NOTE 2: The above implementation is also the same as the one below, with a few caveats:
                #   Exactly equivalent when removing priors.
                #   Equivalent up to precision when including clipped prior.
                #   Crashes when prior is not clipped due to multiplying zero by -inf.
                # neglogp_per_step = -tf.nn.log_softmax(logits + tf.clip_by_value(priors, -2.4e38, 0)) * actions_one_hot
                # neglogp_per_step = tf.reduce_sum(neglogp_per_step, axis=2)
                # neglogp = tf.reduce_sum(neglogp_per_step, axis=1) # Sum over time

                # If entropy_gamma = 1, entropy_gamma_decay_mask == mask
        entropy_gamma_decay_mask = torch.tensor(self.entropy_gamma_decay) * mask # ->(batch_size, max_length)
        entropy_per_step = self.safe_cross_entropy(probs, logprobs, axis=2) # Sum over action dim -> (batch_size, max_length)
        entropy = torch.sum(entropy_per_step * entropy_gamma_decay_mask, dim=1) # Sum over time dim -> (batch_size, )
        return neglogp, entropy


    def debug(self, batch_size):
        self.batch_size = batch_size
                # Returns RNN emit outputs TensorArray (i.e. logits), final cell state, and final loop state
        self.initial_obs = self.task.reset_task(self.prior)
        self.initial_obs = np.tile(self.initial_obs, (self.batch_size, 1))
        self.initial_prior = self.initial_prior.expand(self.batch_size, self.n_choices)
        self.initial_cell_input = self.state_manager.get_tensor_input(self.initial_obs)
        time = 0
        finished, next_input, initial_state, emit_output, loop_state = self.loop_fn(time, cell_output=None, cell_state=None, loop_state=None)
        state = initial_state
        while not all(finished):
            # import pdb;pdb.set_trace()
            (output, cell_state) = self.cell(next_input, state)
            time = time + 1
            (finished, next_input, next_state, emit, loop_state) = self.loop_fn(time, cell_output=output, cell_state=cell_state, loop_state=loop_state)

        actions_ta, obs_ta, priors_ta, next_obs, next_prior, next_lengths, next_finished = loop_state
        self.actions = torch.stack(actions_ta).transpose(0, 1)  # Shape: (batch_size, max_length)
        self.obs = torch.stack(obs_ta).transpose(0, 2).transpose(0,1)  # Shape: (batch_size, obs_dim, max_length)
        self.priors = torch.stack(priors_ta).transpose(0, 1)  # Shape: (batch_size, max_length, n_choices)

        self.length = next_lengths  # Assuming next_lengths is already a tensor
        self.finished = next_finished  # Assuming next_finished is already a tensor
        return self.actions, self.obs, self.priors, self.length, self.finished


    def train_step(self, b,  sampled_batch, pqt_batch):
        self.baseline = b
        self.optimizer.zero_grad()
        neglogp, entropy = self.make_neglogp_and_entropy(sampled_batch)
        r = sampled_batch.rewards
        # Entropy loss
        entropy_loss = -self.entropy_weight * torch.mean(entropy)
        loss = entropy_loss

            # import pdb;pdb.set_trace()
        if not self.pqt or (self.pqt and self.pqt_use_pg):

            # Baseline is the worst of the current samples r
            pg_loss = torch.mean((r - self.baseline) * neglogp)
            # Loss already is set to entropy loss
            loss += pg_loss

        # Priority queue training loss
        if self.pqt:
            pqt_neglogp, _ = self.make_neglogp_and_entropy(pqt_batch)
            pqt_loss = self.pqt_weight * torch.mean(neglogp)
            loss += pqt_loss
        loss.backward()
        self.optimizer.step()
