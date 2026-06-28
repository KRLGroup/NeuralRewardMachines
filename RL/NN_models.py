import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .sequence_models.s4 import S4Block

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(),
        )
        
    def forward(self, x):

        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs)
        return dist, value
'''
class CNN_feature_extraction(nn.Module):

    def __init__(self, channels=3, output_dim=5, nodes_linear=4704):
        super().__init__()
        self.cnn_encoder = nn.Sequential(
             nn.Conv2d(channels, 3, 7, stride=2),
             nn.ReLU(inplace = False),
             nn.Conv2d(3, 6, kernel_size=7, stride=2),
             nn.ReLU(inplace = False),
             nn.Flatten()
        )
        self.fully_connected = nn.Sequential(
             nn.Linear(nodes_linear, output_dim)
        )

    def forward(self, x):
         x = self.cnn_encoder(x)
         return self.fully_connected(x)
    
'''
class Net(nn.Module):    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, kernel_size=5)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(125, 50)
        self.fc2 = nn.Linear(50, 16)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 3))
        # x = F.relu(F.max_pool2d(self.conv2_drop(x), 3))
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class RNN(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
    
    def forward(self, x, h, c):
        return self.rnn(x, (h, c))


class S4Encoder(nn.Module):
    """Sequence encoder that swaps the LSTM for an S4 (S4D/diagonal) layer.

    Used as a drop-in alternative to RNN in the A2C loop: it consumes one
    observation per timestep in recurrent (step) mode, carrying the SSM state
    across the episode. The S4 kernels are implemented in float32 while the rest
    of NRM runs in float64, so the module is kept single precision and the input
    and output are cast at the boundary.
    """

    def __init__(self, input_size, d_model, d_state):
        super(S4Encoder, self).__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_size, d_model)
        self.s4 = S4Block(d_model, mode='diag', transposed=False, d_state=d_state)
        self.activation = nn.GELU()

    def init_state(self, batch):
        # setup_step discretizes the SSM (dA, dB, dC) from the current parameters.
        # It must be recomputed every episode: the optimizer updates the underlying
        # parameters in place between episodes, which would otherwise invalidate the
        # cached step matrices for backprop.
        self.s4.setup_step()
        return self.s4.default_state(batch)

    def forward_step(self, x, state):
        # x: (batch, input_size) -> output: (batch, d_model)
        in_dtype = x.dtype
        x = self.input_proj(x.float())
        y, state = self.s4.step(x, state)
        y = self.activation(y)
        return y.to(in_dtype), state
