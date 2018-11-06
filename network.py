# Contain the "rollout"(Policy) network. Which will ultimatly decice which action we take from a given state.
import torch
import torch.nn as nn # neural netork modules
import torch.nn.functional as F # optimizer

from IPython.core.debugger import set_trace

class Module(nn.Module):
    def __init__(self): # Define the network in detail
        super().__init__()
        self.inL = nn.Linear(25+2,40)
        self.outL = nn.Linear(40,25) # output is only a board state.


    def forward(self, input):
        # set_trace() # debugging
        x = self.inL(input) # put input in input layer
        x = self.outL(x)

        return x # Return output, whatever it is.



class policy():
    #This class is responsible for the rollout from the network.
    #It will be using the pre-trained neural network from different time instancess

    def __init__(self):
        pass

    def act(self):
        # Take an input and run it into the network, and output an action.

        pass

# The network should have 5x5 inputs + an player id (one_hot vector)