# Contain the "rollout"(Policy) network. Which will ultimatly decice which action we take from a given state.
import torch
import torch.nn as nn # neural netork modules
import torch.nn.functional as F # optimizer

from IPython.core.debugger import set_trace

class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3,1)

    def forward(self, X):
        # set_trace() # debugging
        x = self.lin(X)
        return x