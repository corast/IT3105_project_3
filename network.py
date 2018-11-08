# Contain the "rollout"(Policy) network. Which will ultimatly decice which action we take from a given state.
import torch
import torch.nn as nn # neural netork modules
import torch.nn.functional as F # optimizer
import torch.optim as optim
import os
import numpy as np
from IPython.core.debugger import set_trace

torch.manual_seed(2809)
np.random.seed(2809)

class Module(nn.Module):

    def __init__(self,insize = 27,outsize = 25): # Define the network in detail
        super().__init__() # Dunno
        self.inL = nn.Linear(insize,40)
        self.drop1 = nn.Dropout(p=0.5)
        self.hL = nn.Linear(40,outsize)

        #self.outL = nn.Softmax(outsize) # output is only a board state.


    def forward(self, input):
        # set_trace() # debugging
        x = self.inL(input) # put input in input layer
        x = F.relu(x) # output activation.
        x = self.drop1(x)
        x = self.hL(x)
        x = F.relu(x)
        x = F.log_softmax(x, dim=1)
        return x # Return output, whatever it is.


def weights_init(model): # Will reset states if called again.
    if isinstance(model, nn.Linear):
        model.weights.data.fill_(1.0)
        model.bias.data.zero_() # Bias is set to

def train(train_loader, model, optimizer, loss_function,gpu=False):
    #train_loder is a training row,
    # Switch to training mode
    model.train()

    end = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    module.to(device) # Put model on the specified device.
    for i,(input, target) in enumerate(train_loader): # 
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

    optimizer.zero_grad()
    output = model(x)
    loss = loss_function(output,y)
    loss.backward()
    optimizer.step()
    

    return loss.data[0]

def save_checkpoint(state, filename="models/checkpoint.pth.tar"): #Save as .tar file
    torch.save(state, filename)

def load_checkpoint(model, optimizer, losslogger, filename="models/checkpoint.pth.tar"):
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['optimizer'])
        losslogger = checkpoint['losslogger']
        print("=> loaded checkpoint '{}' (epoch {})".format(filename,start_epoch))
    else:
        print(" => no checkpoint found at '{}'".format(filename))




checkpoint_path = "models/checkpoint.pth.tar"
model = Module(25*2 + 2,25)
model.apply(weights_init) # initialize weights to random, with bias set to 0.
optimizer = optim.Adam(model.parameters(),
                        lr=5e-4,betas=(0.9,0.999),eps=1e-08,weight_decay=1e-4)

loss_function = nn.NLLLoss()
#criterion = nn.CrossEntropyLoss()
save_checkpoint(state= {
    'epoch':epoch + 1, # We want to start from another epoch.
    'arch':arch,
    "state_dict":model.state_dict(),
    "optimizer":optimizer.state_dict(),
    },filename=checkpoint_path)


checkpoint = torch.load(checkpoint_path)

def store(self):
    #Store ourself in a file for later use
    pass



class module_handler():
    #This class is responsible for the rollout from the network.
    #It will be using the pre-trained neural network from different time instancess

    def __init__(self):
        pass

    def act(self):
        # Take an input and run it into the network, and output an action.
        pass