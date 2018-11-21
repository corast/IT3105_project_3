# Contain the "rollout"(Policy) network. Which will ultimatly decice which action we take from a given state.
import torch
import torch.nn as nn # neural netork modules
import torch.nn.init as init
import torch.nn.functional as F # optimizer
import torch.optim as optim
import torch.nn.modules.loss as pyloss
from torch.nn.modules import Module
from torch.nn.functional import _Reduction
from  torch.nn.modules.loss import _Loss

import os
import numpy as np
from IPython.core.debugger import set_trace
import time
import Datamanager

import misc

#class _Loss(Module):
#    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean'):
#        super(_Loss, self).__init__()
#        if size_average is not None or reduce is not None:
#            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
#        else:
#            self.reduction = reduction



class MultiClassCrossEntropyLoss(_Loss):
    def forward(self, input, target):
        loss = -torch.mean(torch.sum(torch.sum(torch.sum(target*torch.log(input), dim=-1), dim=-1), dim=-1))
        return loss

class CategoricalCrossEntropyLoss(_Loss):

    def forward(self, input, target):
        #print("Categorical Cross Entropy", input.shape, target.shape)
        loss = target*torch.log(input) + (1-target)*torch.log(1-input)
        loss = -torch.mean(torch.sum(torch.sum(loss,dim=-1),dim=-1),dim=-1)
        #loss = -torch.mean(torch.sum(torch.sum(torch.sum(target*torch.log(input), dim=-1), dim=-1), dim=-1))
        return loss

class RootMeanSquareLoss(_Loss):
    def forward(self, input, target):
        return torch.sqrt(F.mse_loss(input, target))

def NETWORK_TEST():
    model = Model(nn.Linear(52, 40),nn.ReLU(),nn.Linear(40,25), nn.Softmax(dim=-1),name="netName", filepath="Testing")
    #nn.Linear(input_dim, 40),nn.ReLU(),nn.Linear(40,target_dim), nn.Softmax(dim=-1)) 
    print(model)

class Flatten(nn.Module):
    def forward(self, x):
        #print(x.shape)
        return x.view(x.size()[0], -1)

class Model(nn.Sequential):
    def __init__(self, *args,name="Network", filepath=None, input_type=1):
        #print("args",args)
        super().__init__(*args) # Pass rest of network to parent, to create the network with Sequential.
        self.name = name
        self.input_type = input_type
        if(filepath is not None): # weights from filepath.
            self.load_model(filepath)
        

    def evaluate(self, input):
        """ Use the network, as policy """
        self.eval() # Turn off training, etc.
        return self(input) # Should use the paren's forward function.

    def store(self,epoch,optimizer,loss, datapath=""): # Need model, and optimizer
        #Store ourself in a file for later use
        save_dir = "models/"+self.name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = save_dir+"/"+ self.name + "_" + str(epoch) # save a new network with an unique ID, name + epoch
        
        # Need to check if directory exists

        torch.save({'epoch':epoch,
                    'model_state_dict':self.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'loss':loss,
                    'datapath':datapath,
                    },save_path)

    def load_model(self, path, optimizer=None):
        if os.path.isfile(path):
            #model = model # TheModelClass(*args, **kwargs)
            optimizer = optimizer # TheOptimizerClass(*args, **kwargs)
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model_state_dict'])
            
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            if(optimizer is None):  # Only if we want to keep training.
                return loss, epoch
            else:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return loss, epoch # Return all this info
        else:
            print(" => no checkpoint found at '{}'".format(path))

    
    # params; input:dim*dim+2 [board_state + PID] in binary reversed.
    def get_action(self, input, legal_moves): #should return an specific action.
        if(not (type(input) == torch.Tensor)):
            print(type(input),"is not a tensor")
        prediction =  self.evaluate(input) # * Return dim*dim vector.
        #Return argmax as touple.
        output = torch.Tensor.numpy(prediction.detach()) # convert to numpy array.
        #legal_moves = rollout_state.get_legal_actions_bool() # 1 is legal, 0 is illegal.
        dim = np.sqrt(len(legal_moves)).astype(int)

        dims = (dim,dim)
        actions = np.multiply(output, legal_moves) # Need to select the highest value from this one.

        zeros = np.count_nonzero(actions)
        if(zeros == 0):
            print("We don't know what to do ...")
            normalize_legal_moves = misc.normalize_array(legal_moves) # Weights for moves.
            dims = (np.sqrt(len(legal_moves)),np.sqrt(len(legal_moves)))
            return np.unravel_index(np.random.choice(range(len(legal_moves)), p=normalize_legal_moves), dims=dims)

        action = np.unravel_index(np.argmax(actions),dims)
        return action # Return greedy action.

def train(model, casemanager_train:Datamanager, optimizer, 
        loss_function, batch=10, iterations=1, gpu=False, casemanager_test:Datamanager=None,verbose=1):
    #train_loder is a training row,
    start = time.time()
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #module.to(device) # Put model on the specified device. 

    loss_train = 0
    loss_test = 0
    #print("Training network {}".format(model.name))
    if(casemanager_test is None): # * If we want to evaluate against another dataset, different from train.
        # We only train and show loss from this.
        
        for t in range(1,iterations+1): #Itterade dataset x times with batch_size("all") if epochs.
            
            #loss_test = evaluate(casemanager_test, model=model, loss_function=loss_function)
            loss_train_i = train_batch(casemanager_train, 
            model=model, optimizer=optimizer, loss_function=loss_function, batch=batch)
            if(t % verbose == 0 or t == iterations + 1):
                print("itteration {} loss_train: {:.8f}".format(t, loss_train_i))
            loss_train += loss_train_i
        return loss_train/iterations # * average loss
    else:
        for t in range(1,iterations+1): #Itterade dataset x times with batch_size("all") if epochs.
            #loss_test = evaluate(casemanager_test, model=model, loss_function=loss_function)
            loss_train_i = train_batch(casemanager_train, 
            model=model, optimizer=optimizer, loss_function=loss_function, batch=batch)
            loss_test_i = evaluate_test(casemanager_test, 
            model=model, loss_function=loss_function)
            if(t % verbose == 0 or t == iterations + 1):
                print("itteration {}  loss_train: {:.8f} loss_test: {:.8f} ".format(t,loss_train_i, loss_test_i))
            loss_train += loss_train_i ; loss_test += loss_test_i
        return loss_train/iterations, loss_test/iterations # * Return average loss of both.

def weights_init(model): # Will reset states if called again.
    if isinstance(model, nn.Linear):
        init.xavier_uniform_(model.weight) # good init with relus.
        #model.weights.data.fill_(1.0)
        init.constant_(model.bias,0.01) # good init with relus, want every bias to contribute.
        #model.bias.data.zero_() # Bias is set to all zeros.

def train_batch(casemanager:Datamanager, model, optimizer, loss_function, batch):
    # Switch to training mode
    model.train()
    x,y = casemanager.return_batch(batch)# 10 training cases
    y_pred = model(x)
    loss = loss_function(y_pred,y) 
    #print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_test(casemanager:Datamanager, model, loss_function):
    model.eval() # Change behaviour of some layers, like no dropout etc.
    with torch.no_grad(): # Turn off gradient calculation requirements, faster.
        data,target = casemanager.return_batch("all")
        prediction = model(data)
        return loss_function(prediction,target).item() # Get loss value.

def save_checkpoint(state, filename="models/checkpoint.pth.tar"): #Save as .tar file
    torch.save(state, filename)

"""
checkpoint_path = "models/checkpoint.pth.tar"
model = Module(25*2 + 2,25)
model.apply(weights_init) # initialize weights to random, with bias set to 0.
optimizer = optim.Adam(model.parameters(),
                        lr=5e-4,betas=(0.9,0.999),eps=1e-08,weight_decay=1e-4)
"""
#loss_function = nn.NLLLoss()
#criterion = nn.CrossEntropyLoss()

#checkpoint = torch.load(checkpoint_path)
def HEX_CNN(name, dim):
    input_dim = (dim*dim*2)+2
    target_dim = dim*dim 
    return Model(                                    # O = (5-3 +2) +1 = 5 
        nn.Conv2d(3,3,kernel_size=(3,3),stride=1,padding=1), # -> 3*5*5 = 75 # 1,3,5,5 output
        nn.ReLU(),
        #nn.MaxPool2d(kernel_size=(3,3),padding=1,stride=1), # -> 1,3,5,5
        Flatten(),
        nn.Linear(75, 25),
        nn.Softmax(dim=-1), name=name,input_type=2)

def train_architecture_testing():
    torch.manual_seed(999) # set seeds
    #np.random.seed(999)
    #Load datamanager for both files.
    dataset_train = Datamanager.Datamanager("Data/random_fix.csv",dim=5,modus=2)
    #dataset_test = Datamanager.Datamanager("Data/data_r_test.csv",dim=5)
    model = HEX_CNN(name="CNNET_TEST", dim=5)
    #model = Model(nn.Linear(52,80), nn.ReLU(), nn.Linear(80,25), nn.Softmax(dim=-1), name="rms_mod")
    #model =  Model(nn.Conv1d(50,52,kernel_size=4,stride=1,padding=1),nn.ReLU(),nn.MaxPool1d(kernel_size=4, stride=2, padding=1), nn.Linear(50,25) ,nn.Softmax(dim=-1), name="rms_mod")
    # Dimentions: 52 -> 100, maxpol: 100 -> 
    #print(model)
    #exit()
    # Create a model to train on.
    optimizer = optim.Adam(model.parameters(), lr=1e-3,betas=(0.9,0.999),eps=1e-6) # 0.14, 0.18, 2: 0.10 ,0.133
    #optimizer  = optim.SGD(model.parameters(), lr=0.01,momentum=0.2, dampening=0) 4 ...
    #optimizer = optim.RMSprop(model.parameters(), lr=0.005,alpha=0.99,eps=1e-8) # 0.10 , 0.12 test
    #optimizer = optim.Adagrad(model.parameters(), lr=1e-2, lr_decay=0,weight_decay=0) # 0.40 (0.45 train) 0.65 test
    #print(optimizer)
    

    # * Loss functions which multitargets
    #loss_function = pyloss.MSELoss()
    loss_function = pyloss.MSELoss(reduction='sum') # a bit better
    #loss_function = pyloss.L1Loss()
    #loss_function = pyloss.SmoothL1Loss()
    #loss_function = pyloss.NLLLoss2d()
    #loss_function = pyloss.MultiLabelSoftMarginLoss()
    #loss_function = CategoricalCrossEntropyLoss()
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5)
    for itt in range(500):
        loss_train = train(model,batch=50, iterations=10,
        casemanager_train=dataset_train, optimizer = optimizer, loss_function=loss_function, verbose=200)
        #scheduler.step(loss_train)
        print("itteration {}  loss_train: {:.8f} lr:{}".format(itt, loss_train, optimizer.param_groups[0]["lr"]))
        #print(optimizer["lr"])
    model.store(epoch=10000, optimizer = optimizer, loss = loss_train)

#train_architecture_testing()
