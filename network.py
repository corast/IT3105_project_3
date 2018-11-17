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

class TestLoss():
    def __init__(self):
        super(self).__init__()

    def forward(self, input, target): # sum should be 1. and 
        # What is the loss between target and input.
        difference = abs(target - input)
        #loss = -torch.mean(torch.sum(torch.sum(torch.sum(target*torch.log(input), dim=1), dim=1), dim=1))
        return difference



class Model(nn.Module):

    def __init__(self,insize = 52,outsize = 25, name="network"): # Define the network in detail
        super().__init__() 
        self.outsize = outsize
        self.insize = insize
        self.name = name # want each module to have an unique name when we save to file.

        self.inL = nn.Linear(insize,30)
        #self.drop1 = nn.Dropout(p=0.2)
        self.hL1 = nn.Linear(30,30)
        #self.drop2 = nn.Dropout(p=0.2)
        self.hL2 = nn.Linear(30,outsize)
        #self.outL = nn.Softmax(outsize) # output is only a board state.

    def forward(self, input):
        # set_trace() # debugging
        x = self.inL(input) # put input in input layer
        #x = F.relu(x) # output activation.
        #x = self.drop1(x)
        #x = F.relu(x)
        x = self.hL1(x)
        #x = self.drop2(x)
        x = self.hL2(F.relu(x))
        x = F.softmax(x, dim=-1)
        #print("b",x)
        #x = F.sigmoid(x)
        #x = F.softmax(x,dim=0)  # 1d tensor
        #print("a",x)
        #x = F.log_softmax(x,dim=1) # softmax the final output
        return x # Return output, whatever it is.
    
    
    # Log_softmax -> Scale output between 0 and 1 + sum to 1
    #
    def backward(self):
        pass

    def evaluate(self, input):
        """ Use the network, as policy """
        self.eval() # Turn off training, etc.
        return self.forward(input)

    def store(self,epoch,optimizer,loss): # Need model, and optimizer
        #Store ourself in a file for later use
        print("Storing ourself to file...")
        save_path = "models/"+ self.name + "_" + str(epoch) # save a new network with an unique ID, name + epoch
        torch.save({'epoch':epoch,
                    'model_state_dict':self.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'loss':loss
                    },save_path)

    # params; input:dim*dim+2 
    def get_action(self, input, legal_moves): #should return an specific action.
        if(type(input) == list):
            #print("LIST IN GET_ACTION")
            input = torch.FloatTensor(input)
        elif(input is np.ndarray):
            print("INPUT IS ndarray")    
            input = torch.from_numpy(input).float() 
        #print(init)
        prediction =  self.evaluate(input) # * Return dim*dim vector.
        #Return argmax as touple.
        output = torch.Tensor.numpy(prediction.detach()) # convert to numpy array.
        #legal_moves = rollout_state.get_legal_actions_bool() # 1 is legal, 0 is illegal.
        dim = np.sqrt(len(legal_moves))
        dims = (dim,dim)
        actions = np.multiply(output, legal_moves) # Need to select the highest value from this one.
        action = np.unravel_index(np.argmax(actions),(5,5))
        return action # Return greedy action.

def train(model, casemanager_train:Datamanager, optimizer, 
        loss_function, batch=10, iterations=1, gpu=False, casemanager_test:Datamanager=None,verbose=1):
    #train_loder is a training row,
    start = time.time()
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #module.to(device) # Put model on the specified device. 

    loss_train = 0
    loss_test = 0
    print("Training network {}".format(model.name))
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


class NetWork(): # Hold an model, and coresponding optimizer.
    def __init__(self, optimizer, layers=[60,60],input_size=52, output_size = 25):
        self.optimizer = optimizer
        self.layer = layers
        self.input_size = input_size
        self.output_size = output_size

    
def load_model(path, model, optimizer=None):
    if os.path.isfile(path):
        model = model # TheModelClass(*args, **kwargs)
        optimizer = optimizer # TheOptimizerClass(*args, **kwargs)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if(not optimizer is None):  # Only if we want to keep training.
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, optimizer, loss, epoch # Return all this info
    else:
        print(" => no checkpoint found at '{}'".format(path))


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

def train_architecture_testing():
    torch.manual_seed(999) # set seeds
    #np.random.seed(999)
    #Load datamanager for both files.
    dataset_train = Datamanager.Datamanager("Data/data_random.csv",dim=5)
    dataset_test = Datamanager.Datamanager("Data/data_r_test.csv",dim=5)
    
    model = Model(insize=52, outsize=25, name="testing_network")
    print(model)
    # Create a model to train on.
    optimizer = optim.Adam(model.parameters(), lr=1e-2,betas=(0.9,0.999),eps=1e-6, weight_decay=1e-4)
    #optimizer  = optim.SGD(model.parameters(), lr=0.01,momentum=0.2, dampening=0.1)
    #optimizer = optim.RMSprop(model.parameters(), lr=0.05,alpha=0.99,eps=1e-8,weight_decay=0.1)
    #optimizer = optim.Adagrad(model.parameters(), lr=1e-3, lr_decay=0,weight_decay=0.1)

    # * Loss functions which multitargets
    #loss_function = pyloss.MultiLabelMarginLoss()
    #loss_function = pyloss.MSELoss()
    loss_function = pyloss.MSELoss(reduction='sum') # a bit better
    #loss_function = pyloss.L1Loss()
    #loss_function = pyloss.SmoothL1Loss()
    #loss_function = pyloss.NLLLoss2d()
    #loss_function = pyloss.MultiLabelSoftMarginLoss()
    #loss_function = CategoricalCrossEntropyLoss()

    loss_train, loss_T = train(model,batch=100, iterations=10000,
    casemanager_train=dataset_train,casemanager_test=dataset_test,optimizer = optimizer,loss_function=loss_function,verbose=10)
    model.store(epoch=10000, optimizer = optimizer, loss = loss_train)

#torch.cuda.manual_seed_all(999)

def test_network():
    #torch.manual_seed(999) # set seed
    #torch.cuda.manual_seed_all(999)
    #np.random.seed(2809)
    #torch.backends.cudnn.deterministic = True
    dataset_test = Datamanager.Datamanager("Data/data_r_test.csv",dim=5)
    dataset_train = Datamanager.Datamanager("Data/data_r_train.csv",dim=5)
    model = Model(dataset_test.inputs, dataset_test.outputs)
    model.apply(weights_init)

    #optimizer = optim.Adam(model.parameters(), lr=5e-4,betas=(0.9,0.999),eps=1e-08,weight_decay=1e-4)
    #optimizer  = optim.SGD(model.parameters(), lr=5e-4,momentum=0.01, dampening=0)
    optimizer = optim.RMSprop(model.parameters(), lr=0.005,alpha=0.99,eps=1e-8,weight_decay=0)
    #optimizer = optim.Adagrad(model.parameters(), lr=1e-2,lr_decay=0,weight_decay=0)

    #loss_function = nn.CrossEntropyLoss()
    #loss_function = nn.BCEWithLogitsLoss()
    #loss_function = nn.KLDivLoss()
    #loss_function = nn.NLLLoss2d
    #loss_function = nn.MultiLabelMarginLoss()
    loss_function = pyloss.MSELoss(reduction='sum')
    #train(casemanager_train=dataset_train,casemanager_test=dataset_test, model=model,optimizer=optimizer,loss_function=loss_function,iterations=200,batch=50)
    
    data_input, data_target = dataset_test.return_batch(1)

    output = model.evaluate(data_input)
    output = torch.Tensor.numpy(output.detach())
    print(output)
    #print(output.shape)
    print("sum outputs",np.sum(output) ," : ",output.tolist())
    legal_moves = np.array([[0,0,0,0,0],[0,1,0,1,1],[0,1,1,1,1],[0,0,0,0,0],[0,0,1,1,1]]) # 5x5
    legal_moves = legal_moves.reshape(1,25)
    #print(legal_moves)
    actions = np.multiply(output,legal_moves)
    #print(actions.shape)
    #print(actions.ravel().shape)
    action = actions.ravel().tolist() # id array

    def generate_network(settings):
        # dictionay should contain array of the network
        loss_function = settings.get("loss_function")
        optimizer = settings.optimizer # TODO: need to add nework params.
        layers = settings.get("layers")
#train_architecture_testing()

network_dict = {
    "layers":[40,30,60],
    "optimizer":optim.Adadelta,
    "optim_params":{"lr":1.0,"rho":0.9,"eps":1e-6},
    "loss_function":nn.MultiLabelMarginLoss(),
}