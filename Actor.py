# We need an actor, so we can switch between agents to play the game,
# We want to be able to play as a human (command prompt), an Good AI vs an Bad AI.
# External AI (Interface), etc.

import network
import numpy as np
import misc
import torch

class Actor():
    def __init__(self, model=None, filepath=None):
        if(model is None):
            #actor is simply random.
            self.network = False
        else:
            if(not filepath is None): # Load weights from file, otherwise 
                model.load_model(filepath)
            else: # Randomly init weights for untrained network.
                model.apply(network.weights_init)
            # TODO: set action
            self.network = True
            self.model = model 

    def get_action(self, board_state, legal_moves): # board_state dim*dim+2
        if(self.network):
            if(type(board_state) == list):
                #print("LIST IN GET_ACTION")
                board_state = torch.FloatTensor(board_state)
            elif(type(board_state) == np.ndarray):
                #print("INPUT IS ndarray")    
                board_state = torch.from_numpy(board_state).float() 
            #Assumes it is an tensor.
            return self.model.get_action(board_state, legal_moves)

        # The actor make plays on an game, based on the game.
        # Should handle any game
        # This could either be MTC + ANN, or an interface (Keith's) or an player via terminal
        else:
            # Legal moves shape (1,dim*dim)
            # return random legal move.
            #np.unravel_index(np.argmax(actions),(5,5))
            normalize_legal_moves = misc.normalize_array(legal_moves) # Weights for moves.
            dims = (np.sqrt(len(legal_moves)),np.sqrt(len(legal_moves)))
            return np.unravel_index(np.random.choice(range(len(legal_moves)), p=normalize_legal_moves), dims=dims)

#ANET = Actor(model = network.Module(insize = 52, outsize = 25, name="network"),filepath = "models/testing_network_2000")
from base.Game import Game
def tournament(game:Game, model=[], filepaths=[], random=False): # We need to load a model from path.
    actors = []
    for anet,path in model,filepaths:
        actors.append(Actor(model=anet,filepath=path))


    #Play tournament.
    # round robin tournament.


    # We want to create 
    pass