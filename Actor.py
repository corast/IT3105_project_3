# We need an actor, so we can switch between agents to play the game,
# We want to be able to play as a human (command prompt), an Good AI vs an Bad AI.
# External AI (Interface), etc.

import network
import numpy as np
import misc
import torch
import torch.nn as nn
import HEX
from itertools import permutations 
import copy
import variables

#TODO Names
class Actor():
    def __init__(self, model=None, filepath=None, name="random"):

        if(model is None):
            #actor is simply random.
            self.network = False
            self.name = name
        else:
            self.name = model.name
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
            dim = np.sqrt(len(legal_moves)).astype(int)
            dims = (dim,dim)
            return np.unravel_index(np.random.choice(range(len(legal_moves)), p=normalize_legal_moves), dims=dims)

#ANET = Actor(model = network.Module(insize = 52, outsize = 25, name="network"),filepath = "models/testing_network_2000")
from base.Game import Game
def tournament(game:Game, models=[], random=False, games=10): # We need to load a model from path.
    actors = []
    for anet in models:
        actors.append(Actor(model=anet))
    if(random == True):
        actors.append(Actor()) # Random actor
    #Play tournament.
    # round robin tournament.
    # Every play against every one else.
    #games = len(actors)*len(actors)

    perms = permutations(actors,2) 
    results = []
    for first, second in perms:
        wins = [0,0]
        for i_g in range(games): # play x games
            t_game = copy.deepcopy(game) # copy game before we play.
            result = play_game(t_game, first, second)
            wins[result-1] += 1  # 1-1 = 0, 2-1 = 1
        results.append(wins)

    print("  First        Second   Score")
    for i,(first,second) in enumerate(permutations(actors,2)):
        print("{:^10} vs {:^10} {:^2}:{:^2}".format(first.name, second.name, results[i][0],results[i][1] ))


def play_game(game:Game,first:Actor, second:Actor):
    #create copy of game?
    # Need to switch between which actor is playing
    count = 0
    game_finished = game.get_winner()
    while game_finished is None: # play until game is over.
        board_state = game.get_state_as_input() # Get board state.
        PID = misc.int_to_binary_rev(game.get_current_player())
        legal_moves = game.get_legal_actions_bool()
        action = None

        if(game.get_current_player == 1):
            action = first.get_action(PID + board_state, legal_moves)
        else:
            action = second.get_action(PID + board_state,legal_moves)
        game_finished = game.play(action)
        if(variables.verbose >= variables.play):
            game.display_turn(action) # Display what happened to get to this state.
            game.display_board()
        #game_finished = game.get_winner()
        count += 1
    return game.get_winner()
