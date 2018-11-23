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
    def __init__(self, model:network.Model=None, filepath=None, name="random"):
            #Assumes an model is already preinstanciated with weights.
        self.model = model
        if(model is None): # * Random actor.    
            self.name = name
        else:
            self.name = model.name
            if(not filepath is None): # Load weights from file, otherwise 
                model.load_model(filepath)
            #else: # Randomly init weights for untrained network.
            #    model.apply(network.weights_init)
            # TODO: set action
            self.input_type = model.input_type

    def act(self, pid, board_state, legal_moves, dim): # board_state dim*dim+2
        if(self.model is not None): 
            # Board state should be in the format PID + board_states 26 array.
            if(self.input_type == 1):
                board_input = misc.get_normal_input(pid,board_state)
            elif(self.input_type == 2):
                board_input = misc.get_cnn_input(pid,[board_state],dim)
            elif(self.input_type == 3):
                board_input = misc.get_normal_2(pid, [board_state])

            if(type(board_input) == list):
                #print("LIST IN GET_ACTION")
                board_input = torch.FloatTensor(board_state)
            elif(type(board_input) == np.ndarray):
                #print("INPUT IS ndarray")    
                board_input = torch.from_numpy(board_state).float() 

            #Assumes it is an tensor.
            return self.model.get_action(board_input, legal_moves=legal_moves)

        # The actor make plays on an game, based on the game.
        # Should handle any game
        # This could either be MTC + ANN, or an interface (Keith's) or an player via terminal
        else:
            # Legal moves shape (1,dim*dim)
            # return random legal move.
            #np.unravel_index(np.argmax(actions),(5,5))
            normalize_legal_moves = misc.normalize_array(legal_moves) # Weights for moves. To remove illegal moves.
            dims = (dim,dim)
            return np.unravel_index(np.random.choice(range(len(legal_moves)), p=normalize_legal_moves), dims=dims) # return tuple

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
    games_per = (len(actors)-1)*2 # How many games each actor play, we don't play against ourself, and as first and second

    perms = permutations(actors,2) 
    results = []
    for first, second in perms:
        wins = [0,0]
        for i_g in range(games): # play x games
            t_game = copy.deepcopy(game) # copy game before we play.
            result = play_game(t_game, first, second)
            wins[result-1] += 1  # 1-1 = 0, 2-1 = 1
        results.append(wins)
    
    # TODO: create an matrix of the scores.
    win_list = [0 for i in range(len(actors))] 
    print("  First        Second   Score")
    for i,(first,second) in enumerate(permutations(actors,2)):
        print("{:^10} vs {:^10} {:^2}:{:^2}".format(first.name, second.name, results[i][0],results[i][1] ))
        # count the scores.
        index_first = actors.index(first)
        index_second = actors.index(second)
        win_list[index_first] += results[i][0]
        win_list[index_second] += results[i][1]
    print("Listings")
    for i,actor in enumerate(actors):
        print("{:^10} won {:>3} of {:>3}".format(actor.name,win_list[i],games_per))


    
    



def play_game(game:Game,first:Actor, second:Actor, input_type=1):
    #create copy of game?
    # Need to switch between which actor is playing
    count = 0
    game_finished = game.get_winner()
    while game_finished is None: # play until game is over.
        board_state = game.get_state_as_input() # Get board state. # simple dim*dim array with 0,1 and 2's
        PID = [game.get_current_player()]
        #PID = misc.int_to_binary_rev(game.get_current_player())
        legal_moves = game.get_legal_actions_bool()
        action = None

        if(game.get_current_player == 1):
            action = first.act(PID, board_state, legal_moves, dim=game.get_dimentions()[0])
        else:
            action = second.act(PID, board_state, legal_moves,dim=game.get_dimentions()[0])
        if(action is None):
            raise ValueError("Action is None")
        game_finished = game.play(action)
        if(variables.verbose >= variables.play):
            game.display_turn(action) # Display what happened to get to this state.
            game.display_board()
        #game_finished = game.get_winner()
        count += 1
    return game.get_winner()
