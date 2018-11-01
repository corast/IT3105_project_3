# A game of nim consist of a couple of different things. 
# K : Numer of objects we can choose at a time.
# N : Number of total objects to choose from.
# Rules: player to choose last remaining stone(s) win.
#from base import State
from base.State import State
from base.Game import Game
import copy
import random
import math
import variables

class NIM_State(State):
    def __init__(self, num_pieces, player_turn=1, winner = None): # Keep track of state of game
        self.player_turn = player_turn # First player is always 1, unless otherwise specified
        self.num_pieces = num_pieces # Number of pieces left in game
        self.winner = winner # Store which player won a specific game. {1 or 2, -1 if tie}

    def is_game_over(self): 
        if(self.num_pieces == 0 and self.winner is not None): # If number of pieces == 0 and we have selected a winner from the state.
            return True
        return False
    
    def get_reward(self, player): # get reward from a state, only if we are terminal state
        #We need to check if we are the same player as winner.
        if(player == self.winner): # player, should be root node player.
                return 1 # Player 
        return 0 # We lost 

    def switch_turn(self): # How we switch states.
        if(self.player_turn == 1):
            self.player_turn = 2
        else:
            self.player_turn = 1

    def switch_turns_random(self):
        self.player_turn = random.choice([1,2]) # randomly choice between which player starts 

    def __str__(self): # 
        data = {"Player_turn": self.player_turn,"num_pieces":self.num_pieces,
        "winner":self.winner}
        return data.__str__()
    
class NIM(Game, NIM_State): # Expand Game and NIM_State
    def __init__(self, max_pieces, num_pieces, player_turn=1): # TODO: Fix which player is starting
        self.max_pieces = max_pieces # Max number of pieces that either player can take on their turn.
        self.state = NIM_State(num_pieces=num_pieces,player_turn=player_turn) # Init state for this game.
        print("Game of NIM params: {} number of pieces, {} max pieces each turn".format(self.state.num_pieces,self.max_pieces))
    # ** play on current state
    def play(self, K): # K is number of pieces we want to remove.
        # Return game ongoing or not.
        if(K > self.max_pieces): # if we want to pick more pieces than allowed.
            K = self.max_pieces

        self.state.num_pieces -= K

        if(self.state.num_pieces < 0 ): # Simply prevent negative pieces on the board.
            self.state.num_pieces = 0
        
        if(self.state.num_pieces <= 0): # If we pick the last number of stones, we win. 
            self.state.winner = self.state.player_turn # the winner of this NIM game.
            return self.state.winner # Return the winner, which is the player which picked the last stone(s)
        self.switch_turn() # Change which player is playing next
        return 0 # Return 0, meaning we are still running

    # ** Play on a copy of current state, return resulting game with new state.
    def play_state(self, action):
        # self is a game, so we need to init a game of Nim and try out action K
        # create a copy of the game.
        state = copy.deepcopy(self.state) # copy state of current game.
        state.num_pieces -= action # We draw K/action stones.

        #Need to check if we win or not in this new game state.
        if(state.num_pieces <= 0):
            state.winner = state.player_turn # Update state with the winner, that made previous move.
        state.switch_turn() # Pass turn to next player. 
        return self.game_state(state) # return the game state
    
    def game_state(self, state): #return a new game with the given state.
        game = copy.copy(self) # Make a copy of ourself
        game.state = state # Change the state in this game instance
        return game # return new game.

    # *** GET FUNCTIONS

    def get_actions(self):
        #Return legal actions for current player.
        #Based on which player we are looking at.
        #Returns actions as list
        if(self.state.num_pieces < self.max_pieces): #If we can't choice more pieces than whats left.
            return list(range(1, self.state.num_pieces+1))
        return list(range(1,self.max_pieces+1))

    def get_current_state(self):
        #return current board state as one number # number of pieces played in our case
        return self.state 

    def get_current_player(self):
        """ Return which player is at turn"""
        return self.state.player_turn 

    def get_winner(self):
        return self.state.winner 

    def get_prev_player(self):
        if(self.get_current_player() == 1):
            return 2
        return 1  

    def init_player_turn(self, start_player): # For the batch mode.
        if(start_player == 3):
            self.switch_turns_random()
        else:
            self.state.player_turn = start_player

    def switch_turn(self):
        self.state.switch_turn()

    def switch_turns_random(self):
        #change which turn it is with a random value.
        self.state.switch_turns_random()
    
    def display_turn(self, action):
        print("Player {} selects {} stones. Remaining stones = {}".format(self.get_prev_player(), action, self.state.num_pieces)) #
        if(self.is_game_over()):
            print("Player {} wins".format(self.get_prev_player()))

    def is_game_over(self): # Simply check if game is actually over
        return self.state.is_game_over()

