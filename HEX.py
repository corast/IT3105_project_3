from base.State import State
from base.Game import Game
import copy
import numpy as np
import numpy.random as random
class HEX_Cell():
    def __init__(self):
        self.state = 0 # Tells wheter or not we have been visited before.
        self.x_index = 0 # Keep track of our index
        self.y_index = 0 # Keep track of our index
        self.neighbours = [] # List containing every neighbour of an cell. [(r,c-1), (r+1,y-1), (r+1,c);(r-1,c), (r-1,c+1), (r,c+1)]

class HEX_State(State):
    def __init__(self, dim, player_turn=1,board=None, winner = None): # Keep track of state of game
        self.player_turn = player_turn # First player is always 1, unless otherwise specified
        self.winner = winner # Store which player won a specific game. {1 or 2, -1 if tie}
        if(board is not None): # Means we start with a specific board state
            self.board = board
        else:
            self.board = self.createBoard(dim) # dim

    def createBoard(self, dim):
        #dim can be a touple,list or an int.
        print(type(dim), dim)
        if (type(dim) == tuple or type(dim) == list):
            #we have a touple or list.
            dimx = dim[0]
            dimy = dim[1]
        else:
            dimx = dim
            dimy = dim
        #We need to store the whole board as an array. Example dim = 4
        #   (x,y) (rows, cols)
        #     y     y     y     y
        # x (0,0) (1,0) (2,0) (3,0) x
        # x (0,1) (1,1) (2,1) (3,1) x
        # x (0,2) (1,2) (2,2) (3,2) x
        # x (0,3) (1,3) (2,3) (3,3) x
        #     y     y     y     y
        # each index contains a cell, each cell, is either a goal state for a player, or a regular cell
        # Cells that contain a player is filled with 1 for player 1, or 2 for player 2, as an integer
        # Cells that contain 0 is 
        board = np.zeros((dimx, dimy))
        print(board)
        for x in range(dimx):
            for y in range(dimy):
                print("({},{})".format(x,y), end="")
            print()
        return board

    def change_state(self, action): # change the index, based on which made the move.
        self.board[action[0]][action[1]] = self.player_turn # Set state to current player.

    def check_state(self): # Check a game state, to vertify wether or not current_player made a winning move.
        # First test, check if current player has one piece on every row(player 1) or column(player 2), impossible to win without this.
        for row in self.board: # Check every row firstly
            for element in row:
                if(element == self.player_turn):
                    pass
            return 0 # Return 0, implying we didnt win    
        self.board

    def get_actions(self):
        #return list of all board states that have not been visited yet.
        #return states that have not been visited yet.
        pass

    def is_game_over(self): 
        if(self.winner is not None):
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
        self.player_turn = random.random.choice([1,2]) # randomly choice between which player starts 

    def __str__(self): # 
        data = {"Player_turn": self.player_turn,
        "winner":self.winner}
        return data.__str__()

#hex_state = HEX_State(dim=[4,4])

# Player 1 is Red, player 2 is Blue.
class HEX(Game):
    def __init__(self, dim):
        self.state = HEX_State(dim=dim) # Init state for this game.
        print("Game of HEX with dimentions {} x {}".format(dim,dim))


    def play_state(self, action): # Move, should be an index of where to put an block on the board. Assumes the index is empty state.
        if(not self.state_empty(action)):
            raise ValueError("Passed move {}, is not an acceptable action ".format(action))
        # Then we change the state
        # copy game state
        state_copy = copy.deepcopy(self.state)
        state_copy.change_state(action) # apply changes to state

        #Check if current player won based on this.
    
    def check_state(self):
        pass


    def get_winner(self):
        return self.state.winner

    def state_empty(self, action):
        cell_state = self.state.board[action[0]][action[1]] 
        if(cell_state == 0):
            return True
        return False
        # check current game state after a play, if it was winning move or not.



#hex = HEX(4)

def state_test():

    state_1 = np.zeros((3,3))
    state_1[1][0] = 1
    state_1[0][1] = 2
    state_1[2][1] = 1
    state_1[1][2] = 2
    state_1[2][2] = 1

    state_2 = np.zeros((3,3))
    state_2[0][0] = 1
    state_2[1][1] = 1
    state_2[2][2] = 1

    state_3 = np.zeros((4,4)) # Unconnected winning condition
    state_3[0][0] = 1
    state_3[0][1] = 1
    state_3[2][2] = 1
    state_3[3][3] = 1
    
    print(state_1)

    player_turn = 1 # It is player 1 turn to move
    #Want to check if player 1 won the game of hex or not.
    #We need to check the column for player 1
    #for col in state_1.shape()
    print(state_1.shape)
    if(player_turn == 2): # row
        axis_of_interest = 0
    else: # column
        axis_of_interest = 1
    for x in range(state_1.shape[axis_of_interest]-1):
        #get row, or get column
        if(axis_of_interest == 0):# We check rows
            states_0 = state_1[x]
            states_1 = state_1[x+1]
        else: # we check columns
            states_0 = state_1[:,x]
            states_1 = state_1[:,x+1] # Get next column as well

        #We got the states we are looking for.
        print(states_0," x:", x)
        print(states_1," x:", x)
        print("Next")


def check_paths(state ,x,y): # state is an array of next board state.
    # We simpy


    #print(state_2)

state_test()