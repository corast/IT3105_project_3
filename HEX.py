from base.State import State
from base.Game import Game
import copy
import numpy as np
import numpy.random as random

    # We need to store the whole board as an array. Example dim = 5
    #   (x,y) (rows, cols)
    #                    R
    #        y     y     y     y     y
    #    x (0,0) (0,1) (0,2) (0,3) (0,4) x
    #    x (1,0) (1,1) (1,2) (1,3) (1,4) x
    # B  x (2,0) (2,1) (2,2) (2,3) (2,4) x  B
    #    x (3,0) (3,1) (3,2) (3,3) (3,4) x
    #    x (4,0) (4,1) (4,2) (4,3) (4,4) x
    #        y     y     y     y     y
    #                    R
    # each index contains a cell, each cell, is either a goal state for a player, or a regular cell
    # Cells that contain a player is filled with 1 for player 1, or 2 for player 2, as an integer
    # Cells that contain 0 is empty, and in play
    # Black player (1) is trying to connect y = 0, to y = dim
    # Red player (2) is trying to connet x = 0, to x = dim

class HEX_Cell():
    NONE = 0 # 0
    BLACK = 1 # 1
    RED = 2 # 10
    BOTH = 3 # 11
    def __init__(self,x, y, edge = 0):
        self.state = 0 # Tells wheter or not we have been visited before, and by which player.
        self.edge = edge # Keep track of whether or not we are an edge cell, and which one.
        self.x = x # Keep track of our index
        self.y = y # Keep track of our index
        self.neighbours = [] # List containing every neighbour of an cell. [(r,c-1), (r+1,y-1), (r+1,c);(r-1,c), (r-1,c+1), (r,c+1)]

    def is_edge(self):
        """Check if cell is an edge state. """
        if(self.edge != 0):
            return True
        return False
    def __str__(self): # (edge, state)
        string = "({} {})".format(self.x, self.y)
        return string # Return this cells representation as string.
    
    def __repr__(self):
        string = "{}({},{})".format(self.edge, self.x, self.y)
        return string # Return this cells representation as string.

    def search_path(self, current_player):
        pass

        #TODO: FIX infinite loop conditions.
    def search(self, current_player): #We allways start from left to right, or top to bottom with searching.
        #We need to check if we have a neighbour that we can visit atleast.
        if(self.state == current_player and self.is_edge()): # If we are at an end step, we can stop search.
            return True
        

        #continue searching children.
        set_neighbours = False
        for neighbour in self.neighbours:
            if(neighbour.state == current_player): # If state is set to current player
                set_neighbours = True
                neighbour.search(current_player)
        if(not set_neighbours):
            return False # We cant search more from here, or we fail.

class HEX_State(State):
    def __init__(self, dim, player_turn=1,board=None, winner = None): # Keep track of state of game
        self.player_turn = player_turn # First player is always 1, unless otherwise specified
        self.winner = winner # Store which player won a specific game. {1 or 2, -1 if tie}
        if(board is not None): # Means we start with a specific board state
            self.board = board
        else:
            self.board = self.createBoard(dim) # dim
            self.init_neighbours()
            # We need to set every neighbour

    def createBoard(self, dim):
        """ Create the board, with dimentions specified along with a HEX_Cell as each element. """
        #dim can be a touple,list or an int.
        if (type(dim) == tuple or type(dim) == list): #we have a touple or list.
            dimx = dim[0]
            dimy = dim[1]
        else:
            dimx = dim
            dimy = dim
        board = []
        # state board
        for x in range(dimx):
            row = [] # Row matrix.
            for y in range(dimy): # Create a cell, and put into the array.
                cell = None
                if(x == 0 and y == 0) or (x == dimx-1 and y == 0) or (x == 0 and y == dimy-1) or (x == dimx-1 and y == dimy-1):
                    cell = HEX_Cell(x,y,edge=3)
                elif(x == 0) or (x == dimx-1): # Edge states for Red
                    cell = HEX_Cell(x,y,edge=2)
                elif(y == 0) or (y == dimy-1):
                    cell = HEX_Cell(x,y,edge=1)
                else: # not edge cell.
                    cell = HEX_Cell(x,y,edge=0)
                if(cell is not None):
                    row.append(cell)
                else: # Catch error.
                    raise ValueError("A cell was not created for an index")
            board.append(row)
        return board
    
    def init_neighbours(self):
        """ Connects every cell to its neighbours. """
        dimx = len(self.board)
        dimy = len(self.board[0])
        for r,row in enumerate(self.board):
            for c,cell in enumerate(row):
                # We need to figure out what neighbours we should take from the board, based on this.
                # Each have 6 neighbours at most.
                # Create list of index, of potential neighbours.
                index_neighbours = [(r-1,c),(r-1,c+1),(r,c+1), (r,c-1),(r+1,c-1),(r+1,c)]
                if(cell.is_edge()):
                    #We need to remove indexes that are too large.
                    #Remove elements from touple which are smaller than 0 or greated than dimentions.
                    index_neighbours = [index for index in index_neighbours if ( 0 <= index[0] < dimx) and (0 <= index[1] < dimy)]
                # If the values are negative or greater than the dimention
                for index in index_neighbours:
                        cell.neighbours.append(self.board[index[0]][index[1]])

    def change_state(self, action): 
        """ Change state of board with an action to state."""# change the index, based on which made the move, assumes this is a legal action.
        #action is a touple or list (x,y)
        if((self.board[action[0],action[1]]).state == 0):
            self.board[action[0]][action[1]] = self.player_turn # Set state to current player.
        else:
            raise ValueError("Illegal action taken trying to puth piece in field ({},{})".format(action[0],action[1]))

    def check_state(self): # Check a game state, to vertify wether or not current_player made a winning move.
        #player 1 is playing Black pieces, whilst player 2 is playing Red pieces.
        if(self.player_turn == 1): # We want to check first column
            rows_of_interest = self.board[:,0] 
        else: # We need to check first row
            rows_of_interest = self.board[0]
        edge_cells = []
        present = False
        for cell in rows_of_interest:
            if(self.player_turn == cell.state): # simply check if we have a cell with state of player turn along edges.
                present = True
                edge_cells.append(cell) # Add edge cell.
        if(not present):
            return 0
        
        #TODO: search the network, from the edge cells, exit state sh



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
        self.player_turn = random.choice([1,2]) # randomly choice between which player starts 

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
        # One simple elimination test, is to check wheter or not there is one used block along the edges.
        

        pass


    def get_winner(self):
        return self.state.winner

    def state_empty(self, action):
        cell_state = self.state.board[action[0]][action[1]] 
        if(cell_state == 0):
            return True
        return False
        # check current game state after a play, if it was winning move or not.



hex = HEX(5)
print(hex.state)

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

    pass
    #print(state_2)

#state_test()