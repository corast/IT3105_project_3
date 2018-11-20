from base.State import State
from base.Game import Game
import copy
import numpy as np
import numpy.random as random
from math import cos, sin, sqrt, radians
from colorama import Fore, Style, Back, init
from clint.textui import colored
#from clint import coloured
import variables
import time

from misc import *
# TODO: Evaluate whether or not we need to update an neighbour state.
# Keith board. 1d array, of board. player_turn, R1, R2, R3 etc. 1 is p1, 2 is p2, 0 is empty. 
    # We need to store the whole board as an array. Example dim = 5
    #   (x,y) (rows, cols)
    #                      R (3)
    #            y     y     y     y     y
    #       x (0,0) (0,1) (0,2) (0,3) (0,4) x
    #       x (1,0) (1,1) (1,2) (1,3) (1,4) x
    # B (1) x (2,0) (2,1) (2,2) (2,3) (2,4) x 2 (B)
    #       x (3,0) (3,1) (3,2) (3,3) (3,4) x
    #       x (4,0) (4,1) (4,2) (4,3) (4,4) x
    #           y     y     y     y     y
    #                      R (4)
    # each index contains a cell, each cell, is either a goal state for a player, or a regular cell
    # Cells that contain a player is filled with 1 for player 1, or 2 for player 2, as an integer
    # Cells that contain 0 is empty, and in play
    # Black player (1) is trying to connect y = 0, to y = dim
    # Red player (2) is trying to connet x = 0, to x = dim
    # TODO: Merge self.connected with self.edge_v # Don't need two different ones.
class HEX_Cell():
    NONE = [False, False, False, False] # We are not an edge node.
    BLACK_L = [True,False,False,False] # 
    BLACK_R = [False,False,False,True] # 
    RED_T = [False,True,False,False] # 
    RED_B = [False,False,True,False] # 
    # It is impossible to be right and left or top and bottom at the same time.
    # But it is possible to be left/right and top/bot at same time.
    def __init__(self,x, y, edge = 0, edge_v = NONE):
        self.state = 0 # Tells wheter or not we have been visited before, and by which player.
        #self.edge = edge # Keep track of whether or not we are an edge cell, and which one.
        self.edge_v = edge_v # Keep track of what specific edge we are, if any (to spare us the time of searching)
        self.x = x # Keep track of our index
        self.y = y # Keep track of our index
        # We need to keep track of, if we are able to connect to an edge.
        self.connected = HEX_Cell.NONE
        self.neighbours = [] # List containing every neighbour of an cell. [(r,c-1), (r+1,y-1), (r+1,c);(r-1,c), (r-1,c+1), (r,c+1)]

    def is_edge(self):
        """Check if cell is an edge state. """ 
        if(self.edge_v.__eq__(HEX_Cell.NONE)):
            return False
        return True
    
    def is_edge_of_interest(self,player):
        # Check whether or not we are a connection node to an edge of interest.
        if(player == 1):
            return self.logical_or_and(HEX_Cell.BLACK_L, HEX_Cell.BLACK_R, self.edge_v)
        else:
            return self.logical_or_and(HEX_Cell.RED_B, HEX_Cell.RED_T, self.edge_v)
    
    def logical_or_and(self, a, b ,c):
        a = np.logical_or(a,b)
        a = np.logical_and(a,c) # Check whether or not we are atleast one of the edges of interest
        if(sum(a) == 0):
            return False
        return a.tolist()

    def is_clear(self): 
        """ Returns whether or not a board cell is free."""
        if(self.state == 0):
            return True
        return False

    def update_state(self,player): # Return True if we won.
        #We take in a player, number which we use to update our state.
        self.state = player
        edge = self.is_edge_of_interest(player)
        if(edge): # need to check if we are connected to one of the edges we are looking for
            self.connected = edge # set that we are connected to an edge of interest.

        self.update_connection(player) # Propagate victory.
        
        #Now we are update with highest connection based on every neighbour. 
        if(self.terminal_state()):
            return True # return that we won.
        #otherwise we need to update neighbours with a new state.
        
        for neighbour in [neigh for neigh in self.neighbours if neigh.state == player]: # Update all neighbrouring stats
            neighbour.update_neighbours(player,self.connected)
        # Now that we have updatet ourself, we need to check with the neighbours, find the one with the highest connections.
        
    def update_connection(self,player):
        # We update ourself, based on the neighbours connections
        # Find most connected neighbour, and set our state to the same

        #search neighbours for the one with the highest connection degree.
        for neighbour in [neigh for neigh in self.neighbours if neigh.state == player]: # Only look at set neighbour states.    
            self.connected = (np.logical_or(self.connected, neighbour.connected)).tolist() # Want to keep correct list type.

    # TODO: fix visited neighbours?
    def update_neighbours(self, player, connected): # Update our neighbours
        self.connected = connected
        for neighbour in [neigh for neigh in self.neighbours if neigh.state == player]:
            if(not neighbour.connected.__eq__(connected)):
                neighbour.update_neighbours(player, connected)

    def terminal_state(self):
        if(sum(self.connected) == 2):
            return True
        return False

    def __str__(self): # (edge, state)
        string = "({} {})".format(self.x, self.y)
        return string # Return this cells representation as string.
    
    def __repr__(self): # 
        string = "({})".format(self.state)
        return string # Return this cells representation as string.

    def __eq__(self, other):
        #An cell should be the same, if they have the same condinates.
        return self.x == other.x and self.y == other.y

class HEX_State(State):
    def __init__(self, dim, player_turn=1,board=None, winner = None,legal_cells = None): # Keep track of state of game
        self.player_turn = player_turn # First player is always 1, unless otherwise specified
        self.winner = winner # Store which player won a specific game. {1 or 2, -1 if tie}
        if(board is not None): # Means we start with a specific board state
            # TODO: handle difference in board input.
            self.board = board
            # TODO: legal states ...
            self.legal_cells = legal_cells
        else:
            self.board = self.createBoard(dim) # dim
            self.legal_cells = np.ones((dim,dim)).astype(int) # Simple array, to keep track of legal moves we can make from a given board. 
            #self.state_array = [0]*self.dimx*self.dimy
            self.state_array = np.zeros((self.dimx,self.dimy)).astype(int)
            #self.state_array = np.zeros((self.dimx*self.dimy*2)).astype(int).tolist()
            self.init_neighbours() # connect neighbouring cells. To easily search for complete path later.

    def createBoard(self, dim):
        """ Create the board, with dimentions specified along with a HEX_Cell as each element. """
        #dim can be a touple,list or an int.
        if (type(dim) == tuple or type(dim) == list): #we have a touple or list.
            self.dimx = dim[0]
            self.dimy = dim[1]
        else:
            self.dimx = dim
            self.dimy = dim
        board = np.zeros((self.dimx,self.dimy),dtype=HEX_Cell)
        #board = np.array()# [] # store board.
        for x in range(self.dimx):
            for y in range(self.dimy): # Create a cell, and put into the array.
                cell = None
                edge_v = [False, False, False, False]
                if(y == 0): # Left
                    edge_v[0] = True
                if(y == self.dimx-1): # Right
                    edge_v[3] = True
                if(x == 0): # Top
                    edge_v[1] = True
                if(x == self.dimy-1): # Bottom
                    edge_v[2] = True
                #Change value of element position to an cell
                board[x][y] = HEX_Cell(x,y,edge_v=edge_v)
        #print(type(board))
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

    def show_board(self):
        for row in self.board:
            for cell in row:
                print(" {} ".format(cell.state),end="")
            print("")

    def change_state(self, move): 
        """ Change state of board with an action to state."""# change the index, based on which player made the move
        #action is a touple or list (x,y)
        x = move[0]; y = move[1]
        if((self.board[x][y]).is_clear()): # check if cell is clear
            self.legal_cells[x][y] = 0 # We need to update legal states aswell.
            self.state_array[x][y] = self.player_turn # Set current state of cell equal to ourself.
            if(self.board[x][y].update_state(self.player_turn)): # Tell cell to update it's value.
                self.winner = self.player_turn # Set that someone won.
                #print("!!!!!!!!!!! PLAYER {} WON !!!!!!!!!!!!!!".format(self.player_turn))
            # * update state_array
            # State array is a simple array dimxdim array that keep track of game cons.
            
            #one_hot = int_to_binary_rev(self.board[x][y].state, 2) # Return the state as a binary array.
            #for v,value in enumerate(one_hot):
            #    index = x*self.dimx*2+y*2+v # [0,1,2] -> [0,0,1,1,2,2]
            #    self.state_array[index] = value

            self.switch_turn() # We are no longer in play.
        else:
            raise ValueError("Illegal action taken trying to puth piece in field ({},{})".format(x,y))
        
        #TODO: search the network, from the edge cells, exit state, to show finished game path.

    def get_legal_actions(self):
        #return list of all board states that have not been visited yet.
        #return states that have not been visited yet.
        #return self.legal_states # Should be as an 1d list too, but with index instead of 1/0 values
        legal_states = []
        for row in self.board:
            for cell in row:
                if(cell.is_clear()): # Every cell that is clear
                    legal_states.append((cell.x,cell.y))
        return legal_states
    
    def get_legal_actions_1d(self): # * to check with output of Neural Network. After the loss function. bc,we want to learn the rules too.
        #Return legal actions as a 1d list, each move as a and 0 or 1, depending on legality of move.
        return self.legal_cells.ravel()

    def get_board_1d(self):
        #return all cells as an array [R1,R2,R3,R4,R5]
        return self.board.ravel()
    
    def get_state_as_input_nn(self): # Return this state as it should be input to the Neural Network.
        # State + PID
        # How can we go from an array of dimxdim to dimxdimx2 + 2 ?
        # We need to create a dimxdimx2 array
        #state_array = [0 for x in range(self.dimx*self.dimy*2)]
        state_array = np.zeros((self.dimx*self.dimy*2)).astype(int).tolist() 
        for x,row in enumerate(self.board):
            for y,cell in enumerate(row):
                one_hot = int_to_binary_rev(cell.state, 2) # Return the state as a binary array.
                for v,value in enumerate(one_hot):
                    index = x*self.dimx*2+y*2+v # [0,1,2] -> [0,0,1,1,2,2]
                self.state_array[index] = value
        #PID =  int_to_binary_rev(self.player_turn, 2)
        #state_array.extend(PID)
        return state_array
    
    def get_state_as_input(self):
        return self.state_array.ravel().tolist() # flatten list before we send.

    def get_state_as_cnn_input(self): # turn board into: 5x5:[player 1][player 2] + [player id] 
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
        """Testing """
        self.player_turn = int(random.choice([1,2])) # randomly choice between which player starts 

    def __str__(self): # bla
        data = {"Player_turn": self.player_turn,
        "winner":self.winner}
        return data.__str__()

    def state_occupied(self, action):
        cell_state = self.board[action[0]][action[1]].state
        if(cell_state != 0): # 0 is clear state.
            return True
        return False

    def draw_board(self): # * Draw a board in the command line
        #This function is suppose to draw the game board.
        board_p = [[] for i in range((self.dimx-1)+self.dimx)]
        for x,row in enumerate(self.board):
            #brow = []
            for y,cell in enumerate(row):
                x_ = x+y
                #y_ = -x + y + self.dimy-1
                board_p[x_].insert(0,cell) # Add to row as first index, corresponding to output.

        # * With colour
        for x,row in enumerate(board_p):
                # TODO: COLOUR THE PATH GREEN THAT HAS WON.
            for y,cell in enumerate(row):
                # Figure out what colour the state should be in. 
                state = Fore.WHITE+str(cell.state)
                if(cell.state == 2):
                    state = colored.red(str(cell.state))
                elif(cell.state == 1):
                    state = colored.blue(str(cell.state))

                # if state is current piece, we can colour it yellow.
                # Green states should be the correct path.

                if(y == 0): # * If we are first element of the row
                    i_y = self.intent_index(x)
                    print("{0:>{i}}".format("",i=i_y*2),end="") # create indent spaces, 
                    if(y==len(row)-1): # If we are the only element ( top and bottom)
                        print(" {} ".format(state),end="")
                    else:
                        # We need to print a wall, if we are not the middle row.
                        # We know that dim-1 or i_y == 0 is the middle row
                        if(i_y == 0): # We are middle row in plot. No wall
                            string = " "
                        else:
                            # Need to check wheter we are a blue or red edge
                            if(cell.edge_v.__eq__(HEX_Cell.BLACK_L)):
                                string = colored.blue("/")
                            else:
                                string = colored.red("\\")
                        print("{}{}".format(string,state),end="")
                elif(y == len(row)-1): # * If we are the last element on the row
                    i_y = self.intent_index(x) 
                    if(i_y == 0): # We are middle row in plot. No wall
                        string = ""
                    else:
                        # * We need to check wheter or not we are a blue or red edge.
                        if((cell.edge_v).__eq__(HEX_Cell.RED_T)): # We need to print red if we are a red edge, or blue otherwise
                            string = colored.red("\\")
                        else:
                            string = colored.blue("/")
                    print("---{}{}".format(state,string),end="")
                else: # * Any element in between the rows.
                    print("---{}".format(state),end="")
            print("")

        # We want the board to look like this:
        #                       
        #                               1
        #                              / \
        #                            /0---1\
        #                            / \ / \
        #                          /0---1---2\
        #                          / \ / \ / \
        #                        /0---1---2---2\
        #                        / \ / \ / \ / \
        #                       1---1---0---2---2
        #                        \ / \ / \ / \ /   
        #                        \2---1---2---2/
        #                          \ / \ / \ /
        #                          \1---2---0/
        #                            \ / \ /
        #                            \1---0/
        #                              \ /  
        #                               1                    
        #                                       
        #                                                       
        #                                               
        # Number of indentations:
        #xpad = cos(rotation)*xi - sin(rotation)*yi
        #ypad = sin(rotation)*xi + cos(rotation)*yi
        # We need to expand the array with spare cells.
        #print(np.zeros(((self.dimx-1)+self.dimx, (self.dimy-1)+self.dimy)))

    # Calulate 45 degrees ø(rotation). [x',y'] = k * [[cos(ø),sin(ø)],[-sin(ø),cos(ø)]]*[x,y]
    # ø = pi/4 (45 degrees in radians with x axis SE), k = sqrt(2), since sin(ø) = cos(ø) = sqrt(2)/2 , if ø is pi/4
    # gives [x',y']=[[1,1],[-1,1]][x,y] = [x+y,-x+y], taking into account negative indexes w.r.t. x axis, and zero indexing. [x+y,-x+y+dim-1] 
    def rotate_index(self,x,y): 
        return x+y,-x+y+self.dimy-1

    def intent_index(self,x):
        return abs(self.dimy - (x+1))
        
# Player 1 is Red, player 2 is Blue.
class HEX(Game):
    def __init__(self, dim):
        self.state = HEX_State(dim=dim) # Init state for this game.
        if(variables.verbose >= variables.play):
            print("Game of HEX with dimentions {} x {}".format(dim,dim))

    # ** Play on a copy of current state, return resulting game with new state.
    def play_state(self, action): # Move, should be an index of where to put an block on the board. Assumes the index is empty state.
        # Then we change the state
        # copy game state
        state_copy = copy.deepcopy(self.state) # Copy roots game state.
        #print("play_state ",action)
        state_copy.change_state(action) # apply changes to state
        return self.game_state(state_copy)
        
    def game_state(self, state):
        game = copy.copy(self)
        game.state = state
        return game
        #Check if current player won based on this.

    def play(self, action): # ** Play on this state directly. No Copy.
        # We can make a change on the state.
        self.state.change_state(action)

        if(self.is_game_over()):
            return True

    
    # ** Get functions

    def get_actions(self): # Actions we can take is based on the current state.
        #Return legal actions for current player.
        #Based on which player we are looking at.
        return self.state.get_legal_actions()

    def get_legal_actions_bool(self):
        return self.state.get_legal_actions_1d()

    def get_winner(self):
        return self.state.winner

    def get_current_player(self):
        """ Return which player is at turn"""
        return self.state.player_turn 

    def get_current_state(self):
            #return current board state.
        return self.state 

    def get_prev_player(self):
        if(self.get_current_player() == 1):
            return 2
        return 1  

    def get_dimentions(self):
        return self.state.dimx,self.state.dimy # Returns 25x25 dimentions.

    # ** Check(is) functions

    def is_cell_empty(self, action):
        cell_state = self.state.board[action[0]][action[1]] 
        if(cell_state == 0):
            return True
        return False
        # check current game state after a play, if it was winning move or not.

    def is_game_over(self): # Simply check if game is actually over
        return self.state.is_game_over()

    # ** Misc functions

    def switch_turns_random(self):
        #change which turn it is with a random value.
        self.state.switch_turns_random()

    def init_player_turn(self, start_player): # For the batch mode.
        if(start_player == 3):
            self.switch_turns_random()
        else:
            self.state.player_turn = start_player

    def get_state_as_input(self):
        return self.state.get_state_as_input()

    # ** Visualization functions.

    def display_turn(self, action): # TODO: Fix display_turn for each game.
        print("Player {} selects board cell {}".format(self.get_prev_player(), action)) #
        if(self.is_game_over()):
            print("Player {} wins".format(self.get_prev_player()))
    
    def display_board(self):
        self.state.draw_board()
