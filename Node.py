# Statemanager.
# Interface between game and MCTS.
# For assigment 3, we want to try a different game, must be able to handle this change.
import copy # Make copies of game states
import numpy as np
import variables
from base.State import *
from base.Game import *
import Datamanager

class Node():
    def __init__(self, game:Game, parent=None, action=None, node_depth=0):
        self.game = game # How we move from one state to the next, by playing at a specific state.
        self.parent = parent # Parent is None for root node.
        self.children = [] # Cointains the different choices we have at each stage.
        #self.state = game.state # state is state for game.
        self.action = action # Action it took to get to this state (Previous move).
        self.num_visits = 0 # Number of visits to this state.
        self.wins = 0 # Number of wins with specific state, from previous player perspective.
        self.node_depth = node_depth # We increase this number every time we add to tree.
        self.score = 0 # Inital score, UBC1 value in our case.
        
    def get_actions(self):
        """ Return a list of possible actions to take from a state S """
        return self.game.get_actions() # Return possible actions we can take from a game state.

    # **** SCORE FUNCTIONS

    def get_score(self, child, c=2): 
        child.score = self.UCB1(child, c) # Update score for an child
        return child.score
    
    def UCB1(self, child, c): # From wikipedia article Exploration/Explotation.
        #TODO: check we are the root node of the tree or not. If we are not calculate different score based on minimizing wins.
        if(variables.verbose >= variables.debug_all):
            print(self.num_visits ,"child.Q",child.Q(),self.num_visits,"/",child.num_visits, "U:",np.sqrt(c*np.log(self.num_visits)/child.num_visits))
        return child.Q() + np.sqrt(c*np.log(self.num_visits)/child.num_visits)
                # explotation + exploration(UCT) : components

    def Q(self): # Return Q value from current node.
        #wins = self.wins
        #loses = wins - self.num # Number of wins along this route - Number of visits.
        return self.wins/self.num_visits # return w_i / n_i 

    # ****** SELECT FUNCTION

    def best_child(self, root_player, c=2, action=False, data=False): # Return best child from a parent node 
        # need to check children to self, and pick the action which has the best value / Visits.
        # Must handle children that has not been visited yet
        # create a list of every child which has not been explored
        # * If data=True, we need to return the number of visits to each action state as well.
        """
        #TODO: What to do when we simulate less than child states, and want to make a move?
        if(action): # we need to choose only from the explored children
            explored = [child for child in self.children if child.num_visits == 0]
            choices = [self.get_score(child) for child in explored]
            return self.children[np.argmax(choices)]
        """
        #TODO: handle creating a case from the child values. Number of visits, when requested by the actual game. 

        if(data): # * If we want to return important information about the child states values, for creating a training case.
            # We should return an array with 25x25 values.
            dimention = self.game.get_dimentions() # Should return two dimentions, x and y.
            data_visits = np.zeros((dimention[0]*dimention[1])).astype(int).tolist() # create array to keep states.
            for child in self.children:
                # We need to get the action, and use it to update the array.
                action = child.action # Should be a touple for HEX.
                visits = child.num_visits # What we want to store.
                data_visits[action[0]*dimention[0]+action[1]] = visits
            print(data_visits)
            #print(self.game.get_legal_actions_bool()) 

        # * If we have one unexplored node left.
        unexplored = [child for child in self.children if child.num_visits == 0]
        if(len(unexplored) != 0):
            return np.random.choice(unexplored) # Randomly choose from unvisited nodes.
        # * Select child with best score
        choices = [self.get_score(child, c) for child in self.children] # Get score from each child node.
        return self.children[np.argmax(choices)] # Select index of best child.
        #return self.children[np.argmin(choices)] # else we want to minimize winning (i.e. we are not rewarded from winning)    

    # ***** EXPAND FUNCTIONS

    def init_children(self,verbos=0): # connect every possible child to its parent. Expand whole parent.
        actions = self.game.get_actions() # Get actions we can take from a given state in a game.
        #self is the parent we want to add the children to.
        #print("init_children", actions)
        for action in actions: # for each action, we need to play and add the corresponding state node to parent
            self.expand(action)

    def expand(self, action): # Expand a parent with an action, generating an child.
        game = self.game.play_state(action) # returns the game with new state from doing this action
        node = Node(game=game, action=action, parent=self, node_depth=self.node_depth+1) # New node should be a result of taking one action from parent
        self.children.append(node) # Add child to parent node

    # ***** ROLLOUT FUNCTIONS
    def rollout_policy(self, actions): # Allows us to change policy if needed
        choice = np.random.choice(len(actions)) #randomly choice one of the indexes
        return actions[choice] #return choice, from our action list based on index
    
    def rollout(self, root_player): # How we traverse the rest of the tree to terminal
        rollout_state = copy.deepcopy(self.game) # copy game state, since we don't want to keep states from this point onward.
        game_finished = self.is_termal_node()
        while not game_finished: # 
            possible_moves = rollout_state.get_actions() # Return legal actions.
            #print("possible moves", possible_moves, rollout_state.state.num_pieces)
            action = self.rollout_policy(possible_moves) # Choice one of legal actions, based on a few criteria.
            game_finished = rollout_state.play(action) # Do this action to game state.
        #Return reward based on if we won or not, -1 if we lost, 1 if we won.
        return rollout_state.get_winner() # * Return which player won, and give wins accordingly.
        #return rollout_state.get_reward(root_player),rollout_state.get_winner()# We need to check if current player from root won. (who made the play to leaf)

    # *** BACKPROGAPAGE FUNCTION

    def backpropagate(self, winner, root_player): # Must keep track of the reward for each step. 
        self.num_visits += 1 # Store numer of visits.
        if self.is_root(): #if we root of tree, we return, as root state is not something we can visit again anyway.
            return # We are done.
        # wins is is based on previous player, in this state.
        if(self.parent.game.get_current_player() == winner): # If winner is the same that made the move.
            self.wins += 1
            # ? Old code, might be usefull someday
            #if(self.parent.game.get_current_player() == root_player):# maximizer
            #    self.wins += 1
            #else: # minimizer
            #    self.wins += 1 
        self.parent.backpropagate(winner, root_player) # The reward from the state, should backprogogate back up root    

    # ***** CHECK FUNCTIONS

    def is_termal_node(self): # Check if we are a leaf node
        return self.game.is_game_over() # We check state if we are done
        #return len(self.children) == 0 #We shouldnt have any children in this case. 
    
    def is_root(self):
        if(self.parent is None): # Check if we have no parent, means we must be root.
            return True
        return False
    
    def is_fully_expanded(self): # Check If we have expanded all nodes possible from a state.
        # If we have added all children states we can from every action.
        return len(self.children) == len(self.game.get_actions()) #

    def is_fully_explored(self): # We have checked every action from this node atleast once. i.e. the children do not have any values.
        for child in self.children:
            if child.num_visits == 0:
                return False
        return True

    # *** DISPLAY/VERBOSE FUNCTIONS

    def print_state(self,action): # self is current game state
        self.game.display_turn(action)


    def show_tree(self,depth):
        """ Display the tree, from root and all """
        if(self.is_root()):
            self.print_tree(depth)
        else:
            root = self.get_root()
            root.print_tree(depth)

    def print_tree(self, depth):
        if(self.node_depth > depth):
            return
            #Print a tree starting from self, assume is root.
        if(self.is_termal_node()): # If we are a leaf node
            tabs = self.node_depth-1
            if(self.node_depth != 1):
                print("  "*(tabs+1),end="")
            print("++",end="")
            print("P_{}: {} {}/{} winner {}"
            .format(self.game.get_current_player(), self.action, self.wins, self.num_visits, self.game.get_winner()))
        elif(self.node_depth == 1): # If we are next layer # ! Doest work
            print("--",end="")
            print("P_{}: {} {}/{} {}"
            .format(self.game.get_current_player(),self.action, self.wins, self.num_visits,self.score))
        else:
            tabs = self.node_depth-1
            if(tabs > 0):
                print("  "*(tabs+1),end="")
                print("--", end="")
            print("P_{}: {} {}/{} {}".format(
                self.game.get_current_player(),self.action,
                self.wins, self.num_visits,self.score))
        for child in self.children:
            child.print_tree(depth)

    def get_root(self):
        if(self.parent is None): #We are at start state
            return self
        self.parent.get_root()

    def __str__(self):
        data = {"node_depth":self.node_depth,"previous_action":self.action,
        "wins":self.wins, "visits":self.num_visits,"score":self.score}
        return data.__str__()
