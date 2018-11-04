from Node import *
from NIM import NIM
import variables
#from variables import *
import copy

class MCTS():
    def __init__(self, node:Node, memory_state = 1):
        self.root = node # Set root node of MCTS
            #TODO: use memory states, intra episode or not
        self.memory_state = memory_state # How whether or not we want to keep memory in simulation.
        # 1 - means we don't store anything between inter-episodes.
        # 2 - means we store intra-episode tree.

    def tree_policy(self, node, root_player): #The policy which we choose the node(Leaf) to rollout.  i.e. Tree Search
        """ Return node which has not been visited, or the node showing the most promise(or leaf). """
        if not node.is_termal_node(): # * If we are terminal node, return self.
            if not node.is_fully_expanded(): # If we have not instanciated every child state from this node.
                node.init_children() # create all child states from node.
                return node.best_child(root_player) # Return best child from the parent node (one of the unvisited ones).
            else:
                if not node.is_fully_explored(): # * If children are not fully explored(visited once).
                    return node.best_child(root_player) #  Return best child, which should be an unexplored children
                    # * If we have expanded the node, and explored all the children, we look for the best child.
                return self.tree_policy(node.best_child(root_player,c=2), root_player) # continue search from best child.
        return node # leaf node

    def simulate_best_action(self, node, num_sims):
        """ Return best action from a given state, simulating play from that state, with num_sims itterations to update states.
            node must be a copy of current root state, otherwise we will not simulate play."""
        # We need to create a copy of the node passed in, because we don't want to make changes to original root node.
        # Keep track of who is current playing the real game.
        root_player = node.game.get_current_player() # Seperate simulation vs real game.
        for i in range(0, num_sims): # M = num_sims    
            leaf = self.tree_policy(node, root_player) # Return leaf node we are going to use for rollout from node state.
            #print("leaf", leaf, leaf.game)
            # Victor is the node in the whole simulated tree that is considered best.
            winner = leaf.rollout(root_player) # Rollout from this node, and get the reward from this stage. 
            leaf.backpropagate(winner, root_player) # Go from leaf node and update the values
                        # c = 2, too high exploration, we might actually try to explore more than guarantee winning.
        #TODO: Do we choose UCT1 value, or only Q value? We can set c=0, to remove exploration. Prev c = 1.5
        
        victor = (node.best_child(root_player, c=0,action=True)) # Get best state node from tree.
        return victor

        #return node.best_child(node.game.get_current_player()).action # After we are done, we select best action from parent.
    # TODO: Episode = game, intra-epsodes moves ingame, + keep tree in intra episode.
    
    def play_full_game(self, root_node, num_sims): # Play a whole game using MTCS for both players, from root_node
        """ Play full game with MCTS. node is final state of game after completion """
        node = copy.deepcopy(root_node) # make copy, so we wont make changes to real game
        while(not node.is_termal_node()): # We are termal_node if game is finished at state.
            victor = self.simulate_best_action(node, num_sims) # Get best state node from node state.
            # * Change root state to best action/state from simulation, but keep all previous history.
            node = victor
            if(variables.verbose >= variables.debug):
                victor.parent.show_tree(100)
                print("victor tree -v-")
            node.parent = None # Best state is now the root state. Effectivly prunes tree as well.
            if(variables.verbose >= variables.play): #if we want to display the turns of the real game.
                if(variables.verbose >= variables.debug):
                    victor.show_tree(100)
                #We know the action that the current player took, and the next state we are in.
                #print("victor",victor, victor.game.get_current_state())
                #print(type(victor))
                #print(victor.parent, victor.children)
                #victor.show_tree(100)
                if(variables.verbose >= variables.play):
                    victor.game.display_turn(victor.action) # Display what happened to get to this state.
            #node = Node(game=game, parent=node, action=action, node_depth=node.node_depth+1)
        return node # return termal node, which is the final state of our game.

        #TODO: had self.root.is_termal_node(), and self.root = Node(game=new_state.game,parent=self.root, action=new_state.action, node_depth=new_state.node_depth)
    # ! check if this is correct, by playing in batch mode, with perfect initial position. 
    def play_batch(self, num_sims, batch, start_player=1):
        if(start_player < 1 or start_player > 3):
            raise ValueError('Value of {} as P is not supported'.format(start_player))
        
        wins = [0,0] # Keep track of amount of wins for both players,
        start_state = [0,0] # Keep track of amount of times each player start a game.
        for game in range(1,batch+1): # from 1 to batch plays.
            sim_node = copy.deepcopy(self.root) # create a copy of the start state.
            sim_node.game.init_player_turn(start_player) # Change who begins in a given game state

            if(sim_node.game.get_current_player() == 1):
                start_state[0] += 1
            else:
                start_state[1] += 1
            #print(start_player,"current_player",sim_node.game.get_current_player(), "prev",sim_node.parent)
            if(variables.verbose >= variables.play):
                print("Game {} ###".format(game))
            sim_node = self.play_full_game(root_node=sim_node, num_sims=num_sims)
            #print("sim_node",sim_node)
            winner = sim_node.game.get_winner() # get winning player in terminal state
            if(winner == 1):
                wins[0] += 1
            else:
                wins[1] += 1
        if(variables.verbose >= variables.result):
            print("First moves: Player_1: {} Player_2: {}".format(start_state[0], start_state[1]))
            p_p1 = wins[0]/batch
            p_p2 = wins[1]/batch
            print("Results {} games: Player_1: {:.2f}%({}), Player_2: {:.2f}% ({})".format(batch, p_p1*100, wins[0], p_p2*100, wins[1]))
        # Player turn should be P.
        # We need to create new games of nim for each starting player.

def test_game():
    game = NIM(num_pieces=15, max_pieces=3)
    root = Node(game=game) # root with game we want to play
    mcts = MCTS(node=root) 
    variables.verbose = 2
    node = mcts.play_full_game(root_node=root,num_sims=3000)
    #node.show_tree()
    #print(node)

def test_batch():
    game = NIM(num_pieces=9, max_pieces=3)
    root = Node(game=game) # root with game we want to play
    mcts = MCTS(node=root) # create MCTS alg object.
    mcts.play_batch(batch=10,num_sims=2000,start_player=1)

def testMCTS():
    #game2 = NIM(num_pieces=10, max_pieces=3)
    game = NIM(num_pieces=5, max_pieces=3)
    print(type(game))
    root = Node(game) # root with game we want to play

    mcts = MCTS(node=root)

    next_state = mcts.simulate_best_action(node=mcts.root,num_sims=30)
    print(type(next_state), next_state)
    print(next_state)
    #mcts.show_tree()
#test_batch()
#test_game()
#testMCTS()
""" 
    * Backprogagate
    def backpropagation(self, node): # Backpropagation - Passing the evaluation of a final state back up the tree, 
        # updating relevant data (see course lecture notes) at all nodes and edges on the path from the final state to the tree root.
        node.stats = update_stats(node,resul)
        backpropagation(node.parent)

    * Rollout
    def leaf_evaluation(self): # Estimating the value of a leaf node in the tree by doing a rollout simulation using
        # the default policy from the leaf node’s state to a final state.
        pass
    
    * Expansion, init children to an state.
    def node_expansion(selfnode): # Generating some or all child states of a parent state, and then connecting the tree
        # node housing the parent state (a.k.a. parent node) to the nodes housing the child states (a.k.a. child
        # nodes).
        pass

    * Tree policy
    def tree_search(self,node):  #  rollout_policy, Traversing the tree from the root to a leaf node by using the tree policy.
        pass
"""
    