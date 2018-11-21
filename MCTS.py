from Node import *
from NIM import NIM
from Datamanager import Datamanager # For saving cases.
import variables
#from variables import *
import copy
import time


class MCTS():
    def __init__(self, node:Node, dataset:Datamanager=None, rollout_policy=None, random_rollout=False, greedy=False, time_limit = False, gather_data=False):
        self.root = node # Set root node of MCTS
            #TODO: use memory states, intra episode or not
        #self.memory_state = memory_state # How whether or not we want to keep memory in simulation.
        #self.action = action # {1:play, 2:play_tourn, 3:train, 4:data,5:test}
        self.gather_data = gather_data
        self.rollout_policy = rollout_policy
        self.random_rollout = random_rollout # overwrite network policy
        self.greedy = greedy # If we want an greedy rollout policy or not, with respect to ANET

        self.time_limit = time_limit
        #TODO: handle multiple policies between players.
        # 1 - means we don't store anything between inter-episodes.
        # 2 - means we store intra-episode tree.
        if(dataset is not None): # Dont need an datamanger if no file to manage.
            self.dataset = dataset
        else:
            raise ValueError("Requires a datamanager to store the data..")

        #TODO: change policy to only use a network when needed.

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

    def simulate_best_action(self, node, num_sims,epsilon=0): # If time is True, num_sims is in seconds.
        """ Return best action from a given state, simulating play from that state, with num_sims itterations to update states.
            node must be a copy of current root state, otherwise we will not simulate play."""
        # We need to create a copy of the node passed in, because we don't want to make changes to original root node.
        # Keep track of who is current playing the real game.
        root_player = node.game.get_current_player() # Seperate simulation vs real game.
        #TODO: add a time constraint instead if we want, instead of m simulations
        if(self.time_limit): # num_sims = seconds we are allowed to run
            time_start = time.time() # get time now.
            simulations = 0
            while(time.time() < time_start + num_sims): # if we still have time to run another simulation. 1 second is enough time
                leaf = self.tree_policy(node, root_player) # Return leaf node we are going to use for rollout from node state.
                if(self.rollout_policy is None): # Check if we only want the data.
                    winner = leaf.rollout(root_player)
                    
                else: # We want to use another policy for rollout.
                    #TODO: pass the policy to the rollout function.
                    winner = leaf.rollout(root_player, self.rollout_policy, greedy=self.greedy, epsilon=epsilon) # Rollout from this node, and get the reward from this stage. 
                #print("leaf", leaf, leaf.game)
                # Victor is the node in the whole simulated tree that is considered best.
                # TODO: send network with backpropogate function.

                leaf.backpropagate(winner, root_player) # Go from leaf node and update the values
                            # c = 2, too high exploration, we might actually try to explore more than guarantee winning.
                simulations += 1
            #print("we completed", simulations, "sumulations")
        else:
            for i in range(0, num_sims): # M = num_sims    
                leaf = self.tree_policy(node, root_player) # Return leaf node we are going to use for rollout from node state.
                if(self.rollout_policy is None or self.random_rollout): # Check if we only want the data.
                    winner = leaf.rollout(root_player)
                else: # We want to use another policy for rollout.
                    #TODO: pass the policy to the rollout function.
                    winner = leaf.rollout(root_player, self.rollout_policy, epsilon=epsilon) # Rollout from this node, and get the reward from this stage. 
                #print("leaf", leaf, leaf.game)
                # Victor is the node in the whole simulated tree that is considered best.
                # TODO: send network with backpropogate function.

                leaf.backpropagate(winner, root_player) # Go from leaf node and update the values
                            # c = 2, too high exploration, we might actually try to explore more than guarantee winning.
        # TODO: handle creating the case from simulation results. i.e. get number of visits to each node.
        # * [(0,0)=3,(0,1)=1,(0,2)=40,..., PID]
        # We need to create a seperate node.best_child function for when we actually selects a move, since we might want to get the data aswell.
        if(self.gather_data): # * Whether not we need to add data to buffer or not.
            victor,data = node.get_best_child(c=0, data=True) # Get best state node from tree.
            #Store the data.
            self.dataset.update_buffer(data = data) # Add row to buffer.
            #self.dataset.update_csv_limit(data = [data]) # Add data row to buffer. 
        else: # We dont do 
            victor = node.get_best_child(c=0) # Get best state node from tree.
        return victor # Return best state node, which we want to keep.

        #return node.best_child(node.game.get_current_player()).action # After we are done, we select best action from parent.
    # TODO: Episode = game, intra-epsodes moves ingame, + keep tree in intra episode.
    
    def play_full_game(self, root_node, num_sims,epsilon = 0): # Play a whole game using MTCS for both players, from root_node
        """ Play full game with MCTS. node is final state of game after completion """
        node = copy.deepcopy(root_node) # make copy, so we wont make changes to real game
        while(not node.is_termal_node()): # We are termal_node if game is finished at state.
            victor = self.simulate_best_action(node, num_sims,epsilon=epsilon) # Get best state node from node state.
            # * Change root state to best action/state from simulation, but keep all previous history.
            node = victor
            if(variables.verbose >= variables.debug):
                victor.parent.show_tree(100)
                print("victor tree -v-")
            # ? Remove parent from tree, effectivly prune impossible states.
            node.parent = None # Best state is now the root state. 
            if(variables.verbose >= variables.play): #if we want to display the turns of the real game.
                if(variables.verbose >= variables.debug):
                    victor.show_tree(100)
                    
                if(variables.verbose >= variables.play):
                    victor.game.display_turn(victor.action) # Display what happened to get to this state.
                    victor.game.display_board()
        # Check if we need to update csv file.
        if(self.dataset is not None):
            self.dataset.update_csv_limit() # 
        return node # return termal node, which is the final state of our game.

    #TODO: had self.root.is_termal_node(), and self.root = Node(game=new_state.game,parent=self.root, action=new_state.action, node_depth=new_state.node_depth)
    def play_batch(self, num_sims, games, start_player=1,verbose=True):
        if((start_player < 1 or start_player > 3) and type(start_player) != int):
            raise ValueError('Value of {} as P is not supported'.format(start_player))
        
        wins = [0,0] # Keep track of amount of wins for both players,
        start_state = [0,0] # Keep track of amount of times each player start a game.
        for game in range(1,games+1): # from 1 to batch plays.
            sim_node = copy.deepcopy(self.root) # create a copy of the start state.
            sim_node.game.init_player_turn(start_player) # Change who begins in a given game state

            if(sim_node.game.get_current_player() == 1):
                start_state[0] += 1
            else:
                start_state[1] += 1
            #print(start_player,"current_player",sim_node.game.get_current_player(), "prev",sim_node.parent)
            if(variables.verbose >= variables.play and verbose):
                print("Game {} ###".format(game))
            sim_node = self.play_full_game(root_node=sim_node, num_sims=num_sims)
            #print("sim_node",sim_node)
            winner = sim_node.game.get_winner() # get winning player in terminal state
            if(winner == 1):
                wins[0] += 1
            else:
                wins[1] += 1
        if(variables.verbose >= variables.result and verbose):
            print("First moves: Player_1: {} Player_2: {}".format(start_state[0], start_state[1]))
            p_p1 = wins[0]/games
            p_p2 = wins[1]/games
            print("Results {} games: Player_1: {:.2f}%({}), Player_2: {:.2f}% ({})".format(games, p_p1*100, wins[0], p_p2*100, wins[1]))
        # Player turn should be P.
        # We need to create new games of nim for each starting player.

    def play_batch_with_training(self, optimizer, loss_function, games, training_frequency, storage_frequency, num_sims, init_data_games=0, data_sims = 0, epoch=0, start_player=1, iterations=10, batch=30, init_train=False): # * Function to train for each batch.
        # batch_size is number of games we play, num_sims is seconds we simulate for each move.
        # storage_frequency is how often we save an agent. training_frequency is how many games between training.
        # If batch is 1, we train after each game, otherwise we choose.
        # We also don't want to start training until buffer is filled with random data.
        if(start_player < 1 or start_player > 3):
            raise ValueError('Value of {} as P is not supported'.format(start_player))

        # fill buffer using random rollout, since this is better than untrained network.
        size = self.dataset.get_buffer_size()
        # If size != 0, it means we don't train from scratch.
        # If size == 0, we should do some random rollouts to gather some data, before training.
        # Play 10 games, with random rollout, 800 sims per move -> 600*moves rollout, should be an OK aproximation to begin with.
        if(size == 0 and init_data_games != 0 and data_sims != 0): # means we want to gather som random data firstly.
            print("Gathering random data")
            self.random_rollout = True
            self.play_batch(num_sims = data_sims, games=init_data_games, start_player=3, verbose=False) # Automaticly use the policy if availible for a move.
            self.random_rollout = False
        elif(size != 0 and init_train and epoch == 0): # If buffer has data, and we want to pre train, and start from first epoch.
            # We want to train policy on data we have in buffer, atleast a little bit.
            dataset_test = Datamanager("Data/data_r_test.csv",dim=5)
            print("Pretraining on {} with test {}".format(self.dataset.filepath, dataset_test.filepath))
            #exit()
            init_optimizer = torch.optim.RMSprop(self.rollout_policy.parameters(), lr=0.005,alpha=0.99,eps=1e-8)
            #init_optimizer = copy.copy(optimizer) # Don't want to change params on self training optimizer.
            init_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(init_optimizer, 'min')
            for itt in range(1,200+1): # 10 itterations, batch 50
                loss_train,loss_test = network.train(self.rollout_policy,batch=50, iterations=10,
                casemanager_train=self.dataset,casemanager_test=dataset_test, optimizer = init_optimizer,loss_function=loss_function,verbose=100)
                init_scheduler.step(loss_test)  
                print("pre_itteration {}  loss_train: {:.9f} loss_train: {:.9f} lr: {} ".format(itt, loss_train,loss_test, optimizer.param_groups[0]["lr"]))
            # Store network again.
            #self.rollout_policy.store(epoch=-1, optimizer=init_optimizer, loss=loss_train)

        if(epoch == 0): # Save network if we are starting from scratch.
            #Start by saving the network as agent 0.
            self.rollout_policy.store(epoch=1, optimizer=optimizer, loss=1000)#Max loss to begin with.
            #pass

        if(variables.verbose > variables.play):
            print("Start training with self play using policy network")

        training_count = epoch # Count number of times we have trained, to easily check if needing to store.
        loss_history = [] # Store previous losses
        #scheduler = torch.optim.lr_scheduler(optimizer,step_size = 30, )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5)
        #97,5 90, 80, 60
        #1/version 
        epsilon = 1
        decrease = 0.025 # For each training dercrease by 5
        epsilon = max(0,1-training_count*decrease) # 4 * 0.025 = 0.10, 10 * 0.025 = 0.25; 20 *0.025 =0.5, 40*0.025 = 1

        for game in range(1+epoch, games + 1 + epoch): # number of games we play
            print("Game {}".format(game))
            sim_node = copy.deepcopy(self.root) # Copy root state of game. # So that we have a starting point to simulate from.
            sim_node.game.init_player_turn(start_player) # Change who begins in a given game state
            # * Play game
            print("Using epsilon",epsilon, "training_count", training_count)
            sim_node = self.play_full_game(root_node=sim_node, num_sims=num_sims, epsilon=epsilon) # get last state of game. # Testing
            
            # Check winner.
            if(game % training_frequency == 0): # We want to train between every game?
                #We train.
                loss = network.train(self.rollout_policy,casemanager_train=self.dataset, optimizer=optimizer, 
                loss_function=loss_function, iterations=iterations,batch=batch)
                print("Epoch {} loss {:.8f} lr:{}".format(game, loss,optimizer.param_groups[0]["lr"]))
                loss_history.append(loss) # Add current loss to history.
                
                #print(optimizer.param_groups[0]["lr"])
                # collect the last x losses from history.
                training_count += 1 # update training count
                epsilon = max(0,1-training_count*decrease) # 4 * 0.025 = 0.10, 10 * 0.025 = 0.25; 20 *0.025 =0.5, 40*0.025 = 1
                if(epsilon <= 0): # Don't want to decrease learning rate until we remove random rollout.
                    scheduler.step(loss) # If loss stagnates over 10 games, we need to decrease it.
                # Check if we want to store this trained policy network. 
                if(training_count % storage_frequency == 0 or game == games+epoch): # store every x training times, or at last itteration.
                    # Decrease epsilon value.
                    # training count, is basicly epoch with tc = 1
                    
                    #epsilon = epsilon/(training_count+1) # 
                    print("Storing network...")
                    self.rollout_policy.store(epoch=training_count, optimizer = optimizer, loss = loss, datapath=self.dataset.filepath)
                    # We need to save our epoch. networkName_0, networkName_1, networkName_2, etc.

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