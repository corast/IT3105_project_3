# Controller 
#from GameSimulator import *  # Not used..
from NIM import *
from HEX import *
import argparse
import variables
#from variables import *
from Node import *
from MCTS import *
#pytorch
import torch.optim as optim
import torch.nn as nn

#torch.manual_seed(2809)
#np.random.seed(2809)


def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def check_positive_max(value,max=5):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    elif ivalue >= max: # 4 is max
        raise argparse.ArgumentTypeError("%s is an invalid positive int value, 4 is max" % value)
    return ivalue

def check_integer(max):
    class customAction(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            print(values)
            if(max > values):
                setattr(args,self.dest, max)
                print("Value error")
                #raise ValueError("max_pieces must be smaller than num_pieces in game")
                setattr(values, self.dest, max-1)
            print(values)
            #setattr(args, self.dest, values)
    return customAction

def play_nim(max_pieces, num_pieces):
    return NIM(max_pieces,num_pieces) # Return NIM game.

def play_hex(dim): # TODO: init start player etc.
    return HEX(dim)

def ANET():
    return network.Module(insize=52,outsize = 25,name="ANET")



FUNCTION_MAP = {'NIM' : play_nim,
                'HEX' : play_hex}


    # g = num_sims
    #
if __name__=="__main__":
    #Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Reinforcement learning MCTS')
    #parser.add_argument("--game", action="store_true", help="foo help")
    parser.add_argument("-v","--verbose", default=1,
                            required=False,type=check_positive_max,
                            help="1 = result, 2 = play(only at 2), 3 = debug(sim tree), 4 = debug_all") # integer
    parser.add_argument("-s","--num_sims", default=100,
                        required=True, type=check_positive)
    parser.add_argument("-p","--start_player", default=1,choices=[1,2,3],
                        type=check_positive, required=True)
    parser.add_argument("-b","--batch",help="number of games we play", default=1,
                    type=check_positive)
    
    parser.add_argument("-a","--action", choices=["train","play","play_tourn","test","data"],
                        required=True, type=str) 

    parser.add_argument("-r","--rollout",choices=["random","ANET"],
                        default = "random") 
    parser.add_argument("-tl","--time_limit", default = False, type=bool,
                                required=False)

    subparsers = parser.add_subparsers(title="game", dest="game",help="sub-game help"
                    ,required=True)
    
    #command nim 
    parser_a = subparsers.add_parser("NIM",help="NIM help")
    parser_a.add_argument("-m","--max_pieces", default=3,
                        required=True,type=check_positive,
                        help="Max number of pieces we can pick in each state")
    parser_a.add_argument("-n","--num_pieces", default=10,
                        required=True,type=check_positive,
                        help="Total number of pieces to begin with")

    #command other game..
    parser_b = subparsers.add_parser("HEX",help="HEX help")
    parser_b.add_argument("-d","--dimentions", type=check_positive, help="Dimention of board", 
                default=5, required=False)

    args = parser.parse_args()

    game = None # Init game.
    variables.verbose = args.verbose # set global verbose rate.

    if(args.game == "NIM"):
        #handle nim
        #check if m/max_pieces > n/num_pieces
        if(args.max_pieces > args.num_pieces): 
            raise ValueError("max_pieces must be smaller than num_pieces in game")
        elif(args.max_pieces == 1):
            raise ValueError("max_pieces must be greater than 1")
        game = play_nim(max_pieces=args.max_pieces, num_pieces=args.num_pieces)
    elif(args.game == "HEX"):
        game = play_hex(dim=args.dimentions)

    # Handle action arg
    action = variables.action.get(args.action) # None is default in our case.
    rollout = variables.rollout_policy.get(args.rollout)
    time_limit = args.time_limit # get boolean if something set
    if(action is None):
        raise ValueError("No action selected")
    
    if(args.rollout == "ANET"):
        # We need to make our network.
        rollout_policy = ANET() # Use default values
        rollout_policy.apply(network.weights_init) # init weights and biases.

    if(game is not None):
        root = Node(game) # Init root node from game state.
        if(args.rollout == "ANET"):
            # create network.

            mcts = MCTS(node=root, action=action, datamanager=Datamanager("Data/data_random.csv", 
                dim=args.dimentions),time_limit=time_limit, rollout_policy=rollout_policy)
        else:
            mcts = MCTS(node=root, action=action, datamanager=Datamanager("Data/data_random.csv", dim=args.dimentions),time_limit=time_limit) 
        #mcts.simulate_best_action(root,10)
        if( action == variables.action.get("train")):
            #We want to train isntead of play batch.
            #optimizer = optim.SGD(rollout_policy.parameters(), lr=5e-4,momentum=0.01, dampening=0)
            optimizer = optim.RMSprop(rollout_policy.parameters(), lr=0.005,alpha=0.99,eps=1e-8,weight_decay=0)
            loss_function = nn.MultiLabelMarginLoss()
            mcts.play_batch_with_training(optimizer=optimizer,loss_function=loss_function,batch_size=100,training_size=1,k=1, num_sims=args.num_sims)
        else: # Assume we just want to play
            mcts.play_batch(batch=args.batch,num_sims=args.num_sims,start_player=args.start_player)
    else:
        raise ValueError("No game initiated")


# QA 
#   Don't use whole buffer. Will simply make it harder to adjust to new players.
#   
#    Good w/r against random. 200 episodes, beats 2/3 games. 66% is relativly well.
#       Along as it seems to be winning, it's probally in good shape.
#    Start player should win most of the time.
#   
#   Network Architecture:
#       Not very deep. Keiths's, 50, 50, 25, output.
# Alpha go. input state player, one. Expansion code with matrix, where every value represent player (player has huge effect on game), drawback, harder to train ( more weights )
#   Conv, don't usually needed on small games like this (5x5), defently on 10x10 board game.
#   
#   Interface, can test as much as possible. One minute limit.
#       Tournament, a bit faster.

#   Must do rollouts, but not in tournament.

#    Different strategies. Rollout strategy don't necessary be greedy.
#      Stochasticly choose the probability of the network. actio1 0.9 percentage, 9/10 times.
#    In tourn, we might not want to explore as much as during tournament. Epsilon. 
#   
#    One of the players will be random in the tornament.
#
#    Drawback of Relu, might give us an zero as output (Negative input).
#   
#   How many MCTS per move, ideely 1000 per move. Beginning, we don't need to call as often. 
#   Maby do as many rollouts as we have time for.
#   Rollouts in the beginning, not very usefull anyway.

#   Values in distrubution, visits are effected by wins, so should work.