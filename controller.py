# Controller 
#from GameSimulator import *  # Not used..
from NIM import *
import argparse
import variables
#from variables import *
from Node import *
from MCTS import *

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

def play_chess():
    return None


FUNCTION_MAP = {'NIM' : play_nim,
                'CHESS' : play_chess}

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
    parser_b = subparsers.add_parser("CHESS",help="CHESS help")
    parser_b.add_argument("baz", type=int, help="baz help")

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
    elif(args.game == "CHESS"):
        pass

    if(game is not None):
        root = Node(game) # Init root node from game state.
        mcts = MCTS(node=root) 
        mcts.play_batch(batch=args.batch,num_sims=args.num_sims,start_player=args.start_player)