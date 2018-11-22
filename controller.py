# Controller 
#from GameSimulator import *  # Not used..
from HEX import *
import argparse
import variables
import actor
import sys
import os
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.models import load_model
sys.stderr = stderr
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#from variables import *
from Node import *
from MCTS import *
#pytorch
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

def check_bool(value):
    print(type(value))
    if(value == "True"):
        return True
    elif(value == "False"):
        return False
    raise ValueError("Not bool")


def play_hex(dim): # TODO: init start player etc.
    return HEX(dim)

def ANET_TEST(name, dim): # TODO FIX
    input_dim = (dim*dim*2)+2
    target_dim = dim*dim
    return network.Model(nn.Linear(input_dim, 100),nn.ReLU(),
        nn.Linear(100,target_dim), nn.Softmax(dim=-1), name=name)

def ANET_TEST_2(name, dim): # TODO FIX
    input_dim = (dim*dim*2)+2
    target_dim = dim*dim
    return network.Model(nn.Linear(input_dim, 80),nn.ReLU(),
        nn.Linear(80, 40),nn.ReLU(),
        nn.Linear(40,target_dim), nn.Softmax(dim=-1), name=name)
    # K = filter_size, W = input height/leight, P padding, S stride
    # zero-padding = (K-1)/2
    # O = (W-K + 2P)/S + 1

def HEX_CNN(name, dim,filepath=None): # Best so far..
    input_dim = (dim*dim*2)+2
    target_dim = dim*dim 
    input_type = 2
    return network.Model(                                    # O = (5-3 +2) +1 = 5 
        nn.Conv2d(3,3,kernel_size=(3,3),stride=1,padding=1), # -> 3*5*5 = 75 # 1,3,5,5 output
        nn.ReLU(),
        #nn.MaxPool2d(kernel_size=(3,3),padding=1,stride=1), # -> 1,3,5,5
        network.Flatten(),
        nn.Linear(75, 25),
        nn.Softmax(dim=-1), name=name,input_type=input_type,filepath=filepath)

# ? HEX-CNN-20: Rmsprop, SSE loss

def HEX_CNN_TWO(name, dim, filepath=None): 
    input_dim = (dim*dim*2)+2
    target_dim = dim*dim 
    input_type = 2
    return network.Model(                                    # O = (5-3 +2) +1 = 5 
        nn.Conv2d(3,3,kernel_size=(3,3),stride=1,padding=1), # -> 3*5*5 = 75 # 1,3,5,5 output
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=(3,3),padding=1,stride=1), # -> 1,3,5,5
        network.Flatten(),
        nn.Linear(75, 25),
        nn.Softmax(dim=-1), name=name,input_type=input_type,filepath=filepath)
# ? HEX-CNN-TWO: 

# CNN-MAX , ADAM, SSE
def HEX_CNN_POOL(name, dim, filepath=None): 
    input_dim = (dim*dim*2)+2
    target_dim = dim*dim 
    input_type = 2
    return network.Model(                                    # O = (5-3 +2) +1 = 5 
        nn.Conv2d(3,3,kernel_size=(3,3),stride=1,padding=1), # -> 3*5*5 = 75 # 1,3,5,5 output
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2),padding=1,stride=2), # -> 1,3,5,5
        network.Flatten(),
        nn.Linear(27, 25),
        nn.Softmax(dim=-1), name=name,input_type=input_type,filepath=filepath)

def HEX_CNN_L2(name, dim, filepath=None): 
    input_dim = (dim*dim*2)+2
    target_dim = dim*dim 
    input_type = 2
    return network.Model(                                    # O = (5-3 +2) +1 = 5 
        nn.Conv2d(3,3,kernel_size=(3,3),stride=1,padding=1), # -> 3*5*5 = 75 # 1,3,5,5 output
        nn.ReLU(),
        network.Flatten(),
        nn.Linear(75, 50),nn.ReLU(),
        nn.Linear(50, 25),nn.ReLU(),
        nn.Softmax(dim=-1), name=name,input_type=input_type,filepath=filepath)

FUNCTION_MAP = {'NIM' : "play_nim",
                'HEX' : play_hex}

if __name__=="__main__":
    #Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Reinforcement learning MCTS')
    #parser.add_argument("--game", action="store_true", help="foo help")
    parser.add_argument("-v","--verbose", default=1,
                            required=False,type=check_positive_max,
                            help="1 = result, 2 = play(only at 2), 3 = debug(sim tree), 4 = debug_all") # integer
    parser.add_argument("-s","--num_sims", default=600, type=check_positive)
    parser.add_argument("-p","--start_player", default=3,choices=[1,2,3],
                        type=check_positive)
    parser.add_argument("-g","--games",help="number of games we play", default=1,
                    type=check_positive)
    """parser.add_argument("-a","--action", choices=["train","play","play_tourn","test","data"],
                        required=True, type=str)  """
    parser.add_argument("-r","--rollout",choices=["random","ANET"],
                        default = "random") 
    parser.add_argument("-d","--dimentions", type=check_positive, help="Dimention of board", 
                default=5, required=False)
    parser.add_argument("-tl","--time_limit", default = False, type=bool,
                                required=False)

        # dest is where we store string we chose as action, to use later.
    subparsers = parser.add_subparsers(title="action", dest="sub_action",help="sub-game help")
    parser_a = subparsers.add_parser("TRAIN")
    parser_a.add_argument("-i","--iterations",help="number times we iterate training data", default=4,
                    type=check_positive)
    parser_a.add_argument("-b","--batch_size",help="amount of data we train per iteration", default=32,
                    type=check_positive)
    parser_a.add_argument("-sf","--storage_frequency",help="how often we store a mode, w.r.t. training", default=10,
                    type=check_positive)
    parser_a.add_argument("-tf","--training_frequency",help="how often we train, w.r.t games", default=1,
                    type=check_positive)
    
    parser_a.add_argument("-ig","--init_games",help="number of games we init random data", default=0,
                    type=check_positive)
    parser_a.add_argument("-is","--init_sims",help="number of simulations we init random data", default=5000,
                    type=check_positive)
    parser_a.add_argument("-it","--init_train", help="Init training on data",type=bool, default=False)



    parser_b = subparsers.add_parser("TOPP") # TOPP tournament

    parser_b.add_argument("-g","--topp_games",type=check_positive, default=10)
    parser_b.add_argument("-mp","--model_path",type=str, required=True)
    #parser_b.add_argument("-a","--agents",type=str, nargs='*' , required =True)
    parser_b.add_argument("-r","--random",type=bool, default=False)

    parser_c = subparsers.add_parser("DATA") # Only store data.
    parser_d = subparsers.add_parser("FIX")
    parser_e = subparsers.add_parser("PLAY")

    args = parser.parse_args()

    game = None # Init game.
    variables.verbose = args.verbose # set global verbose rate.

    #Can only play hex.
    game = play_hex(dim=args.dimentions)

    # Handle action arg
    #action = variables.action.get(args.action) # None is default in our case.
    #rollout = variables.rollout_policy.get(args.rollout)
    #time_limit = args.time_limit # get boolean if something set
    games = args.games
    
    if(game is not None):
        root = Node(game) # Init root node from game state.
        if(args.rollout == "ANET"):
            if(False): # If we are not using keras
                import torch.optim as optim
                import torch.nn.modules.loss as pyloss
                import network
                # create network.
                rollout_policy = network.HEX_CNN_L4("CNN-L4",args.dimentions) # Use default values
                rollout_policy.apply(network.weights_init) # init weights and biases.
                print("input_type",rollout_policy.input_type)
                #exit()
                print("Network", rollout_policy,"input-type",rollout_policy.input_type)
                #optimizer = optim.RMSprop(rollout_policy.parameters(), lr=0.005,alpha=0.99,eps=1e-8)
                #optimizer = optim.RMSprop(rollout_policy.parameters(), lr=0.005,alpha=0.99,eps=1e-8)
                optimizer = optim.Adam(rollout_policy.parameters(), 
                lr=0.001,betas=(0.9,0.999),eps=1e-6,amsgrad=True,weight_decay=0.005)
                #loss_function = nn.MultiLabelMarginLoss()
                loss_function = pyloss.MSELoss(reduction='sum') # a bit better
                #loss_function = pyloss.MSELoss()

                #TODO: handle continue training from file.
                mcts = MCTS(node=root,
                    time_limit=args.time_limit, rollout_policy=rollout_policy)
            else: 
                import network_keras
                from keras.optimizers import SGD,Adam,Adagrad,RMSprop
                from keras import losses

                sgd = SGD(lr=0.01)
                rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
                adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
                adagrad = Adagrad(lr=0.01, epsilon=None, decay=0.0)
                
                mse = losses.mean_squared_error
                cce = losses.categorical_crossentropy
                #Loss functions:
                #"categorical_crossentropy", "mse"
                # ! no _ in name, otherwise we won't find latest version when training from file.
                rollout_policy = network_keras.NN_50_25(name="K-NN.25",dim=args.dimentions)

                mcts = MCTS(node=root,
                    time_limit=args.time_limit, rollout_policy=rollout_policy)
        else:
            mcts = MCTS(node=root, 
                time_limit=args.time_limit) 
        #mcts.simulate_best_action(root,10)
        if(args.sub_action == "TRAIN"): # We want to train against ourself.
            
            datamanager = Datamanager("Data/buffer_K_NN_50.csv",dim=args.dimentions,limit=500)
            print(datamanager.filepath)
            mcts.dataset = datamanager

            action = variables.action.get("train") # Not sufe if used anymore
            mcts.gather_data = True # need to specificly set this.
            iterations = args.iterations # default 5
            batch = args.batch_size # default 50
            storage_frequency = args.storage_frequency # default 10
            training_frequency = args.training_frequency # default 1

            init_games = args.init_games # default 10
            init_sims = args.init_sims # default 5000
            init_train = args.init_train
            epoch = 0 # rollout_policy
            
            # Load previous model from file if exists.
            if(rollout_policy is not None): # * Load from prev saved model if exists.
                name = rollout_policy.name
                path, epoch = misc.find_newest_model_keras(name)
                if(path is not None): # TODO: make sure optimizer is the same, otherwise error here.
                    # Load model from path.
                    rollout_policy.load(path)
                    #loss, epoch = rollout_policy.load_model(path,optimizer=optimizer) # Load optimizer settings too.

                    print("Loading self from path \"{}\" at epoch {}".format(path, epoch))
            else:
                print("Using random rollout only ----------- ")
            mcts.play_batch_with_training(epoch=epoch, games=games,training_frequency=training_frequency, storage_frequency=storage_frequency, 
            num_sims=args.num_sims, iterations=iterations,batch=batch,data_sims=init_sims,init_data_games=init_games, init_train=init_train)
        elif(args.sub_action == "TOPP"):
            # Load models from file and start tournament between them.
            topp_games = args.topp_games
            """
            model_path = args.model_path
            #Model path should be same name as path..
            model = HEX_CNN("E-HEX-CNN",5)
            print("TOPP", model_path, topp_games)
            # TODO: pass subfolder in top
            print(misc.find_models(model_path))
            actors = []
            for path in misc.find_models(model_path):
                s_path = path.split("_") # get epoch name.
                name = model_path + "-"+s_path[-1]
                #name = "-".join(model_path) # Last index should be our name
                actors.append(Actor.Actor(HEX_CNN(name=name,dim=5,filepath=path)))
                #models.append()
            print(actors)
            """
            model_1 = network.HEX_CNN(name="HEX_CNN-1", dim=args.dimentions, filepath="models/CNN-SSE-ADAM/CNN-SSE-ADAM_500")
            model_2 = network.HEX_CNN_L2(name="HEX-CNN-L2", dim=args.dimentions, filepath="models/CNN-L2-ADAM/CNN-L2-ADAM_500")
            #model_50 = network.HEX_CNN(name="HEX-CNN-POOL-50", dim=args.dimentions, filepath="models/HEX-CNN-POOL/HEX-CNN-POOL_50")
            #model_80 = HEX_CNN_TWO(name="HEX-CNN-POOL-80", dim=args.dimentions, filepath="models/HEX-CNN-POOL/HEX-CNN-POOL_80")
            #model_120 = HEX_CNN_TWO(name="HEX-CNN-POOL-120", dim=args.dimentions, filepath="models/HEX-CNN-POOL/HEX-CNN-POOL_120")
            models = [model_1,model_2]
            #actors = [actor.Actor(model_1),actor.Actor(model_20),actor.Actor(model_50),actor.Actor(model_80),actor.Actor(model_120)]
            actor.tournament(game,models,games=topp_games)
            #model1 = network.Model(nn.Linear(52,80), nn.ReLU(), nn.Linear(80,25), nn.Softmax(dim=-1), name="rms_mod",filepath="models/rms_mod/rms_mod_10000")
            #Actor.tournament(game,games=100, 
            #models=[model1], random=True)
            pass
        elif(args.sub_action == "DATA"):
            mcts.gather_data = True # need to specificly set this.
            mcts.play_batch(games=games,num_sims=args.num_sims,start_player=args.start_player)
        elif(args.sub_action == "FIX"):
            datamanager.fix_board_state()
        elif(args.sub_action == "PLAY"): # Assume we just want to play
            mcts.play_batch(games=games,num_sims=args.num_sims,start_player=args.start_player)
            

            #optimizer = optim.SGD(rollout_policy.parameters(), lr=5e-4,momentum=0.01, dampening=0)
            #optimizer = optim.RMSprop(rollout_policy.parameters(), lr=0.005,alpha=0.99,eps=1e-8,weight_decay=0)
            
    else:
        raise ValueError("No game initiated")

# Loss functions:
# mse_loss and mse_loss

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