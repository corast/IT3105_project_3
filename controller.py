# Controller 
#from GameSimulator import *  # Not used..
from HEX import *
import argparse
import variables
import actor
#from variables import *
from Node import *
from MCTS import *
#pytorch
import torch.optim as optim
import torch.nn as nn
import torch.nn.modules.loss as pyloss

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
                        default = "random", required=False) 
    parser.add_argument("-d","--dimentions", type=check_positive, help="Dimention of board", 
                default=5, required=False)
    parser.add_argument("-tl","--time_limit", default = False, type=bool,
                                required=False)

        # dest is where we store string we chose as action, to use later.
    subparsers = parser.add_subparsers(title="action", dest="sub_action",help="sub-game help"
                    ,required=True)
    parser_a = subparsers.add_parser("TRAIN")
    parser_a.add_argument("-i","--iterations",help="number times we iterate training data", default=4,
                    type=check_positive)
    parser_a.add_argument("-b","--batch_size",help="amount of data we train per iteration", default=30,
                    type=check_positive)
    parser_a.add_argument("-sf","--storage_frequency",help="how often we store a mode, w.r.t. training", default=10,
                    type=check_positive)
    parser_a.add_argument("-tf","--training_frequency",help="how often we train, w.r.t games", default=1,
                    type=check_positive)
    parser_a.add_argument("-gr","--greedy_rollout",help="if we use greedy ANET", default = True,type=check_bool)
    
    parser_a.add_argument("-ig","--init_games",help="number of games we init random data", default=0,
                    type=check_positive)
    parser_a.add_argument("-is","--init_sims",help="number of simulations we init random data", default=5000,
                    type=check_positive)
    parser_a.add_argument("-it","--init_train", help="Init training on data",type=bool, default=False)


    parser_b = subparsers.add_parser("TOPP") # TOPP tournament

    parser_b.add_argument("-g","--topp_games",type=check_positive, default=1)
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
    
    datamanager = Datamanager("Data/buffer_NN-25-4-PB-RMSP-G.csv",dim=args.dimentions,limit=750)
    print(datamanager.filepath)
    # ! HEX-CNN buffer_HEX_CNN Adam
    # ! HEX-CNN-L2 buffer_HEX_CNN-L2 Adam

    # ! HEX-CNN buffer_HEX_CNN Adam
    # ! HEX-CNN-2 buffer_HEX_CNN_2 Adam

    # ! NN-50-tanh-two buffer_NN_50_tanh_two 3000 sims Adam
    # ! NN-50-tanh buffer_NN_50_tanh Adam nongreed
    # ! NN-25 buffer_NN_25 -s 5000 Epsilon = 0.5 Adam nongree
    
    # ! NN_50_norm buffer_NN_50_norm_tree.csv -s 5000 Epsilon = 0.5 Adam nongree

    # ! NN-25 buffer_NN_25 -s 5000 (-200 pr storage) , Eps = 0.5 , 4 itt, batch 30 Adam nongreed
    # ! NN-25-20 buffer_nn_25 -s 5000 (-200 pr storage), eps = 0.5 , 20 its, batch 30 Adam Nongreed
    # ! NN-25-20-WB buffer_nn_25_WB -s 5000 (-200 pr storage), eps = 0.5, 20 its, batch 30 Adam Nongree
    # ! NN-25-20-PB buffer_nn-25-PB-RMSP -s 5000 (-200 pr storage), eps = 0.5, 20 its, batch 30 RMSP Nongreed
    # ! NN-25-20-PB-G buffer_nn-25-PB-RMSP-G -s 5000 (-100 pr storage), eps = 0.5, 20 its, batch 30 RMSP greed 750 buffer
    # ! NN-25-4-PG-G buffer_NN-25-4-PG-G-RMSP -s 5000 (-100 pr storage), eps = 0.5, 4 its, batch 30 RMSP greed 750 buffer

    
    # ! HEX-CNN buffer_HEX_CNN  3k sim
    # ! HEX-CNN-2 buffer_HEX_CNN_2 4k sim# pre-loaded
    # ! HEX-CNN-3 buffer_HEX_CNN_3 4k sim# pre-loaded + many itterations training.
    # ! HEX-CNN-4 buffer_HEX_CNN_4 # pre-loaded + only greedy after epsilon.
    if(game is not None):
        root = Node(game) # Init root node from game state.
        if(args.rollout == "ANET"):
            # create network.
            
            #rollout_policy = network.HEX_CNN("HEX-CNN-2",args.dimentions) # Use default values
            #rollout_policy = network.HEX_CNN_L2("NN-50",args.dimentions) # Use default values
            rollout_policy = network.NN_25("NN-25-4-PB-RMSP-G",args.dimentions) # Use default values
            #rollout_policy = network.NN_50("NN-50-tanh-two",args.dimentions) # Use default values
            #rollout_policy = network.NN_50_norm("NN-50-norm", args.dimentions)
            #rollout_policy.apply(network.weights_init) # init weights and biases.
            print("Network", rollout_policy,"input-type", rollout_policy.input_type)
            #optimizer = optim.RMSprop(rollout_policy.parameters(), lr=0.005,alpha=0.99,eps=1e-8)
            #optimizer = optim.RMSprop(rollout_policy.parameters(), lr=0.005,alpha=0.99,eps=1e-8)
            #optimizer = optim.Adam(rollout_policy.parameters(), lr=0.001,betas=(0.9,0.999),eps=1e-6)
            #loss_function = nn.MultiLabelMarginLoss()
            #loss_function = pyloss.MSELoss(reduction='sum') # a bit better
            #loss_function = pyloss.MSELoss()

            #TODO: handle continue training from file.
            greedy = args.greedy_rollout
            mcts = MCTS(node=root, dataset=datamanager,
                time_limit=args.time_limit, rollout_policy=rollout_policy,greedy=greedy)
        else:
            mcts = MCTS(node=root, dataset=datamanager, 
                time_limit=args.time_limit) 
        #mcts.simulate_best_action(root,10)
        if(args.sub_action == "TRAIN"): # We want to train against ourself.
            action = variables.action.get("train")
            mcts.gather_data = True # need to specificly set this.
            iterations = args.iterations # default 5
            batch = args.batch_size # default 50
            storage_frequency = args.storage_frequency # default 10
            training_frequency = args.training_frequency # default 1

            init_games = args.init_games # default 10
            init_sims = args.init_sims # default 5000
            init_train = args.init_train
            epoch = 0 # rollout_policy
            # * OPTIMIZERS
            #optimizer = optim.RMSprop(rollout_policy.parameters(), lr=0.005,alpha=0.99,eps=1e-8)
            optimizer = optim.RMSprop(rollout_policy.parameters(), lr=0.005,alpha=0.99,eps=1e-8)
            #optimizer = optim.Adam(rollout_policy.parameters(), lr=0.001,betas=(0.9,0.999),eps=1e-6)
            #optimizer  = optim.SGD(model.parameters(), lr=0.01,momentum=0.2, dampening=0) 
            #optimizer = optim.Adagrad(model.parameters(), lr=1e-2, lr_decay=0,weight_decay=0)
            # * LOSS FUNCTIONS
            #loss_function = nn.MultiLabelMarginLoss()
            #loss_function = pyloss.MSELoss(reduction='sum') # a bit better
            #loss_function = network.CategoricalCrossEntropyLoss()
            #loss_function = pyloss.MSELoss()
            loss_function = pyloss.MSELoss(reduction='sum') # a bit better

            # Load previous model from file if exists.
            if(rollout_policy is not None): # * Load from prev saved model if exists.
                name = rollout_policy.name
                path = misc.find_newest_model(name)
                if(path is not None): # TODO: make sure optimizer is the same, otherwise error here.
                    # Load model from path.
                    loss, epoch = rollout_policy.load_model(path,optimizer=optimizer) # Load optimizer settings too.

                    print("Loading self from path \"{}\" at epoch {}".format(path, epoch))
            else:
                print("Using random rollout only ----------- ")
            mcts.play_batch_with_training(optimizer=optimizer,epoch=epoch,
            loss_function=loss_function,games=games,training_frequency=training_frequency, storage_frequency=storage_frequency, num_sims=args.num_sims,
            iterations=iterations,batch=batch,data_sims=init_sims,init_data_games=init_games, init_train=init_train)
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
            model_1 = network.HEX_CNN(name="v1", dim=args.dimentions, filepath="models/HEX-CNN-2/HEX-CNN-2_1")
            for params in model_1.parameters():
                print(params.data)
            #print(model_1.parameters().data)
            model_2 = network.HEX_CNN(name="v50", dim=args.dimentions, filepath="models/HEX-CNN-2/HEX-CNN-2_50")
            model_3 = network.HEX_CNN(name="v100", dim=args.dimentions, filepath="models/HEX-CNN-2/HEX-CNN-2_100")
            model_4 = network.HEX_CNN(name="v140", dim=args.dimentions, filepath="models/HEX-CNN-2/HEX-CNN-2_140")
            for params in model_4.parameters():
                print(params.data)
            #model_50 = network.HEX_CNN(name="HEX-CNN-POOL-50", dim=args.dimentions, filepath="models/HEX-CNN-POOL/HEX-CNN-POOL_50")
            #model_80 = HEX_CNN_TWO(name="HEX-CNN-POOL-80", dim=args.dimentions, filepath="models/HEX-CNN-POOL/HEX-CNN-POOL_80")
            #model_120 = HEX_CNN_TWO(name="HEX-CNN-POOL-120", dim=args.dimentions, filepath="models/HEX-CNN-POOL/HEX-CNN-POOL_120")
            models = [model_1,model_2,model_3,model_4]
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
            raise ValueError("DONT USE FIX")
            #datamanager.fix_board_state_player()
        elif(args.sub_action == "PLAY"):
            mcts.play_batch(games=games,num_sims=args.num_sims,start_player=args.start_player)
        else: # Assume we just want to play
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