# Abstract game class, every game must inheret. Standard functions are used by system.
# Only handles board games, with two players.
import abc #Abstract Base Classes
class Game():
    def __init__(self, state, player_turn):
        pass
    # handles creating the different games, NIM, w/e we want to create.
    @abc.abstractclassmethod
    def get_actions(self): # Get legal moves
        """ Returns allowed actions from a state """
        raise NotImplementedError("Implement get_actions function")

    @abc.abstractclassmethod
    def get_current_state(self): # get current state of the board.
        """ Return information about current state. For debugging """
        raise NotImplementedError("Implement get_current_state function")

    @abc.abstractclassmethod
    def get_reward(self):
        """ Return value of a terminal state, and which player won. """
        raise NotImplementedError("Implement get_reward function")
    
    @abc.abstractclassmethod
    def get_current_player(self): # Return which player is at play.
        raise NotImplementedError("Implement get_current_player function")
    
    @abc.abstractclassmethod
    def play(self, action): # We make a play(move) based on an action.
        """ Need to return what player won the game if it was a winning move, or a tie if not. 
        0 if still going, player_number if that player won with this move, -1 if it was a tie(Can happen in some games, chess etc.) """
        raise NotImplementedError("Implement play function")
    
    @abc.abstractclassmethod
    def play_state(self, action):
        """ Return the state from a spefic game, without making changes to the original game. """
        raise NotImplementedError("Implement play_state function")

    @abc.abstractclassmethod
    def init_player_turn(self, start_player):
        """ chose which player start for a given game. """
        raise NotImplementedError("Implement init_player_turn function")
    
    @abc.abstractclassmethod
    def get_winner(self):
        raise NotImplementedError("Implement get_winner function")

    @abc.abstractclassmethod
    def get_dimentions(self):
        raise NotImplementedError("Implement get_dimentions function")

    @abc.abstractclassmethod
    def get_state_as_input(self):
        raise NotImplementedError("Implement get_dimentions function")    

    # Debugging
    @abc.abstractclassmethod
    def get_legal_actions_bool(self):
        # Return legal actions as a 0 or 1, if the are are actually legal or not
        raise NotImplementedError("Implement get_legal_actions function")

    