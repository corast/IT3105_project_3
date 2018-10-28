import abc #Abstract Base Classes
class State():
    def __init__(self): # TODO: add common states in parent class.
        pass
    @abc.abstractclassmethod
    def game_result(self):
        raise NotImplementedError("Implement game_result function")

    @abc.abstractclassmethod
    def is_game_over(self):
        """ Return True if game is completed, by looking at the state """
        raise NotImplementedError("Implement is_game_over function")