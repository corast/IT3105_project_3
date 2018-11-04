# We need an actor, so we can switch between agents to play the game,
# We want to be able to play as a human (command prompt), an Good AI vs an Bad AI.
# External AI (Interface), etc.


class Actor():
    def __init__(self):
        pass

    def get_action(self):
        # The actor make plays on an game, based on the game.
        # Should handle any game
        # This could either be MTC + ANN, or an interface (Keith's) or an player via terminal
        pass