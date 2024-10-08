import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.qtable = {}
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.qtable = pickle.load(file)
    
    # initialize q-table or q-network


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation

    self.logger.debug("Querying model for action.")
    # return np.random.choice(ACTIONS, p=self.model)
    game_string = state_to_string(game_state)
    if game_string in self.qtable:
        random_prob = .1
        if self.train and random.random() < random_prob:
            self.logger.debug("Choosing action purely at random.")
            # 80%: walk in any direction. 10% wait. 10% bomb.
            # maybe calculate and exclude moves that are not possible
            return np.random.choice(ACTIONS, p=[.25, .25, .25, .25])
        return ACTIONS[np.random.choice(
            np.flatnonzero(
                self.qtable[game_string] == np.max(self.qtable[game_string])))]

    else:
        # add state to dict
        self.qtable[game_string] = np.zeros(4)
        #print("added", len(game_string))
        #print(game_string)
        #print("")
        # filter out moves that are not possible
        return np.random.choice(ACTIONS, p=[.25, .25, .25, .25])


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)


def state_to_string(game_state: dict) -> str:
    coins = game_state["coins"]
    _, _, _, pos = game_state["self"]

    # maybe optimize by sorting?

    return str(pos) + str(coins)
