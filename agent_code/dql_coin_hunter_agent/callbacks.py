import os
import pickle
import random
import torch, torchvision

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
    self.epsion = 0
    self.min_epsilon = 0.005
    self.max_epsilon = 1
    self.decay_rate = 0.005
    #if self.train or not os.path.isfile("my-saved-model.pt"):
    if self.train or not os.path.isfile("my-saved-model.pt"): 
        self.logger.info("Setting up model from scratch.")
        self.model = DQL_Model(n_inputs = 225, n_hidden = 275, n_outputs = 4)
    
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


class DQL_Model(torch.nn.Module):

    def __init__(self, n_inputs: int, n_hidden: int, n_outputs: int):
        super().__init__()
        self.lin_1 = torch.nn.Linear(n_inputs, n_hidden)
        self.relu = torch.nn.ReLU()
        self.lin_2 = torch.nn.Linear(n_hidden, n_outputs)

        # print("DEFAULT TYPE OF LINEAR\n",self.lin_1.weight.dtype)

        self.lin_1.data = torch.normal(0, 1, size=self.lin_1.weight.data.size())
        self.lin_2.data = torch.normal(0, 1, size=self.lin_2.weight.data.size())


    def forward(self, inputs):
        lin_1_out = self.lin_1(inputs)
        hid_input = self.relu(lin_1_out)
        output = self.lin_2(hid_input)
        return output


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    round = game_state["round"]
    self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon)*np.exp(-self.decay_rate*round)


    # todo Exploration vs exploitation
    #self.epsilon = .1
    if self.train and random.random() < self.epsilon:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS)

    self.logger.debug("Querying model for action.")
    self.output = self.model(state_to_features(game_state)) 
    """
    IDEA: save outpus in a list or something so when computing the qloss function we 
    don't need to compute the same thing twice -- analogue to the deque funtcion for the qmax calculation
    """

    """
    IDEA: Instead of allways choosing the highes value. Convert Qvalues into probability and then choos based on that
    """
    q_values = self.output

    # Convert Q-values to probabilities using softmax
    probabilities = torch.softmax(q_values, dim=0)

    # Choose an action based on these probabilities
    chosen_action_index = np.random.choice(len(ACTIONS), p=probabilities.detach().numpy())

    # Get the corresponding action
    return ACTIONS[chosen_action_index]
    
    #return ACTIONS[np.random.choice(np.flatnonzero(self.output == torch.max(self.output)))]


def state_to_features(game_state: dict) -> np.array:
    """
    Convert Game state dict into a height x with array with game state info for traing. Its entries are:
    2 coin
    1 crate
    0 free tial
    -1 stone wall
    -2 agent
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    coins = game_state["coins"]
    _, _, _, pos = game_state["self"]

    new_field = torch.tensor(game_state['field'][1:-1,1:-1].copy())
    new_field[pos[0]-1, pos[1]-1] = 3   

    for (x, y) in coins:
        new_field[x-1, y-1] = 2

    # maybe filter out walls as inputs
    return new_field.reshape(-1).to(torch.float32)
"""
def state_to_features_alt(game_state: dict) -> np.array:
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    coins = game_state["coins"]
    _, _, _, pos = game_state["self"]
"""