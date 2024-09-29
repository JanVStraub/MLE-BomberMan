import os
import pickle
import random
from collections import deque

import numpy as np
import torch


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SEED = 42


class DQL_Model(torch.nn.Module):

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        # input size: 1x4x11x11
        self.conv_1 = torch.nn.Conv2d(
            in_channels=3, out_channels=256, kernel_size=5)
        self.act_1 = torch.nn.ReLU()
        self.drop_1 = torch.nn.Dropout(dropout)
        # output size: 1x256x7x7

        self.conv_2 = torch.nn.Conv2d(
            in_channels=256, out_channels=64, kernel_size=3)
        # output size: 1x64x7x7

        self.act_2 = torch.nn.ReLU()
        self.maxPool = torch.nn.MaxPool2d(kernel_size=2, ceil_mode=True)
        # output size: 1x64x4x4

        self.flat = torch.nn.Flatten()
        # output size: 1024

        self.lin = torch.nn.Linear(1024, len(ACTIONS))

        self.out = None

        # initialize weights
        torch.nn.init.xavier_uniform_(self.conv_1.weight)
        torch.nn.init.xavier_uniform_(self.conv_2.weight)
        torch.nn.init.xavier_uniform_(self.lin.weight)


    def forward(self, inputs):
 
        output = self.act_1(self.conv_1(inputs))
        output = self.drop_1(output)

        output = self.act_2(self.conv_2(output))
        output = self.maxPool(output)

        output = self.lin(self.flat(output))
        self.out = output

        return self.out


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
    self.scores = []
    
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        torch.manual_seed(SEED)
        self.model = DQL_Model()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
    self.model.to(DEVICE)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    # self.logger.debug("Querying model for action.")
    self.model.out = self.model(state_to_features(game_state).unsqueeze(1))
    self.forward_backward_toggle = True
    
    # check for valid moves
    arena = game_state['field']
    _, _, _, (x, y) = game_state['self']

    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] < 1)): # Maybe remove the explosion map check?
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    valid_actions.append('BOMB') # maybe edit when training with bombs

    epsilon = 0.01 + 0.99 * np.exp(-game_state['round']/1000)
    if self.train and random.random() < epsilon:
        self.logger.debug("Choosing action purely at random.")
        random_choice = np.random.choice(valid_actions)
        return random_choice
    self.logger.debug("Querying model for action.")
    # Choosing the action with the highest Q-value if its not in valid_actions
    # Get indices of valid actions
    valid_indices = [ACTIONS.index(action) for action in valid_actions]
    valid_q_values = self.model.out[0][valid_indices]

    # Select the valid action with the highest Q-value
    best_valid_action_idx = torch.argmax(valid_q_values).item()
    return ACTIONS[valid_indices[best_valid_action_idx]]


def state_to_features(game_state: dict) -> torch.tensor:
    """
    Converts the game state to the input of the model

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # view as 11x11 image around the agent - the agent will always be in the middle
    _, s, b, (x,y) = game_state['self']

    channels = []
    channels.append(np.pad(game_state["field"], pad_width=4)[x-1:x+10, y-1:y+10])
    explosion_map = game_state["explosion_map"]
    # add negative value at centre (players position) if agent can place bomb
    explosion_map[x,y] = -(int)(b)

    # create coin map
    coin_map = np.zeros((17,17))
    for (x,y) in game_state["coins"]:
        coin_map[x,y] = 1.
    channels.append(np.pad(coin_map,pad_width=4)[x-1:x+10, y-1:y+10])

    # create players map (positive value: self, negative: other players)
    players_map = np.zeros((17,17))
    players_map[x,y] = s + 1

    for player in game_state["others"]:
        _, s_o, b_0, (x_o,y_o) = player
        players_map[x_o,y_o] = -s_o - 1
        # value -1 if player at that position can drop bomb
        explosion_map[x_o,y_o] = -(int)(b_o)
    
    explosion_map = np.pad(explosion_map, pad_width=4)[x-1:x+10, y-1:y+10]
    channels.append(explosion_map)
    channels.append(np.pad(players_map, pad_width=4)[x-1:x+10, y-1:y+10])

    stacked_channels = torch.tensor(np.array(np.stack(channels))).to(DEVICE, dtype=torch.float32)
    # shape: 4x11x11
    return stacked_channels
