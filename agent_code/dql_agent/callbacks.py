import os
import pickle
import random
from collections import deque

import numpy as np
import torch


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

DEVICE = 'cpu'


class DQL_Model(torch.nn.Module):

    def __init__(self, n_hidden: int = 128, n_outputs: int = 6, dropout: float = 0.3):
        super().__init__()
        # input size: 17x17 (maybe use different channels for coins or players)
        self.conv_1 = torch.nn.Conv2d(
            in_channels=4, out_channels=25, kernel_size=5)
        self.act_1 = torch.nn.ReLU()
        self.drop_1 = torch.nn.Dropout(dropout)
        # output size: 25x13x13

        self.conv_2 = torch.nn.Conv2d(
            in_channels=25, out_channels=16, kernel_size=3)
        self.act_2 = torch.nn.ReLU()
        # output size: 16x11x11
        self.maxPool = torch.nn.MaxPool2d(kernel_size=2)
        # output size: 16x5x5

        self.flat = torch.nn.Flatten()
        # output size: 400

        self.lin_1 = torch.nn.Linear(400, n_hidden)
        self.act_3 = torch.nn.ReLU()
        self.drop_2 = torch.nn.Dropout(dropout)

        self.lin_2 = torch.nn.Linear(n_hidden, n_outputs)

        self.out = None


        # initialize weights
        torch.nn.init.xavier_uniform_(self.conv_1.weight)
        torch.nn.init.xavier_uniform_(self.conv_2.weight)
        torch.nn.init.xavier_uniform_(self.lin_1.weight)
        torch.nn.init.xavier_uniform_(self.lin_2.weight)


    def forward(self, inputs):
 
        output = self.act_1(self.conv_1(inputs))
        output = self.drop_1(output)

        output = self.act_2(self.conv_2(output))
        output = self.maxPool(output)

        output = self.act_3(self.lin_1(self.flat(output)))
        output = self.drop_2(output)
        output = self.lin_2(output)

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
    self.bomb_history = deque([], 5)
    self.current_round = 0
    
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        torch.manual_seed(42)
        self.model = DQL_Model()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
    self.model.to(DEVICE)

def reset_self(self):
    self.bomb_history = deque([], 5)

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    self.logger.debug("Querying model for action.")
    self.output = self.model(state_to_features(game_state))
    recommended_action = ACTIONS[np.random.choice(np.flatnonzero(self.model.out == torch.max(self.model.out)))]

    # todo Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    
    # check for special cases, e.g. running away from bombs,
    # collect last coin or last steps of round, etc., tactical suicide
    
    """
    Inspired from the rule based agent:
    """
    # Check if we are in a different round
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]
    # Gather information about the game state
    arena = game_state['field']
    _, _, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    #coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)
    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] < 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
    if recommended_action in valid_actions:
        if recommended_action == 'BOMB':
            self.bomb_history.append((x, y))
        return recommended_action
    elif valid_actions:
        action_to_take = random.choice(valid_actions)
        if action_to_take == 'BOMB':
            self.bomb_history.append((x, y)) 
        return action_to_take
    else:
        action_to_take_rand = random.choice(ACTIONS)
        if action_to_take_rand == 'BOMB':
            self.bomb_history.append((x, y)) 
        return action_to_take_rand



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
    channels.append(game_state["field"])
    explosion_map = game_state["explosion_map"]

    # create coin map
    coin_map = np.zeros((17,17))
    for (x,y) in game_state["coins"]:
        coin_map[x,y] = 1.
    channels.append(coin_map)

    # create players map (positive value: self, negative: other players)
    players_map = np.zeros((17,17))
    _, s, b, (x,y) = game_state["self"]
    players_map[x,y] = s + 1
    explosion_map[x,y] = -(int)(b)

    for player in game_state["others"]:
        _, s, b, (x,y) = player
        players_map[x,y] = -s - 1
        explosion_map[x,y] = -(int)(b)
    
    channels.append(explosion_map)
    channels.append(players_map)

    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return torch.tensor([stacked_channels]).to(DEVICE, dtype=torch.float32)
