import os
import pickle
import random
from collections import deque

import numpy as np
import torch


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']# , 'WAIT', 'BOMB']




class DQL_Model(torch.nn.Module):

    def __init__(self, n_hidden: int = 128, n_outputs: int = 4, dropout: float = 0.3):
        super().__init__()
        # input size: 1x4x17x17
        # new input size 1x4x9x9
        self.conv_1 = torch.nn.Conv2d(
            in_channels=3, out_channels=128, kernel_size=5)
        self.act_1 = torch.nn.ReLU()
        self.drop_1 = torch.nn.Dropout(dropout)
        # output size: 1x256x13x13
        # new outpus size: 1x128x5x5

        self.conv_2 = torch.nn.Conv2d(
            in_channels=128, out_channels=32, kernel_size=3)
        self.act_2 = torch.nn.ReLU()
        # output size: 1x64x11x11
        # new output size: 1x32x3x3
        #self.maxPool = torch.nn.MaxPool2d(kernel_size=2, ceil_mode=True)
        # output size: 1x64x6x6

        self.flat = torch.nn.Flatten()
        # output size: 1152
        # new outpus size: 288
        self.lin = torch.nn.Linear(288, n_outputs)

        self.out = None
        
        #self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Cuda is available:", torch.cuda.is_available())
        # initialize weights
        torch.nn.init.xavier_uniform_(self.conv_1.weight)
        # print("CONV 1 SIZE:", self.conv_1.weight.data.size(), "---------------------")
        torch.nn.init.xavier_uniform_(self.conv_2.weight)
        torch.nn.init.xavier_uniform_(self.lin.weight)


    def forward(self, inputs):
 
        output = self.act_1(self.conv_1(inputs))
        output = self.drop_1(output)

        output = self.act_2(self.conv_2(output))
        #output = self.maxPool(output)

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
    self.DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.forward_backward_toggle = False
    self.current_round = 0
    self.scores = []
    self.max_epsilon = 1.0           
    self.min_epsilon = 0.05           
    self.decay_rate = 5e-4
    
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        #torch.manual_seed(42)
        self.model = DQL_Model()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
    self.model.to(self.DEVICE)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    # self.logger.debug("Querying model for action.")
    self.model.out = self.model(state_to_features(game_state))
    self.forward_backward_toggle = True
    # recommended_action = ACTIONS[np.random.choice(np.flatnonzero(self.model.out == torch.max(self.model.out)))]
    
    # Gather information about the game state
    arena = game_state['field']
    _, _, _, (x, y) = game_state['self']
    # Check which moves make sense at all
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
    # if (x, y) in valid_tiles: valid_actions.append('WAIT')
    # valid_actions.append('BOMB') # maybe edit when training with bombs

    epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon)*np.exp(-self.decay_rate*game_state["round"])
    
    #epsilon = .95 * .99 ** game_state["round"]
    if self.train and random.random() < epsilon:
        self.logger.debug("Choosing action purely at random.")
        random_choice = np.random.choice(valid_actions)#, p=[.2, .2, .2, .2, .1, .1]) #Choose random valid action
        return random_choice
    self.logger.debug("Querying model for action.")
    # Choosing the action with the highest Q-value if its not in valid_actions
    # Get indices of valid actions
    valid_indices = [ACTIONS.index(action) for action in valid_actions]
    #print("Valid indices",valid_indices)
    valid_q_values = self.model.out[0][valid_indices]
    # Filter self.model.out to only consider valid actions
   

    # Select the valid action with the highest Q-value
    # Not shure if this is the right way
    best_valid_action_idx = torch.argmax(valid_q_values).item()
    return ACTIONS[valid_indices[best_valid_action_idx]]


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

    arena_size_x = len(game_state["field"][0])
    arena_size_y = len(game_state["field"][1])

    # create coin map
    coin_map = np.zeros((arena_size_x,arena_size_y))
    for (x,y) in game_state["coins"]:
        coin_map[x,y] = 1.
    channels.append(coin_map)

    # create players map (positive value: self, negative: other players)
    players_map = np.zeros((arena_size_x,arena_size_y))
    _, s, b, (x,y) = game_state["self"]
    players_map[x,y] = s + 1
    explosion_map[x,y] = -(int)(b)

    for player in game_state["others"]:
        _, s, b, (x,y) = player
        players_map[x,y] = -s - 1
        explosion_map[x,y] = -(int)(b)
    
    #channels.append(explosion_map)
    channels.append(players_map)

    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.array(np.stack(channels))
    # and return them as a vector
    #return torch.tensor(np.array([stacked_channels])).to(DEVICE, dtype=torch.float32)
    return torch.tensor(np.array([stacked_channels]), dtype=torch.float32).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
