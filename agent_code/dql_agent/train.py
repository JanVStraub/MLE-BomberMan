from collections import namedtuple, deque

import pickle
from typing import List

import torch
import numpy as np
import random
import copy
import matplotlib.pyplot as plt

import events as e
from .callbacks import state_to_features, DQL_Model, ACTIONS, DEVICE

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
MEMORY_SIZE = 50000 # keep only ... last transitions
MEMORY_COUNTER = 0 # keep track of filled up space
GAMMA = 0.99
LR = 0.003
LR_GAMMA = 0.999
BATCH_SIZE = 100

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # loss functions
    self.loss_func = torch.nn.MSELoss()

    self.state_memory = np.zeros((MEMORY_SIZE, 4, 11, 11), dtype=np.float32)
    self.new_state_memory = np.zeros((MEMORY_SIZE, 4, 11, 11), dtype=np.float32)
    self.action_memory = np.zeros((MEMORY_SIZE), dtype=int) # we work with the indices of the actions
    self.reward_memory = np.zeros((MEMORY_SIZE), dtype=np.float32)
    self.terminal_state_memory = np.ones((MEMORY_SIZE), dtype=np.bool)

    self.model.train = True

    # set up optimizer and schduler
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
    self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=LR_GAMMA)

    self.target_model = DQL_Model()
    self.target_model.load_state_dict(self.model.state_dict())
    self.target_model.train = False


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """

    # add custom events
    if is_closer_to_coin(old_game_state, new_game_state):
        events.append(e.CLOSER_TO_COIN_EVENT)
    else:
        events.append(e.FURTHER_FROM_COIN_EVENT)
        self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # add entry to memories    
    index = MEMORY_COUNTER % MEMORY_SIZE
    self.state_memory[index] = state_to_features(old_game_state)
    self.new_state_memory[index] = state_to_features(new_game_state)
    self.action_memory[index] = ACTIONS.index(self_action)
    self.reward_memory[index] = reward_from_events(self, events)
    self.terminal_state_memory[index] = True
    # increase memory counter
    MEMORY_COUNTER += 1

    # train model with a batch of samples from the memory
    # set gradient to zero
    self.optimizer.zero_grad()
    # sample indices
    sample_size = min(MEMORY_COUNTER, MEMORY_SIZE)
    batch_size = min(MEMORY_COUNTER, BATCH_SIZE)
    indices = np.random.coice(sample_size, batch_size, replace=False)

    states_batch = torch.tensor(self.state_momory[indices]).to(DEVICE)
    new_states_batch = torch.tensor(self.new_state_memory[indices]).to(DEVICE)
    actions_batch = torch.tensor(self.action_memory[indices]).to(DEVICE)
    reward_batch = torch.tensor(self.reward_memory[indices]).to(DEVICE)
    terminal_state_batch = torch.tensor(self.terminal_state_memory[indices]).to(DEVICE)

    # compute outputs
    outputs = self.model(states_batch)[:, actions_batch]

    target_outputs = reward_batch + (int)(terminal_state_batch)* \
        GAMMA * torch.max(self.target_model(new_states_batch), dim=1)
    q_loss = self.loss_func(target_outputs, outputs).to(DEVICE)
    self.logger.debug(f"q-Loss = {q_loss}")

    # perform backward path
    q_loss.backward()
    self.optimizer.step()
    self.scheduler.step()


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # add entry to memories    
    index = MEMORY_COUNTER % MEMORY_SIZE
    self.state_memory[index] = state_to_features(old_game_state)
    self.action_memory[index] = ACTIONS.index(last_action)
    self.reward_memory[index] = reward_from_events(self, events)
    self.terminal_state_memory[index] = False
    # increase memory counter
    MEMORY_COUNTER += 1

    _, score, _, _ = last_game_state["self"]
    self.scores.append(score)

    # train model with a batch of samples from the memory
    # set gradient to zero
    self.optimizer.zero_grad()
    # sample indices
    sample_size = min(MEMORY_COUNTER, MEMORY_SIZE)
    batch_size = min(MEMORY_COUNTER, BATCH_SIZE)
    indices = np.random.coice(sample_size, batch_size, replace=False)

    states_batch = torch.tensor(self.state_momory[indices]).to(DEVICE)
    new_states_batch = torch.tensor(self.new_state_memory[indices]).to(DEVICE)
    actions_batch = torch.tensor(self.action_memory[indices]).to(DEVICE)
    reward_batch = torch.tensor(self.rewar_memory[indices]).to(DEVICE)
    terminal_state_batch = torch.tensor(self.terminal_state_memory[indices]).to(DEVICE)

    # compute outputs
    outputs = self.model(states_batch)[:, actions_batch]

    target_outputs = reward_batch + (int)(terminal_state_batch)* \
        GAMMA * torch.max(self.target_model(new_states_batch), dim=1)
    q_loss = self.loss_func(target_outputs, outputs).to(DEVICE)
    self.logger.debug(f"q-Loss = {q_loss}")

    # perform backward path
    q_loss.backward()
    self.optimizer.step()
    self.scheduler.step()

    # update target model
    self.target_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
    self.target_model.train = False

    plot_scores(self.scores)

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def plot_scores(scores):
    plt.figure()
    plt.title("scores while training")
    plt.plot(np.arange(1, len(scores)+1), scores, color="deepskyblue")
    plt.grid()
    plt.xlabel("round")
    plt.ylabel("score")
    plt.savefig("scores.pdf", format="pdf")
    plt.close()

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.WAITED: -0.1,
        e.COIN_COLLECTED: 1,
        e.COIN_FOUND: 0.4,
        e.GOT_KILLED: -15,
        e.KILLED_SELF: -15,
        e.BOMB_DROPPED: 0.1,
        e.KILLED_OPPONENT: 5,
        e.CRATE_DESTROYED: 0.1,
        e.INVALID_ACTION: -0.05,
        e.CLOSER_TO_COIN_EVENT: 0.1,
        e.FURTHER_FROM_COIN_EVENT: -0.12,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def is_closer_to_coin(old_game_state, new_game_state):

    old_distance = 34
    new_distance = 34
    
    x_old, y_old = old_game_state["self"][3]
    x_new, y_new = new_game_state["self"][3]

    for coin in old_game_state["coins"]:
        x, y = coin
        old_is_closer = (np.abs(x_old - x) + np.abs(y_old - y) < old_distance)
        new_is_closer = (np.abs(x_new - x) + np.abs(y_new - y) < old_distance)
        old_distance = (int)(old_is_closer) * (np.abs(x_old - x) + np.abs(y_old - y)) + (1 - (int)(old_is_closer)) * old_distance
        new_distance = (int)(new_is_closer) * (np.abs(x_new - x) + np.abs(y_new - y)) + (1 - (int)(new_is_closer)) * new_distance

    if old_distance == new_distance and new_distance == 0:
        return True

    return new_distance < old_distance
