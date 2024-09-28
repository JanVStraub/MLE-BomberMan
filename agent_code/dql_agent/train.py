from collections import namedtuple, deque

import pickle
from typing import List

import torch
import numpy as np
import copy

import events as e
from .callbacks import state_to_features, DQL_Model, ACTIONS

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1  # keep only ... last transitions
GAMMA = 0.95
UPDATE_FREQ = 3
TARGET_UPDATE_FREQ = 10
LR = 0.01
LR_GAMMA = 0.999

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.model.train = True
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
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    if is_closer_to_coin(old_game_state, new_game_state):
        events.append(e.CLOSER_TO_COIN_EVENT)
    else:
        events.append(e.FURTHER_FROM_COIN_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

    # if new_game_state['step'] % UPDATE_FREQ == 0:
    self.optimizer.zero_grad()

    q_loss = torch.tensor([0.])
    y_j = self.transitions[-1][3] + GAMMA * torch.max(self.target_model(self.transitions[-1][2]))
    q_loss = (y_j - self.model.out[0][ACTIONS.index(self_action)])**2

    # accumulate loss
    q_loss.backward()
    self.forward_backward_toggle = False

    # update Q every something steps
    # if new_game_state['step'] % UPDATE_FREQ == 0:
    self.optimizer.step()
    self.scheduler.step()

    # every C-steps: update Q^
    if new_game_state['step'] % TARGET_UPDATE_FREQ == 0:
        self.target_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        self.target_model.train = False


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

    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    self.scores.append(last_game_state["self"][1])

    if not e.SURVIVED_ROUND in events:

        self.optimizer.zero_grad()

        q_loss = torch.tensor([0.])
        y_j = self.transitions[-1][3]
        q_loss = (y_j - self.model.out[0][ACTIONS.index(last_action)])**2

        # calculate loss
        if self.forward_backward_toggle == False:
            print("-------------------forward backward toggle violation!----------------")
        else:
            q_loss.backward()

            self.optimizer.step()
            self.scheduler.step()

            # every C-steps: update Q^
            if last_game_state['step'] % TARGET_UPDATE_FREQ == 0:
                self.target_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
                self.target_model.train = False

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.

    Maybe the rewards are too strong, try decreasing the weight
    """
    game_rewards = {
        e.COIN_COLLECTED: 5,
        #e.COIN_FOUND: 0.4,
        e.GOT_KILLED: -5,
        e.KILLED_SELF: -5,
        e.BOMB_DROPPED: -1,
        #e.KILLED_OPPONENT: 10,
        #e.CRATE_DESTROYED: 0.1,
        #e.INVALID_ACTION: -1,
        e.CLOSER_TO_COIN_EVENT: 0.1,
        #e.FURTHER_FROM_COIN_EVENT: -0.1,
        #e.WAITED: -0.1

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
