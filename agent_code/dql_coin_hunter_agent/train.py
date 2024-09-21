from collections import namedtuple, deque

import pickle
from typing import List

import torch
import random
import copy

import numpy as np

import events as e
from .callbacks import state_to_features, ACTIONS, DQL_Model

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 5  # keep only ... last transitions
SAMPLE_SIZE = 3
GAMMA = 0.95
UPDATE_FREQ = 20
# RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
CLOSER_TO_COIN_EVENT = "CLOSER_TO_COIN"
FURTHER_FROM_COIN_EVENT = "FURTHER_FROM_COIN"


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
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

    self.target_model = DQL_Model(n_inputs = 225, n_hidden = 275, n_outputs = 4)
    self.target_model.load_state_dict(self.model.state_dict())
    self.target_model.train = False

    self.saved_out_model = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    #torch.autograd.set_detect_anomaly(True)
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
    distance_change = coin_distance_change(old_game_state, new_game_state)
    if distance_change < 0:
        events.append(CLOSER_TO_COIN_EVENT)
    elif distance_change > 0:
        events.append(FURTHER_FROM_COIN_EVENT)
    
    #reward = reward_from_events(self, events = events)

    # state_to_features is defined in callbacks.py
    #print("Extracted features:\n", state_to_features(old_game_state))
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    
    # sample
    if len(self.transitions) >= SAMPLE_SIZE:
        sampled_transitions = random.sample(list(self.transitions), SAMPLE_SIZE)
        #print('Sampeld trasitions',sampled_transitions)
        # compute loss
        q_loss = torch.tensor([0.])
        
        for transition in sampled_transitions:
            if e.SURVIVED_ROUND in events:
                y_j = transition[3] 
            else:
                y_j = transition[3] + GAMMA * torch.max(self.target_model(transition[2]))
            q_loss = q_loss + (y_j - self.model(transition[0])[ACTIONS.index(transition[1])])**2 #fix: replace self.output with new model


        # update Q
        self.optimizer.zero_grad()
        #print("Q_LOSS\n", q_loss)
        q_loss.backward(retain_graph=True)
        self.optimizer.step()

        # every C-steps: update Q^
        if new_game_state['step'] % UPDATE_FREQ == 0:
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


    self.transitions.clear()
    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.INVALID_ACTION: -.5,
        e.CLOSER_TO_COIN_EVENT: 0.1,
        e.FURTHER_FROM_COIN_EVENT: -0.05
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def coin_distance_change(old_game_state, new_game_state):

    _,_,_, old_pos = old_game_state["self"]
    _,_,_, new_pos = new_game_state["self"]

    coins = old_game_state["coins"]

    old_distances = []
    new_distances = []
    for coin_pos in coins:
        cx, cy = coin_pos
        old_distances += [np.abs(cx - old_pos[0]) + np.abs(cy - old_pos[1])]
        new_distances += [np.abs(cx - new_pos[0]) + np.abs(cy - new_pos[1])]
    
    distance_change = np.min(np.array(new_distances)) - np.min(np.array(old_distances))

    return distance_change
"""
def remove_random_elements(list11, list21, num_elements):
    list1 = list11
    list2 = list21
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length.")
    
    # Get a list of indices from which to remove elements
    indices = list(range(len(list1)))
    
    # Randomly choose indices to remove
    indices_to_remove = random.sample(indices, num_elements)
    
    # Remove elements at the same indices from both lists
    for idx in sorted(indices_to_remove, reverse=True):
        del list1[idx]
        del list2[idx]
    
    return list1, list2
"""