from collections import namedtuple, deque

import pickle
from typing import List
import numpy as np
import events as e
from .callbacks import state_to_features


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

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
    self.learning_rate = 0.7
    self.gamma = 0.95


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
    distance_change = coin_distance_change(old_game_state, new_game_state)
    if distance_change < 0:
        events.append(CLOSER_TO_COIN_EVENT)
    elif distance_change > 0:
        events.append(FURTHER_FROM_COIN_EVENT)
    
    reward = reward_from_events(self, events = events)
    
    new_max_q_value = 0
    if state_to_string(new_game_state) in self.qtable:
        new_max_q_value = np.max(self.qtable[state_to_string(new_game_state)])

    #print("trying to adress", len(state_to_string(old_game_state)))
    #print(state_to_string(old_game_state))
    #print("")
    self.qtable[state_to_string(old_game_state)][np.argwhere(ACTIONS == self_action)] += self.learning_rate * (
        reward + self.gamma * new_max_q_value - self.qtable[state_to_string(old_game_state)][np.argwhere(ACTIONS == self_action)])

    # state_to_features is defined in callbacks.py
    # self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))


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
    # self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.qtable, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.CLOSER_TO_COIN_EVENT: 0.05,
        e.FURTHER_FROM_COIN_EVENT: -0.05
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def state_to_string(game_state: dict) -> str:
    coins = game_state["coins"]
    _, _, _, pos = game_state["self"]

    # maybe optimize by sorting?

    return str(pos) + str(coins)

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