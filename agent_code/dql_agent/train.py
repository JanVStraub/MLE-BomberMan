from collections import namedtuple, deque

import pickle
from typing import List

import torch
import copy

import events as e
from .callbacks import state_to_features, DQL_Model, ACTIONS

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1  # keep only ... last transitions
GAMMA = 0.99
UPDATE_FREQ = 3
TARGET_UPDATE_FREQ = 10

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
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

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
    # if ...:
    #     events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

    q_loss = torch.tensor([0.])
    if e.SURVIVED_ROUND in events:
        y_j = self.transitions[-1][3]
    else:
        y_j = self.transitions[-1][3] + GAMMA * torch.max(self.target_model(self.transitions[-1][2]))
    q_loss = (y_j - self.output[0][ACTIONS.index(self_action)])**2

    # accumulate loss
    q_loss.backward(retain_graph=True)

    # update Q every something steps
    if new_game_state['step'] % UPDATE_FREQ == 0:
        self.optimizer.zero_grad()
        self.optimizer.step()

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
        e.COIN_FOUND: 0.4,
        e.GOT_KILLED: -20,
        e.BOMB_DROPPED: 0.1,
        e.KILLED_OPPONENT: 5,
        e.CRATE_DESTROYED: 0.1,
        e.INVALID_ACTION: -1,
        e.CLOSER_TO_COIN_EVENT: 0.1,
        e.FURTHER_FROM_COIN_EVENT: -0.1,
        # PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def is_closer_to_coin(old_game_state, new_game_state):



    return False
