from collections import namedtuple, deque

import pickle
from typing import List
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
import copy

import events as e
from .callbacks import state_to_features, DQL_Model, ACTIONS

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 300  # keep only ... last transitions
GAMMA = 0.9
BATCH_SIZE = 16
UPDATE_FREQ = 8
TARGET_UPDATE_FREQ = 64
LR = 0.001
LR_GAMMA = 0.9999 #finally learned at 0.99999

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
    self.transitions.clear()
    self.model.train = True
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
    self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=LR_GAMMA)

    self.target_model = DQL_Model()
    self.target_model.to(self.DEVICE)
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
    #self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    #game event after each 50 rounds the agent revieves a penalty
    #if new_game_state['round'] % 10:
    #    events.append(e.TOO_LONG_EVENT)
    # Idea: Add your own events to hand out rewards
    if is_closer_to_coin(old_game_state, new_game_state):
        events.append(e.CLOSER_TO_COIN_EVENT)
    else:
        events.append(e.FURTHER_FROM_COIN_EVENT)

    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

    if new_game_state['step'] % UPDATE_FREQ == 0 and len(self.transitions) > BATCH_SIZE:
        self.optimizer.zero_grad()  # Clear gradients before batch
    
        total_loss = 0  # Accumulate loss over the batch
        batch = random.sample(list(self.transitions), BATCH_SIZE)
        
        for transition in batch:
            if transition[2] is not None:
                y_j = transition[3] + GAMMA * torch.max(self.target_model(transition[2]))
            else:
                y_j = transition[3]
            
            # Calculate Q-loss for this transition
            q_loss = (y_j - self.model(transition[0])[0][ACTIONS.index(transition[1])])**2
            
            # Accumulate the loss
            total_loss = q_loss + total_loss
        
        # Backpropagate after processing the whole batch
        total_loss.backward()
        
        # Update model parameters and learning rate
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
    #self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    #self.scores.append(last_game_state["self"][1])
    self.optimizer.zero_grad()

    #if not e.SURVIVED_ROUND in events:
    if len(self.transitions) < BATCH_SIZE:
        batch = random.sample(list(self.transitions), len(self.transitions))
    else:
        batch = random.sample(list(self.transitions), BATCH_SIZE)


    total_loss = 0#torch.tensor([0.], requires_grad=True)  # Accumulate loss over the batch
    for transition in batch:
        y_j = transition[3]
        #self.logger.debug(f"Model output: {self.model.out[0][ACTIONS.index(self_action)]}")
        q_loss = (y_j - self.model(transition[0])[0][ACTIONS.index(transition[1])])**2
        total_loss = q_loss + total_loss
    # Backpropagate after processing the whole batch
    total_loss.backward()
    
    # Update model parameters and learning rate
    self.optimizer.step()
    self.scheduler.step()
    if e.KILLED_OPPONENT in events:
        print("Killed opponent")
    for param_group in self.optimizer.param_groups:
        current_lr = param_group['lr']
        self.logger.debug(f"Learning rate after scheduler step: {current_lr}")


    # every C-steps: update Q^
    if last_game_state['step'] % TARGET_UPDATE_FREQ == 0:
        self.target_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        self.target_model.train = False
    
    _, score, _, _ = last_game_state["self"]
    step = last_game_state["step"]
    self.scores.append(score)
    self.step.append(step)
    plot_scores(self.scores, self.step)
    
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
        e.WAITED: -0.1,
        #e.MOVED_DOWN: -.01,
        #e.MOVED_LEFT: -.01,
        #e.MOVED_RIGHT: -.01,
        #e.MOVED_UP: -.01,
        e.COIN_COLLECTED: 15,
        e.COIN_FOUND: 0.4,
        e.GOT_KILLED: -20,
        e.KILLED_SELF: -20,
        e.BOMB_DROPPED: 0.1,
        e.KILLED_OPPONENT: 40,
        e.CRATE_DESTROYED: 5,
        #e.INVALID_ACTION: -0.05,
        e.CLOSER_TO_COIN_EVENT: 3,
        e.FURTHER_FROM_COIN_EVENT: -0.1,
        #e.TOO_LONG_EVENT: -1
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    #self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
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

def plot_scores(scores, step):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    ax1.plot(np.arange(len(scores)), step, color = "blue")
    ax2.plot(np.arange(len(scores)), scores, color = "orange")

    ax1.set_xlabel("Round")
    ax1.set_ylabel("Steps needed")

    ax2.set_ylabel("Score")

    fig.savefig("scores.pdf", format="pdf")
    plt.close()
