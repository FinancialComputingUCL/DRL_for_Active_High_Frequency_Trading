import copy
import enum
from abc import ABC

import gym
import gym.spaces as space
import numpy as np


class Position(enum.Enum):
    neutral = 0
    long = 1
    short = -1

    def __int__(self):
        return self.value


def compute_m_to_m(position, price_scaled, buy_value_scaled, sell_value_scaled):
    m_to_m = 0

    if position == Position.short:
        m_to_m = price_scaled - buy_value_scaled
    if position == Position.long:
        m_to_m = sell_value_scaled - price_scaled

    return m_to_m


class StockEnv(gym.Env, ABC):
    def __init__(self, scaled_data, unscaled_data, evaluation_mode, args):

        self.evaluation_mode = evaluation_mode
        self.last_n_ticks = args.last_n_ticks
        self.use_m_t_m = args.use_m_t_m

        self.scaled_data = copy.deepcopy(scaled_data)
        self.unscaled_data = copy.deepcopy(unscaled_data)

        # Remove prices as they are not used in the state
        self.unscaled_snapshot_state = copy.deepcopy(self.unscaled_data).iloc[:,
                                       np.array([*range(20)]) * 2 + 1]
        self.scaled_snapshot_state = copy.deepcopy(self.scaled_data).iloc[:,
                                     np.array([*range(20)]) * 2 + 1]
        self.state = self.scaled_snapshot_state.iloc[:self.last_n_ticks, :]

        self.state_snapshot_scaled = self.state_snapshot_unscaled = []  # Will store whole prices and volumes for the period the agent is in
        self.buy_value = self.sell_value = self.buy_value_scaled = self.sell_value_scaled = 0  # Current scaled and unscaled buy and sell values
        self.day_profit = self.pos = 0
        self.price = self.price_scaled = 0  # Both the scaled and non-scaled price a trade was effectuated at (Ie. Buy value when going long, sell value when going short)
        self.m_to_m = 0
        self.opened = 0  # Keep track of when a position is opened
        self.closed = 0  # Keep track of when a position is closed
        self.closed_position = 0  # Keep track of the position that was closed

        self.position = Position.neutral  # The current position of the agent
        self.done = False
        self.size = self.unscaled_data.shape[0]

        spaces = {
            'volumes': gym.spaces.Box(low=0, high=1, shape=(1, 200)),
            'position': gym.spaces.Box(low=-1, high=1, shape=(1, 1))
        }

        if self.use_m_t_m:
            spaces.update({'m_to_m': gym.spaces.Box(low=0, high=1, shape=(1, 1))})

        self.observation_space = gym.spaces.Dict(spaces)
        self.action_space = space.Discrete(4)

    def step(self, action):

        self.pos += 1
        closed = False
        self.compute_state_features()
        self.m_to_m = reward = 0

        # If this is the last tick, close any open positions
        if self.last_n_ticks + self.pos == self.size:
            self.done = True
            if self.position == Position.short:
                action = 2
            elif self.position == Position.long:
                action = 0
            elif self.position == Position.neutral:
                action = 1

        # Open short position or close long position
        if action == 0:
            if self.position == Position.neutral:
                self.open_position(Position.short, self.sell_value, self.sell_value_scaled)
            elif self.position == Position.long:
                reward, closed = self.close_position(self.sell_value), True
            elif self.position == Position.short and not self.evaluation_mode:
                reward = -0.5 * (self.sell_value + self.buy_value)

        # Open long position or close short position
        elif action == 2:
            if self.position == Position.neutral:
                self.open_position(Position.long, self.buy_value, self.buy_value_scaled)
            elif self.position == Position.short:
                reward, closed = self.close_position(self.buy_value), True
            elif self.position == Position.long and not self.evaluation_mode:
                reward = -0.5 * (self.sell_value + self.buy_value)

        # Stop loss action
        elif action == 3:
            if not self.position.neutral == self.position:
                unrealised_profit = self.sell_value - self.price if self.position == Position.long else \
                    self.price - self.buy_value
                if self.day_profit + unrealised_profit < 0 and unrealised_profit < 0:  # If current trade is in red
                    # and the day is in red
                    reward, closed = self.close_position(self.buy_value if self.position == Position.short else
                                                         self.sell_value), True

                    self.done = True

                if not self.evaluation_mode:
                    reward = -0.5 * (self.sell_value + self.buy_value)

        next_state = self.create_state()
        return next_state, float(reward), self.done, {'closed': closed, 'open_pos': self.opened,
                                                      'closed_pos': self.closed, 'position': self.closed_position,
                                                      'action': action}

    def reset(self):

        self.state = self.scaled_snapshot_state.iloc[:self.last_n_ticks, :]
        self.state_snapshot_scaled = self.state_snapshot_unscaled = []

        self.opened = self.closed = self.price_scaled = self.m_to_m = 0
        self.price = self.day_profit = self.pos = 0
        self.buy_value = self.sell_value = self.buy_value_scaled = self.sell_value_scaled = 0

        self.done = False
        self.position = Position.neutral
        next_state = self.create_state()

        return next_state

    def close_position(self, value):
        reward = value - self.price if self.position == Position.long else self.price - value
        self.closed, self.closed_position = self.pos, self.position
        self.day_profit += reward
        self.position = Position.neutral
        self.price = self.price_scaled = 0
        return reward

    def open_position(self, position, value, scaled_value):
        self.position, self.opened = position, self.pos
        self.price, self.price_scaled = value, scaled_value

    def compute_state_features(self):
        self.state = self.scaled_snapshot_state.iloc[self.pos:self.last_n_ticks + self.pos, :]
        self.state_snapshot_scaled = self.scaled_data.iloc[self.pos:self.last_n_ticks + self.pos, :]
        self.state_snapshot_unscaled = self.unscaled_data.iloc[self.pos:self.last_n_ticks + self.pos, :]

        self.buy_value = self.state_snapshot_unscaled.iloc[-1, 0]
        self.sell_value = self.state_snapshot_unscaled.iloc[-1, 2]
        self.buy_value_scaled = self.state_snapshot_scaled.iloc[-1, 0]
        self.sell_value_scaled = self.state_snapshot_scaled.iloc[-1, 2]

    def create_state(self):

        state = {'volumes': np.expand_dims(np.array(self.state).flatten(), axis=0),
                 'position': np.expand_dims(np.array([int(self.position)]), axis=1)}

        if self.use_m_t_m:
            self.m_to_m = compute_m_to_m(self.position, self.price_scaled, self.buy_value_scaled,
                                         self.sell_value_scaled)
            state.update({'m_to_m': np.expand_dims(np.array([self.m_to_m]), axis=1)})

        return state
