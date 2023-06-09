import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import requests
import json


# Define trading environment
class TradingEnv:
    def __init__(self, window_size=10, steps=100):
        self.window_size = window_size
        self.steps = steps
        self.reset()

    def reset(self):
        self.data = json.loads(requests.get('http://192.168.50.100:6000/binance/btcusdt').text)
        self.current_step = self.window_size
        self.bought_price = 0
        self.sold_price = 0
        self.bought = False
        return self.data

    def _take_action(self, action):
        current_price = self.data['bookTicker']['a']

        reward = 0

        if action == 0 and not self.bought:  # Buy

            if (self.sold_price > current_price):
                reward += self.sold_price / current_price
            else:
                reward -= self.sold_price / current_price

            self.bought_price = current_price
            self.bought = True

        elif action == 1 and self.bought:  # Sell

            reward = 0

            if (self.bought_price < current_price):
                reward += self.bought_price / current_price
            else:
                reward -= self.bought_price / current_price

            self.bought = False
            self.sold_price = current_price

        elif action == 2:  # Wait
            pass

        return reward

    def step(self, action):
        reward = self._take_action(action)
        self.current_step += 1
        done = self.current_step == self.steps
        obs = self.data
        return obs, reward, done, {}

# Define the RL model
class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.000001, gamma=0.95, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.999, target_update_freq=10, batch_size=32000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=32000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.timestep = 0
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())

    def _build_model(self):
        input_layer = Input(shape=self.state_size)
        x = Dense(32, activation="relu")(input_layer)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(32, activation="relu")(x)
        output_layer = Dense(self.action_size, activation="linear")(x)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        return model

# Initialize trading environment
env = TradingEnv()

# Set hyperparameters

state_size = env.window_size * 4
action_size = 3
num_episodes = 100

# Initialize agent
agent = DQNAgent(state_size, action_size)

# Set constants
total_reward = 0

# Train the model
for episode in range(num_episodes):

    state = env.reset()

    state_array = np.array(list(state.values()))
    state_size = len(state_array)
    next_state_array = np.reshape(state_array, [1, state_size])

    done = False

    while not done:
        if np.random.rand() <= agent.epsilon:
            action = np.random.randint(action_size)
        else:
            q_values = agent.model.predict(next_state_array, verbose=0)
            action = np.argmax(q_values[0])

        next_state, reward, done, _ = env.step(action)

        next_state_array = np.array(list(next_state.values()))
        state_size = len(next_state_array)
        next_state_array = np.reshape(next_state_array, [1, state_size])
        total_reward += reward

        # Store experience in replay memory
        agent.memory.append((state, action, reward, next_state_array, done))

        # Update the state
        state = next_state_array

        # Perform experience replay if memory is full
        if len(agent.memory) >= agent.batch_size:
            batch = random.sample(agent.memory, agent.batch_size)
            for state, action, reward, next_state_array, done in batch:
                if done:
                    target = reward
                else:
                    target = reward + agent.gamma * np.amax(agent.model.predict(state, verbose=0)[0])

                target_f = agent.model.predict(state, verbose=0)
                target_f[0][action] = target
                agent.model.fit(state, target_f, epochs=1, verbose=1)

        # Decay epsilon
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

print("Total Reward = {}".format(total_reward))
