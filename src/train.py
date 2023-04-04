import numpy as np
from agent import DQNAgent
from env import TradingEnv

# Initialize trading environment
env = TradingEnv()

state_size = len(env.data)
action_size = 3
num_episodes = 100

# Initialize agent
agent = DQNAgent(state_size, action_size)

# Set constants
total_profit = 0
i = 0


class Algorithm:
    def __init__(self):
        self.agent = DQNAgent(state_size, action_size)

    def act(self, state):
        if np.random.rand() <= self.agent.epsilon:
            return np.random.choice(action_size)
        q_values = self.agent.model.predict(np.array([state]), verbose=0)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, done):
        if len(self.agent.memory) >= self.agent.batch_size:
            self.agent.memory.pop(0)
        self.agent.memory.append((state, action, reward, done))

    def replay(self):
        if len(self.agent.memory) < self.agent.batch_size:
            return
        batch = self.agent.memory[-self.agent.batch_size:]
        states = [sample[0] for sample in batch]
        q_values = self.agent.model.predict(states, verbose=0)
        next_states = [sample[3] for sample in batch]
        next_q_values = self.agent.model.predict(next_states, verbose=0)
        for i, (state, action, reward, done) in enumerate(batch):
            if done:
                q_values[i][action] = reward
            else:
                q_values[i][action] = reward + self.agent.gamma * np.max(next_q_values[i])
        self.agent.model.fit(states, q_values, verbose=0)
        if self.agent.epsilon > self.agent.epsilon_min:
            self.agent.epsilon *= self.agent.epsilon_decay


alg = Algorithm()

bought = False

btc = 1
usdt = 40000

sold_price = 0
bought_price = 0

while True:
    state = env.data
    current_price = env.dataDict['bookTicker']['a']
    action = alg.act(state)

    reward = 0

    if action == 0 and not bought and usdt >= current_price:

        usdt -= current_price * 0.0045
        btc += 0.0045
        bought_price = current_price
        if 0 < sold_price > current_price:
            reward += sold_price / current_price
        elif 0 < sold_price <= current_price:
            reward -= sold_price / current_price
        bought = True

    elif action == 1 and bought and btc >= 0:

        usdt += current_price * 0.0045
        btc -= 0.0045
        sold_price = current_price
        if 0 < bought_price < current_price:
            reward += bought_price / current_price
        elif 0 < bought_price >= current_price:
            reward -= bought_price / current_price
        bought = False

    alg.remember(list(state), action, reward, False)
    alg.replay()

    total_profit += reward
    print(f"Step: {i+1} | Total profit: {total_profit} | BTC: {btc} | USDT: {usdt}")
    i += 1

