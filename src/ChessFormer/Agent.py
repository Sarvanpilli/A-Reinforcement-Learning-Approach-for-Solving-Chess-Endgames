import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from .Model import ChessTransformer

class ChessAgent:
    def __init__(self, state_size=3, action_size=32, d_model=64):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = ChessTransformer(d_model=d_model, num_actions=action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor(np.array([i[0] for i in minibatch])).to(self.device)
        actions = torch.LongTensor([i[1] for i in minibatch]).to(self.device)
        rewards = torch.FloatTensor([i[2] for i in minibatch]).to(self.device)
        next_states = torch.FloatTensor(np.array([i[3] for i in minibatch])).to(self.device)
        dones = torch.FloatTensor([i[4] for i in minibatch]).to(self.device)

        # Q(s, a)
        curr_Q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Q(s', a')
        next_Q = self.model(next_states).detach().max(1)[0]
        
        # Target = r + gamma * max Q(s', a')
        expected_Q = rewards + (1 - dones) * self.gamma * next_Q
        
        loss = self.criterion(curr_Q, expected_Q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save(self, filename):
        torch.save(self.model.state_dict(), filename)
        
    def load(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=self.device, weights_only=True))
