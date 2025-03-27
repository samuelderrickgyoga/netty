import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Main and target networks
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.update_counter = 0
    
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def train(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return 0
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values.detach())
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Periodically update target network
        self.update_counter += 1
        if self.update_counter % 100 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def save(self, filepath):
        torch.save({
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint['epsilon']

def preprocess_data(filepath):
    """
    Preprocess IoT sensor data for DQN
    
    Args:
        filepath (str): Path to CSV file
    
    Returns:
        tuple: Preprocessed states, normalized states
    """
    # Load data
    data = pd.read_csv(filepath)
    
    # Separate features
    states = data.drop(['action', 'reward', 'done'], axis=1).values
    
    # Normalize states
    states_normalized = (states - states.mean(axis=0)) / states.std(axis=0)
    
    return states, states_normalized

def train_dqn(filepath, episodes=1000, max_steps=200):
    """
    Train DQN on IoT sensor data
    
    Args:
        filepath (str): Path to sensor data CSV
        episodes (int): Number of training episodes
        max_steps (int): Maximum steps per episode
    
    Returns:
        DQNAgent: Trained agent
    """
    # Preprocess data
    states, states_normalized = preprocess_data(filepath)
    
    # Initialize agent
    state_dim = states_normalized.shape[1]
    action_dim = 4  # Example: 4 possible actions
    
    agent = DQNAgent(state_dim, action_dim)
    
    # Training loop
    episode_rewards = []
    for episode in range(episodes):
        state_idx = np.random.randint(len(states_normalized))
        state = states_normalized[state_idx]
        
        episode_reward = 0
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state)
            
            # Simulate next state and reward (replace with actual environment logic)
            next_state_idx = np.random.randint(len(states_normalized))
            next_state = states_normalized[next_state_idx]
            reward = np.random.rand()  # Placeholder reward
            done = step == max_steps - 1
            
            # Store in replay buffer
            agent.replay_buffer.add(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.train()
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # Periodic reporting
        if episode % 100 == 0:
            print(f"Episode {episode}, Avg Reward: {np.mean(episode_rewards[-100:]):.2f}, Epsilon: {agent.epsilon:.2f}")
    
    return agent

# Example usage (uncomment and provide actual filepath)
# if __name__ == "__main__":
#     agent = train_dqn('path/to/your/iot_sensor_data.csv')
#     agent.save('dqn_model.pth')