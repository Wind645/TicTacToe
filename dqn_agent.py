import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import random
from collections import deque

class Q_net(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 9)
        # an idea about why the output dim is set to 9 is that I want the neural network to
        # receive an input of state and give out 9 Q values for each corresponding action
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
        
class ReplayBuffer:
    
    def __init__(self, capacity, batch_size):
        self.buffer = deque(maxlen=capacity)
        self.batch_size= batch_size
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def __len__(self):
        return len(self.buffer)
    
    def sample(self, device):
        experiences = random.sample(self.buffer, self.batch_size)
        state, action, reward, next_state, dones = zip(*experiences)
        state = torch.FloatTensor(state).to(device=device)
        action = torch.LongTensor(action).to(device=device)
        reward = torch.FloatTensor(reward).to(device=device)
        next_state = torch.FloatTensor(next_state).to(device=device)
        dones = torch.FloatTensor(dones).to(device=device)
        return state, action, reward, next_state, dones

class dqn_agent:
    
    def __init__(self, device, gamma = 0.95, lr = 0.01, buffer_capacity = 10000, epsilon_start = 1, 
                 epsilon_end = 0.01, batch_size = 64, target_update = 50):
        self.q_net = Q_net().to(device)
        self.fixed_q_net = Q_net().to(device)
        self.fixed_q_net.load_state_dict(self.q_net.state_dict())
        self.fixed_q_net.eval()
        self.replaybuffer = ReplayBuffer(buffer_capacity, batch_size)
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.num_updates = 0
        self.epsilons = torch.linspace(epsilon_start, epsilon_end, 10000)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr)  # 修复: 从self.Q_net改为self.q_net
        
    def flat_state(self, state):
        return state.reshape((-1))
    
    def flat_action(self, action): # change the action from a tuple to an index ranging from 0 to 8
        i, j = action
        return i * 3 + j
    
    def get_q_value(self, state):
        state_flattened = self.flat_state(state)
        Qs = self.q_net(state_flattened.unsqueeze(-1))
        return Qs
    
    def get_fixed_q_net(self, state):
        state_flattened = self.flat_state(state)
        Qs = self.fixed_q_net(state_flattened.unsqueeze(-1))
        return Qs
    
    def select_action(self, state, num_episodes=None):
        if num_episodes is None:
            num_episodes = 0
        
    
        valid_actions = []
        for i in range(3):
            for j in range(3):
                if state[i, j] == 0:  
                    valid_actions.append((i, j))
        
        if not valid_actions: 
            return None
        
        epsilon = self.epsilons[min(num_episodes // 10, len(self.epsilons)-1)]
        
    
        if np.random.random() < epsilon:
            return random.choice(valid_actions)
        
        
        state_tensor = torch.FloatTensor(self.flat_state(state)).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
            
    
        best_value = float('-inf')
        best_action = valid_actions[0]
        
        for action in valid_actions:
            action_idx = self.flat_action(action)
            if q_values[action_idx].item() > best_value:
                best_value = q_values[action_idx].item()
                best_action = action
                
        return best_action
    
    def update(self, info):
        state, action, next_state, reward, done = info  
    
        flat_state = self.flat_state(state)
        flat_next_state = self.flat_state(next_state)
        action_idx = self.flat_action(action)
    
        self.replaybuffer.add(flat_state, action_idx, reward, flat_next_state, done)
    
        if len(self.replaybuffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replaybuffer.sample(self.device)
    
        with torch.no_grad():
            next_q_values = self.fixed_q_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
    
        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
        loss = F.mse_loss(current_q_values, target_q_values)
    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        self.num_updates += 1
        if self.num_updates % self.target_update == 0:
            self.fixed_q_net.load_state_dict(self.q_net.state_dict())
    
    def save(self, path):
        """保存模型到指定路径"""
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'fixed_q_net_state_dict': self.fixed_q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_updates': self.num_updates
        }, path)
        print(f"模型已保存到 {path}")
    
    def load(self, path):
        """从指定路径加载模型"""
        if torch.cuda.is_available():
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
            
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.fixed_q_net.load_state_dict(checkpoint['fixed_q_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.num_updates = checkpoint['num_updates']
        self.q_net.to(self.device)
        self.fixed_q_net.to(self.device)
        print(f"模型已从 {path} 加载")
