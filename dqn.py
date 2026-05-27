import torch
import torch.nn as nn
from collections import deque
import numpy as np
import random
import torch.nn.functional as F
import gymnasium as gym
from typing import Optional

class DQNNetwork(nn.Module):    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQNNetwork, self).__init__()

        self.seq = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.seq(state)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)
    
class DQNAgent:
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128, memory_size: int = 10000, batch_size: int = 64, gamma: float = 0.99, lr: float = 1e-3, device: Optional[str] = None):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.policy_net = DQNNetwork(
            state_size,
            action_size,
            hidden_size
        ).to(device)

        self.target_net = DQNNetwork(
            state_size,
            action_size,
            hidden_size
        ).to(device)

        self.update_target_network()

        self.memory = ReplayBuffer(memory_size)

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=lr
        )

    def predict(self, state: np.ndarray) -> np.ndarray:
        state_t = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_t).cpu().numpy()[0]

        return q_values
    
    def epsilon_greedy(self, state: np.ndarray, epsilon: float):
        if np.random.random() < epsilon:
            return np.random.randint(self.action_size)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            q_values = self.policy_net(state_tensor)

            return q_values.argmax().item()
        
    def train_step(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.policy_net(states)

        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]

            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.mse_loss(
            current_q_values,
            target_q_values
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.memory.push(state, action, reward, next_state, done)

    def sample_memory(self):
        return self.memory.sample(self.batch_size)
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str):
        torch.save(self.policy_net, path)

    def load(self, path: str):
        self.policy_net = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.to(self.device)

        self.target_net = torch.load(path, map_location=self.device, weights_only=False)
        self.target_net.to(self.device)

        self.update_target_network()

def train(env: gym.Env, agent: DQNAgent, n_episodes: int, epsilon: float, epsilon_decay: float, epsilon_min: float, target_update_steps: int, print_every: int = 10):
    episode_rewards = []
    episode_losses = []
    steps = 0

    for episode in range(n_episodes):
        state, _ = env.reset()

        done = False
        total_reward = 0
        losses = []
        while not done:
            action = agent.epsilon_greedy(state, epsilon)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, done)

            loss = agent.train_step()

            if loss > 0:
                losses.append(loss)

            steps += 1

            if steps % target_update_steps == 0:
                agent.update_target_network()

            state = next_state
            total_reward += reward

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        avg_loss = np.mean(losses) if losses else 0

        episode_rewards.append(total_reward)
        episode_losses.append(avg_loss)

        avg_reward = np.mean(episode_rewards[-100:])

        if episode % print_every == 0:
            print(
                f"Episode {episode:4d} | "
                f"Reward: {total_reward:6.1f} | "
                f"Avg(100): {avg_reward:6.2f} | "
                f"Epsilon: {epsilon:.3f} | "
                f"Loss: {avg_loss:.4f}"
            )

    return episode_rewards, episode_losses

def play_episode(env: gym.Env, agent: DQNAgent):
    frames = []
    state, _ = env.reset()

    episode_steps = episode_reward = 0
    done = False
    while not done:
        frames.append(env.render())
        action = agent.epsilon_greedy(state, 0)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        episode_steps += 1
        episode_reward += reward

        state = next_state
    return episode_reward, episode_steps, frames
