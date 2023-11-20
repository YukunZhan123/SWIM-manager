import sys
import time
import os
import socket
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from swim_api import get_system_state, perform_action

torch.autograd.set_detect_anomaly(True)

# Define Actor network architecture
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, state):
        x = self.network(state)
        return self.log_softmax(x)

# Define Critic network architecture
class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        return self.network(state)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Actor-Critic Manager
class ActorCriticManager:
    def __init__(self, state_size, action_size, epsilon, actor_lr=0.005, critic_lr=0.005):
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.epsilon = epsilon
        self.gamma = 0.99

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            action_probs = self.actor(state_tensor).numpy()
        if random.random() < self.epsilon:
            print("random explore")
            action = np.random.choice(len(action_probs))
            self.epsilon *= 0.99
        else:
            print("choose best action")
            action = np.argmax(action_probs)
        return action

    def update_batch(self, states, actions, rewards, next_states, dones):
        print("reward ", reward)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        values = self.critic(states)
        next_values = self.critic(next_states).detach()
        td_targets = rewards + self.gamma * next_values * (1 - dones)

        critic_loss = (td_targets - values).pow(2).mean()
        print("critic_loss ", critic_loss)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute the actor loss
        log_probs = self.actor(states)
        action_log_probs = log_probs.gather(1, actions).squeeze(1)
        advantages = (td_targets - values.detach()).squeeze()
        actor_loss = -action_log_probs * advantages
        actor_loss = actor_loss.mean()  # Ensure actor_loss is a scalar by averaging over all losses
        print("actor_loss ", actor_loss.item())  # Use .item() to get the actual value if you want to print it
        self.actor_optimizer.zero_grad()
        actor_loss.backward()  # This should not raise the error anymore
        self.actor_optimizer.step()

# Utility function
def calculate_utility(state):
    # Define target and threshold levels for response time and throughput
    TARGET_RT = 0.06  # Target response time (lower is better)
    TARGET_TP = 7  # Target throughput (higher is better)


    # Calculate a combined throughput
    combined_tp = state[1] * 6
    combined_rt = state[0] * 0.05

    # Utility is higher when the combined response time is lower than the target, and combined throughput meets the target
    rt_utility = max(0, TARGET_RT - combined_rt) / TARGET_RT
    tp_utility = min(1, combined_tp / TARGET_TP)

    # Cost efficiency is assumed to be inversely proportional to the number of servers
    cost_utility = 1 / (state[-1] * 5) if state[-1] else 0


    # The overall utility is a weighted sum of response time utility, throughput utility, and cost utility
    utility = rt_utility * 0.4 + tp_utility * 0.4 + cost_utility * 0.2
    return utility


def reset():
    print("resetting environment")
    host = "127.0.0.1"
    port = 4242
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn = s.connect((host, port))
    s.sendall(b'set_dimmer 0.5')
    s.recv(1024)


if len(sys.argv) == 2:
    epsilon = float(sys.argv[1])
else:
    epsilon = 1

# Main execution loop
state_size = 5
action_choices = [["add", 0], ["remove", 0], ["nothing", 0.25], ["nothing", -0.25], ["nothing", 0], ["add", 0.25], ["add", -0.25], ["remove", 0.25], ["remove", -0.25]]
action_size = 9
replay_buffer = ReplayBuffer(capacity=10000)
batch_size = 32

manager = ActorCriticManager(state_size, action_size, epsilon)

if os.path.exists('actor.pth') and os.path.exists('critic.pth'):
    manager.actor.load_state_dict(torch.load('actor.pth'))
    manager.critic.load_state_dict(torch.load('critic.pth'))
else:
    print("No saved model weights found, initializing new models.")

reset()
iteration_counter = 0

while True:  # Replace with your specific condition
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("it:", iteration_counter)
    state = get_system_state()
    action = manager.select_action(state)
    done = perform_action(state, action_choices[action])
    time.sleep(1)
    next_state = get_system_state()
    reward = calculate_utility(next_state)

    replay_buffer.push(state, action, reward, next_state, done)
    if len(replay_buffer) > batch_size:
        experiences = replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        manager.update_batch(states, actions, rewards, next_states, dones)

    iteration_counter += 1
    if iteration_counter % 30 == 0:
        torch.save(manager.actor.state_dict(), 'actor.pth')
        torch.save(manager.critic.state_dict(), 'critic.pth')

    if done:
        reset()