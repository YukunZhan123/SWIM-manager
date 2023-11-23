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
        print("critic_loss ", critic_loss.item())
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
        actor_loss.backward()
        self.actor_optimizer.step()


# Utility function
def calculate_utility(state, maxServers, maxServiceRate, RT_THRESHOLD):
    basicRevenue = 1
    optRevenue = 1.5
    serverCost = 10

    precision = 1e-5

    maxThroughput = maxServers * maxServiceRate

    # Unpacking state values (assuming state is [avgResponseTime, avgThroughput, arrivalRateMean, dimmer, avgServers])
    avgResponseTime = state[0] * 0.05  # Assuming state[0] is the average response time
    avgThroughput = state[1] * 6   # Assuming state[1] is the average throughput
    arrivalRateMean = state[2] * 13  # Assuming state[2] is the mean arrival rate
    dimmer = state[3]           # Assuming state[3] is the dimmer value
    avgServers = state[4] * 3   # Assuming state[4] is the average number of servers

    Ur = (arrivalRateMean * ((1 - dimmer) * basicRevenue + dimmer * optRevenue))
    Uc = serverCost * (maxServers - avgServers)
    UrOpt = arrivalRateMean * optRevenue

    utility = 0
    if avgResponseTime <= RT_THRESHOLD and Ur >= UrOpt - precision:
        utility = Ur - Uc
    elif avgResponseTime <= RT_THRESHOLD:
        utility = Ur
    else:
        utility = (max(0.0, arrivalRateMean - maxThroughput) * optRevenue) - Uc

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
batch_size = 16

manager = ActorCriticManager(state_size, action_size, epsilon)

if os.path.exists('actor.pth') and os.path.exists('critic.pth'):
    manager.actor.load_state_dict(torch.load('actor.pth'))
    manager.critic.load_state_dict(torch.load('critic.pth'))
else:
    print("No saved model weights found, initializing new models.")

reset()
iteration_counter = 0

maxServiceRate = 1/0.04452713
maxServers = 3
RT_THRESHOLD = 0.075  # Example response time threshold

while True:  # Replace with your specific condition
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("it:", iteration_counter)
    state = get_system_state()
    action = manager.select_action(state)
    done = perform_action(state, action_choices[action])
    next_state = get_system_state()
    reward = calculate_utility(next_state, maxServers, maxServiceRate, RT_THRESHOLD)

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