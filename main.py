import time
import os
import socket
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from swim_api import get_system_state, perform_action  # Replace with your actual SWIM API calls

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
            nn.Linear(256, action_size),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, state):
        return self.network(state)

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

# Actor-Critic Manager
class ActorCriticManager:
    def __init__(self, state_size, action_size, epsilon=1, actor_lr=0.005, critic_lr=0.005):
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)
        # Set custom learning rates for the actor and critic optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.epsilon = epsilon
        self.gamma = 0.99  # Discount factor for future rewards

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

    def update(self, state, action, reward, next_state, done):
        print("reward ", reward)
        state_tensor = torch.FloatTensor(state)
        next_state_tensor = torch.FloatTensor(next_state)
        reward_tensor = torch.FloatTensor([reward])
        done_tensor = torch.FloatTensor([done])

        # Calculate the critic's estimate of the state's value and next state's value
        value = self.critic(state_tensor)
        next_value = self.critic(next_state_tensor).detach()

        # Calculate the temporal difference target
        td_target = reward_tensor + self.gamma * next_value * (1 - done_tensor)

        # Compute the critic loss
        critic_loss = (td_target - value).pow(2).mean()
        print("critic_loss ", critic_loss)
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)  # Default is retain_graph=False
        self.critic_optimizer.step()

        # Compute the actor loss
        probs = self.actor(state_tensor).unsqueeze(0)
        action_tensor = torch.LongTensor([action]).unsqueeze(1)
        action_probs = probs.gather(1, action_tensor).squeeze(1)

        # Negative log-likelihood loss
        actor_loss = -torch.log(action_probs) * (td_target - value.detach()).squeeze()
        print("actor_loss ", actor_loss)

        # Reset gradients and perform a backward pass for the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


# Utility function
def calculate_utility(state):
    # Define target and threshold levels for response time and throughput
    TARGET_RT = 0.5  # Target response time (lower is better)
    TARGET_TP = 100  # Target throughput (higher is better)


    # Calculate a combined throughput
    combined_tp = state[1]
    combined_rt = state[0]

    # Utility is higher when the combined response time is lower than the target, and combined throughput meets the target
    rt_utility = max(0, TARGET_RT - combined_rt) / TARGET_RT
    tp_utility = min(1, combined_tp / TARGET_TP)

    # Cost efficiency is assumed to be inversely proportional to the number of servers
    cost_utility = 1 / state[-1] if state[-1] else 0


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



# Real-time execution loop
state_size = 5  # Size of the state vector
action_choices = [["add", 0], ["remove", 0], ["nothing", 0.25], ["nothing", -0.25], ["nothing", 0], ["add", 0.25], ["add", -0.25], ["remove", 0.25], ["remove", -0.25]]
action_size = 9  # Number of actions


# Define paths to your saved model files
actor_model_path = 'actor.pth'
critic_model_path = 'critic.pth'

# Initialize your manager
manager = ActorCriticManager(state_size, action_size)

# Check if both the actor and critic model files exist
if os.path.exists(actor_model_path) and os.path.exists(critic_model_path):
    print("Loading saved model weights.")
    manager.actor.load_state_dict(torch.load(actor_model_path))
    manager.critic.load_state_dict(torch.load(critic_model_path))
else:
    print("No saved model weights found, initializing new models.")


reset()

# Add a counter for iterations
iteration_counter = 0

while True:  # Replace with the condition appropriate for your application
    # Monitor

    state = get_system_state()

    # Plan
    action = manager.select_action(state)
    print(action)


    # Execute
    done = perform_action(state, action_choices[action])  # Implement this function
    time.sleep(5)
    next_state = get_system_state()

    # Analyze
    reward = calculate_utility(next_state)

    # Update the manager
    manager.update(state, int(action), reward, next_state, done)

    iteration_counter += 1  # Increment the counter
    if iteration_counter % 50 == 0:
        # Save the actor and critic networks
        torch.save(manager.actor.state_dict(), f'actor.pth')
        torch.save(manager.critic.state_dict(), f'critic.pth')

    if done:  # Implement the logic to determine if the episode has ended
        reset()
        continue
