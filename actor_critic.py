import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Actor(nn.Module):

    def __init__(self,ALPHA,input_dims,n_actions,fc1_dims,fc2_dims):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(*input_dims,fc1_dims)
        self.fc2 = nn.Linear(fc1_dims,fc2_dims)
        self.fc3 = nn.Linear(fc2_dims,n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)
        self.device = torch.device('cpu')
        self.to(self.device)

    def forward(self,x):
        x = torch.tensor(x,dtype = torch.float).to(self.device)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class Critic(nn.Module):

    def __init__(self,ALPHA,input_dims,fc1_dims,fc2_dims):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(*input_dims,fc1_dims)
        self.fc2 = nn.Linear(fc1_dims,fc2_dims)
        self.fc3 = nn.Linear(fc2_dims,1)

        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)
        self.device = torch.device('cpu')
        self.to(self.device)

    def forward(self,x):
        x = torch.tensor(x,dtype = torch.float).to(self.device)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class ActorCriticAgent(object):

    def __init__(self,GAMMA,ALPHA,input_dims,n_actions,fc1_dims,fc2_dims):

        self.actor = Actor(ALPHA,input_dims,n_actions,fc1_dims,fc2_dims)
        self.critic = Critic(ALPHA,input_dims,fc1_dims,fc2_dims)
        self.gamma = GAMMA

        self.reward_memory = []
        self.action_memory = []
        self.value_memory = []

        self.actor_loss = 0
        self.critic_loss = 0

    def choose_action(self,observation):

        probabilities = F.softmax(self.actor.forward(observation))
        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        self.action_memory.append(log_prob)

        return action.item()

    def store_rewards(self,reward):
        self.reward_memory.append(reward)

    def store_values(self,observation):
        value = self.critic.forward(observation)
        self.value_memory.append(value)

    def learn(self,update,n_episodes):

        T = 0
        while T<(len(self.reward_memory)-1):
            delta = (self.reward_memory[T]+self.gamma*self.value_memory[T+1] - self.value_memory[T])
            self.critic_loss += delta**2
            T += 1
        deltaT = (self.reward_memory[T] - self.value_memory[T])
        self.critic_loss += deltaT**2

        T = 0
        while T<(len(self.reward_memory)-1):
            delta = (self.reward_memory[T]+self.gamma*self.value_memory[T+1].detach() - self.value_memory[T].detach())
            self.actor_loss += -(delta*self.action_memory[T])
            T += 1
        deltaT = (self.reward_memory[T] - self.value_memory[T].detach())
        self.actor_loss += -(deltaT*self.action_memory[T])

        if update == "UPDATE":

            self.critic.optimizer.zero_grad()
            self.critic_loss = (1/n_episodes)*self.critic_loss
            self.critic_loss.backward()
            self.critic.optimizer.step()
            self.critic_loss = 0

            self.actor.optimizer.zero_grad()
            self.actor_loss = (1/n_episodes)*self.actor_loss
            self.actor_loss.backward()
            self.actor.optimizer.step()
            self.actor_loss = 0

        self.reward_memory = []
        self.action_memory = []
        self.value_memory = []
