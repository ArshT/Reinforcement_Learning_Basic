import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PolicyNet(nn.Module):

    def __init__(self,ALPHA,input_dims,n_actions,fc1_dims,fc2_dims):
        super(PolicyNet, self).__init__()

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

class PGagent(object):
    def __init__(self,GAMMA,ALPHA,input_dims,n_actions,fc1_dims,fc2_dims):

        self.policy = PolicyNet(ALPHA,input_dims,n_actions,fc1_dims,fc2_dims)

        self.reward_memory = []
        self.action_memory = []
        self.gamma = GAMMA
        self.loss = 0

    def choose_action(self,observation):
        probabilities = F.softmax(self.policy.forward(observation))
        action_probs = torch.distributions.Categorical(probabilities) #Convert probabilities into a distribution
        action = action_probs.sample() #Sample from a distribution
        log_prob = action_probs.log_prob(action) #Log prob of distribution for that action
        self.action_memory.append(log_prob)

        return action.item()

    def store_rewards(self,reward):
        self.reward_memory.append(reward)

    def learn(self,update,n_episodes):
        self.policy.optimizer.zero_grad()

        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        for t in range(len(self.reward_memory)): #Convert rewards to reward to go
            discount = 1
            G_sum = 0
            for k in range(t,len(self.reward_memory)):
                G_sum += discount*self.reward_memory[k] #Discounted reward sum
                discount *= self.gamma
            G[t] = G_sum


        G = torch.tensor(G,dtype=torch.float).to(self.policy.device) #Reward to go created

        for g, logprob in zip(G, self.action_memory):
            self.loss += -g * logprob #Element-wise loss addition

        if update == "UPDATE": #Using n_episodes samples for update
            self.loss = (1/n_episodes)*self.loss #Total loss
            self.loss.backward() #Backprop
            self.policy.optimizer.step() #gradient step
            self.loss = 0

        self.action_memory = []
        self.reward_memory = []
