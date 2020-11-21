import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

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
        self.state_memory = []
        self.next_state_memory = []
        self.terminal_memory = []
        #self.action_batch = torch.tensor([])

        self.actor_loss = 0
        self.critic_loss = 0

    def choose_action(self,observation):

        probabilities = F.softmax(self.actor.forward(observation))
        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        l = torch.tensor(log_prob,dtype = torch.float).to('cpu')

        #try:
        #    self.action_batch = torch.cat((self.action_batch,l),0)
        #except:
        #    self.action_batch = l

        self.action_memory.append(log_prob)
        #try:
        #    print("a")
        #    self.action_batch = torch.cat((self.action_batch,log_prob),axis = 0)
        #except:
        #    print("b")
        #    self.action_batch = log_prob

        return action.item()

    def store_rewards(self,reward):
        self.reward_memory.append(reward)

    def store_states(self,state,state_,done):

        self.state_memory.append(state)
        self.next_state_memory.append(state_)
        self.terminal_memory.append(float(done))

    def learn(self,update,n_episodes):

        state_batch = np.array(self.state_memory)
        state_batch = torch.from_numpy(state_batch)

        next_state_batch = np.array(self.next_state_memory)
        next_state_batch = torch.from_numpy(next_state_batch)

        reward_batch = np.array(self.reward_memory)
        reward_batch = reward_batch.reshape(reward_batch.shape[0],1)
        reward_batch = torch.from_numpy(reward_batch)

        #print(self.action_batch.shape)

        terminal_batch = np.array(self.terminal_memory)
        terminal_batch = torch.from_numpy(terminal_batch)
        terminal_batch = terminal_batch.reshape(terminal_batch.shape[0],1)
        #print(self.action_batch.shape,state_batch.shape)

        #print(reward_batch.shape,terminal_batch.shape,next_state_batch.shape,state_batch.shape)
        dataset = TensorDataset(reward_batch,terminal_batch,next_state_batch,state_batch)
        trainloader = DataLoader(dataset,batch_size = 32,shuffle=False)

        for i,data in enumerate(trainloader,0):
            r_batch,t_batch,next_s_batch,s_batch = data
            value_f = self.critic.forward(s_batch)
            next_value_f = self.critic.forward(next_s_batch)
            delta = (r_batch + (1-t_batch)*self.gamma*next_value_f) - value_f
            D = delta**2
            self.critic_loss += torch.sum(D)

        if update == "UPDATE":
            self.critic.optimizer.zero_grad()
            self.critic_loss = (1/n_episodes)*self.critic_loss
            self.critic_loss.backward()
            self.critic.optimizer.step()
            self.critic_loss = 0

        #A1 = self.critic.forward(state_batch)
        #A1 = A1.detach()
        #A2 = self.critic.forward(next_state_batch)
        #A2 = A2.detach()

        #action_batch = torch.tensor(self.action_memory)

        T = 0
        while T<(len(self.reward_memory)-1):
            value_f = self.critic.forward(state_batch[T]).detach()
            next_value_f = self.critic.forward(next_state_batch[T]).detach()
            delta = (self.reward_memory[T]+self.gamma*next_value_f.detach() - value_f.detach())
            self.actor_loss += -(delta*self.action_memory[T])
            T += 1
        deltaT = (self.reward_memory[T] - self.critic.forward(state_batch[T]).detach())
        self.actor_loss += -(deltaT*self.action_memory[T])

        if update == "UPDATE":
            self.actor.optimizer.zero_grad()
            self.actor_loss = (1/n_episodes)*self.actor_loss
            self.actor_loss.backward()
            self.actor.optimizer.step()
            self.actor_loss = 0

        self.reward_memory = []
        self.action_memory = []
        self.state_memory = []
        self.next_state_memory = []
        self.terminal_memory = []
