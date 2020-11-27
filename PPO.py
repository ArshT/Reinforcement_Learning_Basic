import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):

  def __init__(self,input_dims,n_actions,fc1_dims,fc2_dims):
    super(ActorCritic, self).__init__()

    self.fc1_action = nn.Linear(input_dims,fc1_dims)
    self.fc2_action = nn.Linear(fc1_dims,fc2_dims)

    self.fc1_value = nn.Linear(input_dims,fc1_dims)
    self.fc2_value = nn.Linear(fc1_dims,fc2_dims)

    self.action_layer = nn.Linear(fc2_dims,n_actions)
    self.value_layer = nn.Linear(fc2_dims,1)

    self.to(device)

  def forward(self,observation):
    try:
      state = torch.from_numpy(observation).float().to(device)
    except:
      state = observation

    x = F.tanh(self.fc1_action(state))
    x = F.tanh(self.fc2_action(x))
    action_probs = self.action_layer(x)

    y = F.tanh(self.fc1_value(state))
    y = F.tanh(self.fc2_value(y))
    state_value = self.value_layer(y)

    return action_probs,state_value


class PPO(object):

  def __init__(self,input_dims,n_actions,fc1_dims,fc2_dims,lr,betas,gamma,K_epochs,eps_clip):
    self.lr = lr
    self.betas = betas
    self.gamma = gamma
    self.eps_clip = eps_clip
    self.K_epochs = K_epochs

    self.policy = ActorCritic(input_dims,n_actions,fc1_dims,fc2_dims)
    self.optimizer = torch.optim.Adam(self.policy.parameters(),lr = lr,betas=betas)
    self.policy_old = ActorCritic(input_dims,n_actions,fc1_dims,fc2_dims).to(device)
    self.policy_old.load_state_dict(self.policy.state_dict())

    self.MseLoss = nn.MSELoss()

    self.actions = []
    self.states = []
    self.logprobs = []
    self.rewards = []
    self.is_terminals = []

  def clear_memory(self):

    del self.actions[:]
    del self.states[:]
    del self.logprobs[:]
    del self.rewards[:]
    del self.is_terminals[:]

  def act(self,state):
    action_probs,_ = self.policy_old.forward(state)
    action_probs = F.softmax(action_probs)
    dist = Categorical(action_probs)
    action = dist.sample()

    state = torch.tensor(state).float().to(device)
    self.states.append(state)
    self.actions.append(action)
    self.logprobs.append(dist.log_prob(action))

    return action.item()

  def evaluate(self,state,action):

    action_probs,state_value = self.policy.forward(state)
    action_probs = F.softmax(action_probs)
    dist = Categorical(action_probs)
    action_logprobs = dist.log_prob(action)
    dist_entropy = dist.entropy()

    return action_logprobs, torch.squeeze(state_value), dist_entropy


  def update(self):

      rewards = []
      discounted_reward = 0
      for reward, is_terminal in zip(reversed(self.rewards), reversed(self.is_terminals)):
        if is_terminal:
          discounted_reward = 0
        discounted_reward = reward + (self.gamma * discounted_reward)
        rewards.insert(0, discounted_reward)

      rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
      rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

      old_states = torch.stack(self.states).to(device).detach()
      old_actions = torch.stack(self.actions).to(device).detach()
      old_logprobs = torch.stack(self.logprobs).to(device).detach()

      for _ in range(self.K_epochs):

            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

      self.policy_old.load_state_dict(self.policy.state_dict())
      self.clear_memory()
