import torch as T
import torch
import torch.nn as nn
import torch.optim as Optim
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym

#Class for Noise
class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)



#Class for Critic
class CriticNetwork(nn.Module):

    def __init__(self,beta,input_dims,fc1_dims,fc2_dims,n_actions,device):
        super(CriticNetwork, self).__init__()

        self.fc1 = nn.Linear(*input_dims,fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(fc1_dims)

        self.fc2 = nn.Linear(fc1_dims,fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2,f2)
        nn.init.uniform_(self.fc2.bias.data, -f2,f2)
        self.bn2 = nn.LayerNorm(fc2_dims)

        self.action_value = nn.Linear(n_actions,fc2_dims)

        f3 = 0.003
        self.q = nn.Linear(fc2_dims, 1)
        nn.init.uniform_(self.q.weight.data, -f3, f3)
        nn.init.uniform_(self.q.bias.data, -f3, f3)

        self.device = torch.device(device)
        self.to(self.device)


    def forward(self,state,action):

        state_value = F.relu(self.bn1(self.fc1(state)))
        state_value = self.bn2(self.fc2(state_value))

        action_value = F.relu(self.action_value(action))

        state_action_value = F.relu(torch.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value




#Class for Actor
class ActorNetwork(nn.Module):

    def __init__(self,alpha,input_dims,fc1_dims,fc2_dims,n_actions,device):
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(*input_dims,fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(fc1_dims)

        self.fc2 = nn.Linear(fc1_dims,fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2,f2)
        nn.init.uniform_(self.fc2.bias.data, -f2,f2)
        self.bn2 = nn.LayerNorm(fc2_dims)

        f3 = 0.003
        self.fc3 = nn.Linear(fc2_dims,n_actions)
        nn.init.uniform_(self.fc3.weight.data, -f3, f3)
        nn.init.uniform_(self.fc3.bias.data, -f3, f3)

        self.device = torch.device(device)
        self.to(self.device)

    def forward(self,state):

        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = torch.tanh(self.fc3(x))

        return x





class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env,epsilon,device,gamma=0.99,
                 n_actions=2, max_size=1000000, layer1_size=400,
                 layer2_size=300, batch_size=64,eps_dec = 0.99,eps_end = 0.01):
        self.ALPHA = alpha
        self.BETA = beta
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.mem_size = max_size
        self.EPSILON = epsilon
        self.EPS_DEC = eps_dec
        self.EPS_END = eps_end
        self.device = device

        self.actor = ActorNetwork(self.ALPHA,input_dims,layer1_size,layer2_size,n_actions,device)
        self.target_actor = ActorNetwork(self.ALPHA,input_dims,layer1_size,layer2_size,n_actions,device)
        self.critic = CriticNetwork(self.BETA,input_dims,layer1_size,layer2_size,n_actions,device)
        self.target_critic = CriticNetwork(self.BETA,input_dims,layer1_size,layer2_size,n_actions,device)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=beta)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=alpha)

        self.update_network_parameters(tau=1)

        self.state_memory = np.zeros((self.mem_size, *input_dims))
        self.next_state_memory = np.zeros((self.mem_size, *input_dims))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def choose_action(self, observation):
        state = torch.tensor(observation,dtype=torch.float).to(self.actor.device)
        mu = self.actor.forward(state)
        mu_prime = mu + torch.tensor(self.noise(),dtype=torch.float).to(self.actor.device)*self.EPSILON

        mu_prime = torch.clamp(mu_prime,-1,1)

        return mu_prime.cpu().detach().numpy()

    def store_transition(self,state,action,reward,next_state,terminal):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.next_state_memory[index] = next_state
        self.terminal_memory[index] = 1 - int(terminal)

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        state_batch = self.state_memory[batch]
        action_batch = self.action_memory[batch]
        reward_batch = self.reward_memory[batch]
        next_state_batch = self.next_state_memory[batch]
        terminal_batch = self.terminal_memory[batch]

        return state_batch,action_batch,reward_batch,next_state_batch,terminal_batch

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        states, actions, rewards, next_states, terminals = self.sample_buffer(self.batch_size)

        rewards = torch.tensor(rewards, dtype=torch.float).to(self.critic.device)
        rewards = rewards.view(self.batch_size,1)

        terminals = torch.tensor(terminals).to(self.critic.device)
        terminals = terminals.view(self.batch_size,1)

        next_states = torch.tensor(next_states, dtype=torch.float).to(self.critic.device)

        actions = torch.tensor(actions, dtype=torch.float).to(self.critic.device)

        states = torch.tensor(states, dtype=torch.float).to(self.critic.device)


        target_actions = self.target_actor.forward(next_states)
        next_critic_value = self.target_critic.forward(next_states, target_actions)
        critic_value = self.critic.forward(states, actions)

        target = rewards + self.gamma*next_critic_value*terminals
        target = target.view(self.batch_size, 1)

        self.critic_optimizer.zero_grad()
        critic_loss = F.mse_loss(critic_value,target)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        current_actions = self.actor.forward(states)
        actor_loss = -self.critic.forward(states, current_actions)
        actor_loss.mean().backward()
        self.actor_optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                      (1-tau)*target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                      (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)


class ActionNormalizer(gym.ActionWrapper):
    """Rescale and relocate the actions."""

    def action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (-1, 1) to (low, high)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (low, high) to (-1, 1)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)

        return action
