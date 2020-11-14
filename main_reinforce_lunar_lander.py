import numpy as np
import gym
from reinforce import PGagent
import matplotlib.pyplot as plt
from gym import wrappers

if __name__ == '__main__':
    agent = PGagent(GAMMA=0.99,ALPHA=0.001, input_dims=[8],n_actions=4, fc1_dims=128,fc2_dims=128)
    env = gym.make('LunarLander-v2')
    score_history = []
    score = 0
    num_episodes = 10000
    for i in range(num_episodes):
        print('episode: ', i,'score: ', score)
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_,reward,done,_ = env.step(action)
            agent.store_rewards(reward)
            observation = observation_
            score += reward
        score_history.append(score)
        if (i+1)%10 == 0:
            print("Learning")
            agent.learn(update = "UPDATE",n_episodes= 10)
        else:
            agent.learn("NOT UPDATE",n_episodes = 10)


    for i in range(10):
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            observation = observation_
            env.render()
