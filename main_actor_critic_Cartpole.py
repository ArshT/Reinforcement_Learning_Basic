import numpy as np
import gym
from actor_critic import ActorCriticAgent
import matplotlib.pyplot as plt
from gym import wrappers

if __name__ == '__main__':
    agent = ActorCriticAgent(GAMMA=0.99,ALPHA=0.00003, input_dims=[4],n_actions=2, fc1_dims=2048,fc2_dims=512)
    env = gym.make("CartPole-v0")
    score_history = []
    score = 0
    num_episodes = 2500
    for i in range(num_episodes):
        print('episode: ', i,'score: ', score)
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_,reward,done,_ = env.step(action)
            agent.store_rewards(reward)
            agent.store_values(observation)
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
