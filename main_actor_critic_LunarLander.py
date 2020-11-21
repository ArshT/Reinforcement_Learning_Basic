import numpy as np
import gym
from actor_critic import ActorCriticAgent
import matplotlib.pyplot as plt
from gym import wrappers

if __name__ == '__main__':
    agent = ActorCriticAgent(GAMMA=0.99,ALPHA=0.0003, input_dims=[4],n_actions=2, fc1_dims=2048,fc2_dims=512)
    #gym.envs.register(
    #id='CartPole-v2',
    #entry_point='gym.envs.classic_control:CartPoleEnv',
    #tags={'wrapper_config.TimeLimit.max_episode_steps': 500},
    #reward_threshold=475,
    #)
    env = gym.make("LunarLander-v2")
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
            agent.store_states(observation,observation_,done)
            observation = observation_
            score += reward
        score_history.append(score)
        if (i+20)%20 == 0:
            print("Learning")
            agent.learn(update = "UPDATE",n_episodes= 20)
        else:
            agent.learn("NOT UPDATE",n_episodes = 20)

    print("")
    for i in range(20):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            observation = observation_
            env.render()
            score += reward
        print('episode: ', i,'score: ', score)
