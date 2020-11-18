import gym
from dqn import DeepQNetwork, Agent
import numpy as np
from gym import wrappers

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    brain = Agent(gamma=0.99, epsilon=1.0, n_actions=4,batch_size=128,
                  input_dims=[8],alpha=0.0003,replace = 64)

    scores = []
    eps_history = []
    num_games = 500
    score = 0

    for i in range(num_games):
        if i % 10 == 0 and i > 0:
            avg_score = np.mean(scores[max(0, i-10):(i+1)])
            print('episode: ', i,'score: ', score,
                 ' average score %.3f' % avg_score,
                'epsilon %.3f' % brain.EPSILON)
        else:
            print('episode: ', i,'score: ', score)
        eps_history.append(brain.EPSILON)
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = brain.chooseAction(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            brain.storeTransition(observation, action, reward, observation_,done)
            observation = observation_
            brain.learn()

        scores.append(score)

    for i in range(10):
        done = False
        observation = env.reset()
        while not done:
            action = brain.chooseAction(observation)
            observation_, reward, done, info = env.step(action)
            observation = observation_
            env.render()
