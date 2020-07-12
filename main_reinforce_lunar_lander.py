import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import wrappers
from reinforce_keras import Agent
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)


def plotLearning(scores, filename, x=None, window=5):
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')
    plt.xlabel('Game')
    plt.plot(x, running_avg)
    plt.savefig(filename)

agent = Agent(ALPHA=0.0005, input_dims=8, GAMMA=0.99,n_actions=4, layer1_size=64, layer2_size=64)

env = gym.make('LunarLander-v2')
score_history = []


num_episodes = 1000
env = wrappers.Monitor(env, "tmp/lunar-lander",video_callable=lambda episode_id: True, force=True)

for i in range(num_episodes):
    done = False
    score = 0
    observation = env.reset()
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        agent.store_transition(observation, action, reward)
        observation = observation_
        score += reward
        score_history.append(score)

    _ = agent.learn()
    print('episode: ', i,'score: %.1f' % score,'average score %.1f' % np.mean(score_history[max(0, i-100):(i+1)]))

filename = 'lunar-lander-alpha001-128x128fc-newG.png'
plotLearning(score_history, filename=filename, window=25)
