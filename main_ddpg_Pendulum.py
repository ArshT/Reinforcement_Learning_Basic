import gym
import numpy as np
from ddpg import Agent
from ddpg import ActionNormalizer

#agent.load_models()
# environment
env_id = "Pendulum-v0"
env = gym.make(env_id)
env = ActionNormalizer(env)



agent = Agent(alpha=0.0003, beta=0.0003, input_dims=[env.observation_space.shape[0]], tau=0.001, env=env,epsilon=1.0,
              batch_size=128,  layer1_size=400, layer2_size=300, n_actions=env.action_space.shape[0],device='cpu')

score_history = []
for i in range(1000):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        act = agent.choose_action(obs)
        #if (act[0] > 1 or act[0] < -1) or (act[1] > 1 or act[1] < -1):
        #    print(act)
        new_state, reward, done, info = env.step(act)
        agent.store_transition(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state
        #env.render()

    score_history.append(score)
    if agent.EPSILON > agent.EPS_END:
          agent.EPSILON *= agent.EPS_DEC
    else:
      agent.EPSILON = agent.EPS_END

    if i % 10 == 0 and i > 0:
            avg_score = np.mean(score_history[max(0, i-10):(i+1)])
            print('episode: ', i,'score: ', score,
                 ' average score %.3f' % avg_score)
    else:
            print('episode: ', i,'score: ', score,'epsilon:',agent.EPSILON)
