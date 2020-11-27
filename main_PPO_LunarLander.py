def main():
    ############## Hyperparameters ##############
    env_name = "LunarLander-v2"
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = 4
    render = False
    solved_reward = 230         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 50000        # max training episodes
    max_timesteps = 500        # max timesteps in one episode
    fc1_dims = 64
    fc2_dims = 64         # number of variables in hidden layer
    update_timestep = 2000      # update policy every n timesteps
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None
    #############################################

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    ppo = PPO(state_dim, action_dim,fc1_dims,fc2_dims, lr, betas, gamma, K_epochs, eps_clip)
    print(lr,betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0

    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_timesteps):
            timestep += 1

            # Running policy_old:
            action = ppo.act(state)
            state, reward, done, _ = env.step(action)
            score += reward
            ppo.rewards.append(reward)
            ppo.is_terminals.append(done)

            # update if its time
            if timestep % update_timestep == 0:
                ppo.update()
                ppo.clear_memory()
                timestep = 0

            running_reward += reward
            if render:
                env.render()
            if done:
                break

        avg_length += t
        print('Episode {} \t Score: {}'.format(i_episode,score))

        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
            break

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

if __name__ == '__main__':
    main()
