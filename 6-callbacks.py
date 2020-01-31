import gym
from stable_baselines import A2C, SAC, PPO2, TD3

import numpy as np


def evaluate(model, env, num_episodes=100):
    # This function will only work for a single Environment
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    return mean_episode_reward


# default parameter
eval_env = gym.make("Pendulum-v0")
default_model = SAC("MlpPolicy", "Pendulum-v0", verbose=1).learn(8000)
mean_episode_reward = evaluate(default_model, eval_env, num_episodes=100)
print("SAC with default parameters, mean reward :", mean_episode_reward)
# SAC with default parameters, mean reward : -1207.2242618302773

# Tuned parameter
tuned_model = SAC('MlpPolicy', 'Pendulum-v0', batch_size=256, verbose=1,
                  policy_kwargs=dict(layers=[256, 256])).learn(8000)
mean_episode_reward = evaluate(tuned_model, eval_env, num_episodes=100)
print("SAC with tuned parameters, mean reward :", mean_episode_reward)
# SAC with tuned parameters, mean reward : -165.33821141442877






