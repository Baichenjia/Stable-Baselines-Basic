from stable_baselines import DQN
import gym
import numpy as np


def evaluate(m, num_episodes=100):
    env_tmp = m.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env_tmp.reset()
        while not done:
            action, _ = m.predict(obs)
            obs, reward, done, info = env_tmp.step(action)
            episode_rewards.append(reward)
        all_episode_rewards.append(np.sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)
    return mean_episode_reward


# TEST 1
kwargs = {"double_q": False, "prioritized_replay": False, "policy_kwargs": dict(dueling=False)}
dqn_model = DQN('MlpPolicy', 'CartPole-v1', verbose=1, **kwargs)
# before training
mean_reward_before_training = evaluate(dqn_model, num_episodes=100)
# after training
dqn_model.learn(total_timesteps=10000, log_interval=10)
mean_reward = evaluate(dqn_model, num_episodes=100)
# Result: Mean reward: 228.27 Num episodes: 100


# TEST 2
# kwargs = {"double_q": False, "prioritized_replay": True, "policy_kwargs": dict(dueling=False)}
# dqn_model = DQN('MlpPolicy', 'CartPole-v1', verbose=1, **kwargs)
# # before training
# mean_reward_before_training_prioritized = evaluate(dqn_model, num_episodes=100)
# # after training
# dqn_model.learn(total_timesteps=10000, log_interval=10)
# mean_reward_prioritized = evaluate(dqn_model, num_episodes=100)
# Mean reward: 165.65 Num episodes: 100


# Test 3
# kwargs = {"double_q": False, "prioritized_replay": True, "policy_kwargs": dict(dueling=True)}
# dqn_model = DQN('MlpPolicy', 'CartPole-v1', verbose=1, **kwargs)
# # before training
# mean_reward_before_training_prioritized = evaluate(dqn_model, num_episodes=100)
# # after training
# dqn_model.learn(total_timesteps=10000, log_interval=10)
# mean_reward_prioritized = evaluate(dqn_model, num_episodes=100)
# Mean reward: 79.19 Num episodes: 100
