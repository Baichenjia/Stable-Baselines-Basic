import os
import warnings
import tensorflow as tf
import logging

import time
import numpy as np
import matplotlib.pyplot as plt
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2

######################################################
# 在一个 Env 中 stack 多个独立的环境, 多个环境同时与环境交互.
# !!! 该代码在本机上不能运行，会报 EOF error
######################################################


# 使用 100 个周期进行评价. 返回平均奖励
def evaluate(model, env, num_episodes=100):
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
        all_episode_rewards.append(np.sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    return mean_episode_reward


def make_env(env_id, rank, seed=0):
    # env_id: 环境名称
    def _init():
        env = gym.make(env_id)
        env.seed(seed+rank)      # 每个环境使用一个不同的 seed
        return env

    set_global_seeds(seed)       # global seed
    return _init()


def train_and_test():
    # 使用不同数量的环境进行训练, 使用单个环境进行测试
    env_id = 'CartPole-v1'
    process_to_test = [1, 2, 4, 8, 16]    # 分别使用不同数目的 process 来初始化环境
    num_experiments = 3                   # 使用 3 次实验来取得平均值
    train_steps = 5000                    # 训练 5000 个时间步
    eval_eps = 20                         # 测试 20 个周期
    algo = PPO2

    # 使用单个环境来测试
    eval_env = DummyVecEnv([lambda: gym.make(env_id)])

    # 训练
    reward_averages = []
    reward_std = []
    training_times = []
    total_procs = 0
    for n_procs in process_to_test:
        total_procs += n_procs
        print("Running for n_procs ", n_procs)
        # 如果只有一个环境进程，则使用 DummyVecEnv
        if n_procs == 1:
            train_env = DummyVecEnv([lambda: gym.make(env_id)])
        else:  # 如果有多个进程, 使用 SubprocVecEnv
            train_env = SubprocVecEnv([make_env(env_id, i+total_procs) for i in range(n_procs)],
                                         start_method='spawn')

        rewards = []
        times = []

        for _ in range(num_experiments):       # 多次重复实验目的是为了平均
            # 初始化环境, 训练, 记录训练时间
            train_env.reset()
            model = algo("MlpPolicy", train_env, verbose=0)
            start = time.time()
            model.learn(total_timesteps=train_steps)
            times.append(time.time() - start)
            # 测试
            mean_reward = evaluate(model, eval_env, num_episodes=eval_eps)
            rewards.append(mean_reward)

        train_env.close()
        reward_averages.append(np.mean(rewards))
        reward_std.append(np.std(rewards))
        training_times.append(np.mean(times))
        print("Num Proces:", n_procs, ", Reward:", reward_averages, ", Reward std:",
                                            reward_std, ", times:", training_times)


if __name__ == '__main__':
    train_and_test()





















