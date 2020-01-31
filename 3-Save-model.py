import gym
from stable_baselines import A2C, SAC, PPO2, TD3
from stable_baselines.common.vec_env import DummyVecEnv
import numpy as np
import os

save_dir = "save_model"
os.makedirs(save_dir, exist_ok=True)
env = gym.make("Pendulum-v0")
obs = env.observation_space.sample()


def save_ppo():
    model = PPO2("MlpPolicy", "Pendulum-v0", verbose=0).learn(8000)
    model.save(save_dir + "/PPO2_tutorial")
    print('pre saved', model.predict(obs, deterministic=True))


def load_ppo():
    loaded_model = PPO2.load(save_dir + "/PPO2_tutorial")
    print("loaded", loaded_model.predict(obs, deterministic=True))


def save_a2c():
    model = A2C('MlpPolicy', 'Pendulum-v0', verbose=0, gamma=0.9, n_steps=20).learn(8000)
    model.save(save_dir + "/A2C_tutorial")
    print("pre saved", model.predict(obs, deterministic=True))


def load_a2c():
    loaded_model = A2C.load(save_dir + "/A2C_tutorial")
    print("loaded", loaded_model.predict(obs, deterministic=True))
    print("load gamma=", loaded_model.gamma, ", n_steps=", loaded_model.n_steps)
    # 模型保存模型的超参数和网络参数, 但不保存环境 env. 在 load 模型后需要重新设置环境.
    loaded_model.set_env(DummyVecEnv([lambda: gym.make("Pendulum-v0")]))
    loaded_model.learn(8000)


if __name__ == '__main__':
    # save_ppo()
    # load_ppo()
    save_a2c()
    load_a2c()









