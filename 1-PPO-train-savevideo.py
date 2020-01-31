import gym
import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
import os
import base64
from pathlib import Path
from stable_baselines.common.vec_env import VecVideoRecorder


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


env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])
model = PPO2(MlpPolicy, env, verbose=1)
mean_reward_before_training = evaluate(model, num_episodes=100)

# learn
model.learn(total_timesteps=10000)
mean_reward = evaluate(model, num_episodes=100)

# video
# os.system("Xvfb :1 -screen 0 1024x768x24 &")
# os.environ['DISPLAY'] = ':1'


def record_video(env_id, model, video_length=500, prefix='', video_folder='videos/'):
    """
      :param env_id: (str)
      :param model: (RL model)
      :param video_length: (int)
      :param prefix: (str)
      :param video_folder: (str)
    """
    eval_env = DummyVecEnv([lambda: gym.make(env_id)])
    # Start the video at step=0 and record 500 steps
    eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
        record_video_trigger=lambda step: step == 0, video_length=video_length, name_prefix=prefix)

    obs = eval_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _ = eval_env.step(action)

    # Close the video recorder
    eval_env.close()


record_video('CartPole-v1', model, video_length=1000, prefix='ppo2-cartpole')







