########################################################
## 自己实现一个 Maze 环境, 并使用 Gym 的方式进行封装
########################################################
import numpy as np
import gym
from gym import spaces
from stable_baselines import ACKTR, PPO2
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv


class GoLeftEnv(gym.Env):
    metadata = {'render.modes': ['console']}
    LEFT = 0
    RIGHT = 1

    def __init__(self, grid_size=10):
        # 环境的思路是，agent 初始化在最右侧. 可以执行向左和向右的命令.
        super(GoLeftEnv, self).__init__()
        self.grid_size = grid_size
        self.agent_pos = grid_size - 1      # 初始化在右侧
        # action space
        n_actions = 2
        self.action_space = spaces.Discrete(n_actions)
        # observation space  代表智能体现在的坐标
        self.observation_space = spaces.Box(low=0, high=self.grid_size, shape=(1,), dtype=np.float32)

    def reset(self):
        # 初始化智能体的位置到最右侧
        self.agent_pos = self.grid_size - 1
        return np.array(self.agent_pos).astype(np.float32)

    def step(self, action):
        if action == self.LEFT:
            self.agent_pos -= 1
        elif action == self.RIGHT:
            self.agent_pos += 1
        else:
            raise ValueError("ERROR ACTION")

        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)   # 不超过maze的范围
        done = self.agent_pos == 0
        reward = 1 if self.agent_pos == 0 else 0          # 到达最左侧奖励为1
        info = {}
        return np.array(self.agent_pos).astype(np.float32), reward, done, info

    def render(self):
        pass


# test the env
env = GoLeftEnv(grid_size=10)

obs = env.reset()
env.render()

print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())

GO_LEFT = 0
# Hardcoded best agent: always go left!
n_steps = 20
for step in range(n_steps):
    print("Step {}".format(step + 1))
    obs, reward, done, info = env.step(GO_LEFT)
    print('obs=', int(obs), 'reward=', reward, 'done=', done)
    env.render()
    if done:
        print("Goal reached!", "reward=", reward)
        break

# 使用 stable baselines 中的算法, 在新建的环境中进行训练
env = GoLeftEnv(grid_size=10)
env = Monitor(env, filename=None, allow_early_resets=True)
env = DummyVecEnv([lambda: env])

#####################
# Before training
#####################


def evaluate_the_model(m):
    obs = env.reset()
    for _ in range(20):
        action, _ = m.predict(obs, deterministic=True)
        print("action:", action, end=",")
        obs, reward, done, info = env.step(action)
        print(" next_obs:", obs, ", reward:", reward, ", done:", done)
        if done:
            print("GOAL REACHED")
            break


# train the model
model = PPO2('MlpPolicy', env, verbose=1)

print("Before Training:")
evaluate_the_model(model)

# train the model
model = model.learn(5000)

print("After Training:")
evaluate_the_model(model)




