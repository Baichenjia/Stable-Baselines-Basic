import gym
from stable_baselines import A2C
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
import numpy as np
import os


# 正常的 Gym-wrapper, 没有完成任何工作
class CustomWrapper(gym.Wrapper):
    def __init__(self, env):
        # 输出参数只有一个, 是 env
        super(CustomWrapper, self).__init__(env)

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

##############################
# 1. 限制周期长度的wrapper
##############################


class TimeLimitWrapper(gym.Wrapper):
    ## 显示了周期的最大长度. 在 init 里面初始化 max_steps, 在 step 中检测如果超过 max_steps 就将 done 设为 True
    def __init__(self, env, max_steps=100):
        super(TimeLimitWrapper, self).__init__(env)
        self.max_steps = max_steps
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self.env.reset()

    def step(self, action):
        self.current_step += 1
        obs, reward, done, info = self.env.step(action)
        if self.current_step >= self.max_steps:
            done = True
            info['time_limit_reached'] = True
        return obs, reward, done, info


def test_time_limit_wrapper():
    # 100  {'time_limit_reached': True}
    env = gym.make("Pendulum-v0")
    env = TimeLimitWrapper(env, max_steps=100)
    obs = env.reset()
    done = False
    n_steps = 0
    while not done:
        random_action = env.action_space.sample()
        obs, reward, done, info = env.step(random_action)
        n_steps += 1
    print(n_steps, info)

##############################
# 2. 限制动作范围的 wrapper
##############################


class NormalizeActionWrapper(gym.Wrapper):
    # 将 action space 规约到 -1~1 之间
    # step 函数中, 将输入的 (-1,1) 的动作, 重新规约到原来的动作空间中, 再调用函数进行 step
    def __init__(self, env):
        # 保留原来的 action space 的范围
        action_space = env.action_space
        self.low, self.high = action_space.low, action_space.high
        # 重置 action space 为 [-1,1] 之间
        env.action_space = gym.spaces.Box(low=-1, high=1, shape=action_space.shape, dtype=np.float32)
        super(NormalizeActionWrapper, self).__init__(env)

    def rescale_action(self, scaled_action):
        # 将输入 x~[-1, 1] 之间的动作重新规约到 y~[self.low, high] 之间
        # y = (x-(-1))*[(high-low)/(1-(-1))]+low
        return (scaled_action + 1.0) * (self.high - self.low) * 0.5 + self.low

    def reset(self):
        return self.env.reset()

    def step(self, action):
        rescaled_action = self.rescale_action(action)
        obs, reward, done, info = self.env.step(rescaled_action)
        return obs, reward, done, info


def test_normalize_action_wrapper():
    # 在原始初始化的 env 中采样多个动作, 随后再 NormalizeActionWrapper 后的 env 中采样多个动作
    env = gym.make("Pendulum-v0")
    print("original env:", env.action_space.low, env.action_space.high)
    env.reset()
    for _ in range(5):
        print("sample action:", env.action_space.sample())

    # wrapper
    env = NormalizeActionWrapper(env)
    env.reset()
    for _ in range(5):
        print("Normalized action:", env.action_space.sample())


##############################
# 3. wrapper 与 stable baselines 中的 agent 结合进行训练
##############################


# Monitor 可以记录环境在运行过程中产生的记录 mean episode reward, mean episode length
def test_monitor():
    env = gym.make('Pendulum-v0')
    env = Monitor(gym.make('Pendulum-v0'), filename=None, allow_early_resets=True)
    normalized_env = NormalizeActionWrapper(env)
    normalized_env = DummyVecEnv([lambda: normalized_env])
    # model
    model_2 = A2C('MlpPolicy', normalized_env, verbose=1).learn(1000)


################################
# 4. VecNormalize 是 stable baselines 中提供的规约, 记录在运行过程中的 state 的 std 和 return 的 std
################################
from stable_baselines.common.vec_env import VecNormalize, VecFrameStack


def test_vec_normalize():
    env = DummyVecEnv([lambda: gym.make("Pendulum-v0")])
    normalized_vec_env = VecNormalize(env)
    obs = normalized_vec_env.reset()
    for _ in range(10):
        action = [normalized_vec_env.action_space.sample()]
        obs, reward, _, _ = normalized_vec_env.step(action)
        print(obs, reward)

################################
# 5. VecFrameStack 用于在 Atari 将相邻几帧进行叠加
################################


def test_frame_stack():
    env = DummyVecEnv([lambda: gym.make("Pendulum-v0")])
    obs = env.reset()
    print("Before FrameStack, observation.shape =", obs.shape)   # (1, 3)

    frame_stack_env = VecFrameStack(env, n_stack=4)      # 叠加连续的 4 帧组成状态
    obs = frame_stack_env.reset()
    print("After FrameStack, observation.shape =", obs.shape)   # (1, 12)


if __name__ == '__main__':
    # test_time_limit_wrapper()
    # test_normalize_action_wrapper()
    # test_monitor()
    # test_vec_normalize()
    test_frame_stack()



