"""
This code add event detectors to the Ant3 Environment
"""
import gym
import numpy as np
from gym.envs.mujoco.ant_v3 import AntEnv
from reward_machines.rm_environment import RewardMachineEnv

class MyAntEnv(gym.Wrapper):
    def __init__(self):
        # Note that the current position is key for our tasks
        super().__init__(AntEnv(exclude_current_positions_from_observation=False))

    def step(self, action):
        # executing the action in the environment
        next_obs, original_reward, env_done, info = self.env.step(action)
        self.info = info
        return next_obs, original_reward, env_done, info

    def get_events(self):
        events = ''
        if self.info['x_position'] < -1:
            events+='b'
        if self.info['x_position'] > 1:
            events+='a'
        return events



class MyAntEnvRM1(RewardMachineEnv):
    def __init__(self):
        env = MyAntEnv()
        rm_files = ["./envs/ant/reward_machines/t1.txt"]
        use_reward_shaping = False
        rs_gamma = 0.9
        super().__init__(env, rm_files, use_reward_shaping, rs_gamma)

class MyAntEnvRM4(RewardMachineEnv):
    def __init__(self):
        env = MyAntEnv()
        rm_files = ["./envs/ant/reward_machines/t4.txt"]
        use_reward_shaping = False
        rs_gamma = 0.9
        super().__init__(env, rm_files, use_reward_shaping, rs_gamma)

class MyAntEnvRM5(RewardMachineEnv):
    def __init__(self):
        env = MyAntEnv()
        rm_files = ["./envs/ant/reward_machines/t5.txt"]
        use_reward_shaping = False
        rs_gamma = 0.9
        super().__init__(env, rm_files, use_reward_shaping, rs_gamma)
