"""
This code add event detectors to the Ant3 Environment
"""
import gym
import numpy as np
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from reward_machines.rm_environment import RewardMachineEnv

class MyHalfCheetahEnv(gym.Wrapper):
    def __init__(self):
        # Note that the current position is key for our tasks
        super().__init__(HalfCheetahEnv(exclude_current_positions_from_observation=False))

    def step(self, action):
        # executing the action in the environment
        next_obs, original_reward, env_done, info = self.env.step(action)
        self.info = info
        return next_obs, original_reward, env_done, info

    def get_events(self):
        events = ''
        if self.info['x_position'] < -10:
            events+='b'
        if self.info['x_position'] > 10:
            events+='a'
        if self.info['x_position'] < -2:
            events+='d'
        if self.info['x_position'] > 2:
            events+='c'
        if self.info['x_position'] > 4:
            events+='e'
        if self.info['x_position'] > 6:
            events+='f'
        if self.info['x_position'] > 8:
            events+='g'
        return events



class MyHalfCheetahEnvRM5(RewardMachineEnv):
    def __init__(self):
        env = MyHalfCheetahEnv()
        rm_files = ["./envs/mujoco_rm/reward_machines/t5.txt"]
        super().__init__(env, rm_files)


class MyHalfCheetahEnvRM6(RewardMachineEnv):
    def __init__(self):
        env = MyHalfCheetahEnv()
        rm_files = ["./envs/mujoco_rm/reward_machines/t6.txt"]
        super().__init__(env, rm_files)

class MyHalfCheetahEnvRM7(RewardMachineEnv):
    def __init__(self):
        env = MyHalfCheetahEnv()
        rm_files = ["./envs/mujoco_rm/reward_machines/t7.txt"]
        super().__init__(env, rm_files)

class MyHalfCheetahEnvRM8(RewardMachineEnv):
    def __init__(self):
        env = MyHalfCheetahEnv()
        rm_files = ["./envs/mujoco_rm/reward_machines/t8.txt"]
        super().__init__(env, rm_files)

class MyHalfCheetahEnvRM9(RewardMachineEnv):
    def __init__(self):
        env = MyHalfCheetahEnv()
        rm_files = ["./envs/mujoco_rm/reward_machines/t9.txt"]
        super().__init__(env, rm_files)

class MyHalfCheetahEnvRM10(RewardMachineEnv):
    def __init__(self):
        env = MyHalfCheetahEnv()
        rm_files = ["./envs/mujoco_rm/reward_machines/t10.txt"]
        super().__init__(env, rm_files)

class MyHalfCheetahEnvRM11(RewardMachineEnv):
    def __init__(self):
        env = MyHalfCheetahEnv()
        rm_files = ["./envs/mujoco_rm/reward_machines/t11.txt"]
        super().__init__(env, rm_files)

class MyHalfCheetahEnvRM12(RewardMachineEnv):
    def __init__(self):
        env = MyHalfCheetahEnv()
        rm_files = ["./envs/mujoco_rm/reward_machines/t12.txt"]
        super().__init__(env, rm_files)
