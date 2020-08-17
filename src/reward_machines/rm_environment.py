"""
These are simple wrappers that will include RMs to any given environment.
It also keeps track of the RM state as the agent interacts with the envirionment.

However, each environment must implement the following function:
    - *get_events(...)*: Returns the propositions that currently hold on the environment.

Notes:
    - The episode ends if the RM reaches a terminal state or the environment reaches a terminal state.
    - The agent only gets the reward given by the RM.
    - Rewards coming from the environment are ignored.
"""

import gym
from gym import spaces
import numpy as np
from reward_machines.reward_machine import RewardMachine


class RewardMachineEnv(gym.Wrapper):
    def __init__(self, env, rm_files):
        """
        RM environment
        --------------------
        It adds a set of RMs to the environment:
            - Every episode, the agent has to solve a different RM task
            - This code keeps track of the current state on the current RM task
            - The id of the RM state is appended to the observations
            - The reward given to the agent comes from the RM

        Parameters
        --------------------
            - env: original environment. It must implement the following function:
                - get_events(...): Returns the propositions that currently hold on the environment.
            - rm_files: list of strings with paths to the RM files.
        """
        super().__init__(env)

        # Loading the reward machines
        self.rm_files = rm_files
        self.reward_machines = []
        self.num_rm_states = 0
        for rm_file in rm_files:
            rm = RewardMachine(rm_file)
            self.num_rm_states += len(rm.get_states())
            self.reward_machines.append(rm)
        self.num_rms = len(self.reward_machines)

        # The observation space is a dictionary including the env features and a one-hot representation of the state in the reward machine
        self.observation_dict  = spaces.Dict({'features': env.observation_space, 'rm-state': spaces.Box(low=0, high=1, shape=(self.num_rm_states,), dtype=np.uint8)})
        flatdim = gym.spaces.flatdim(self.observation_dict)
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(flatdim,), dtype=np.float32)

        # Computing one-hot encodings for the non-terminal RM states
        self.rm_state_features = {}
        for rm_id, rm in enumerate(self.reward_machines):
            for u_id in rm.get_states():
                u_features = np.zeros(self.num_rm_states)
                u_features[len(self.rm_state_features)] = 1
                self.rm_state_features[(rm_id,u_id)] = u_features
        self.rm_done_feat = np.zeros(self.num_rm_states) # for terminal RM states, we give as features an array of zeros

        # Selecting the current RM task
        self.current_rm_id = -1
        self.current_rm    = None

    def reset(self):
        # Reseting the environment and selecting the next RM tasks
        self.obs = self.env.reset()
        self.current_rm_id = (self.current_rm_id+1)%self.num_rms
        self.current_rm    = self.reward_machines[self.current_rm_id]
        self.current_u_id  = self.current_rm.reset()

        # Adding the RM state to the observation
        return self._get_observation(self.obs, self.current_rm_id, self.current_u_id, False)

    def step(self, action):
        # executing the action in the environment
        next_obs, original_reward, env_done, info = self.env.step(action)

        # getting the output of the detectors and saving information for generating counterfactual experiences
        true_props = self.env.get_events()
        self.qrm_params = self.obs, action, next_obs, env_done, true_props, info
        self.obs = next_obs

        # update the RM state
        self.current_u_id, rm_rew, rm_done = self.current_rm.step(self.current_u_id, true_props, info)

        # returning the result of this action
        done = rm_done or env_done
        rm_obs = self._get_observation(next_obs, self.current_rm_id, self.current_u_id, done)

        return rm_obs, rm_rew, done, info

    def _get_observation(self, next_obs, rm_id, u_id, done):
        rm_feat = self.rm_done_feat if done else self.rm_state_features[(rm_id,u_id)]
        rm_obs = {'features': next_obs,'rm-state': rm_feat}
        return gym.spaces.flatten(self.observation_dict, rm_obs)           


class RewardMachineWrapper(gym.Wrapper):
    def __init__(self, env, add_qrm, add_rs, gamma, rs_gamma):
        """
        RM wrapper
        --------------------
        It adds qrm (counterfactual experience) and/or reward shaping to *info* in the step function

        Parameters
        --------------------
            - env(RewardMachineEnv): It must be an RM environment
            - add_qrm(bool):   if True, it will add a set of counterfactual experiences to info
            - add_rs(bool):    if True, it will add reward shaping to info
            - gamma(float):    Discount factor for the environment
            - rs_gamma(float): Discount factor for shaping the rewards in the RM
        """
        super().__init__(env)
        self.add_qrm = add_qrm
        self.add_rs  = add_rs
        if add_rs:
            for rm in env.reward_machines:
                rm.add_reward_shaping(gamma, rs_gamma)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        # RM and RM state before executing the action
        rm_id = self.env.current_rm_id
        rm    = self.env.current_rm
        u_id  = self.env.current_u_id

        # executing the action in the environment
        rm_obs, rm_rew, done, info = self.env.step(action)

        # adding qrm if needed
        if self.add_qrm:
            qrm_experience = self._get_qrm_experience(*self.qrm_params)
            info["qrm-experience"] = qrm_experience
        elif self.add_rs:
            rs_experience  = self._get_rm_experience(rm_id, rm, u_id, *self.qrm_params)
            info["rs-experience"] = rs_experience

        return rm_obs, rm_rew, done, info

    def _get_observation(self, next_obs, rm_id, u_id, done):
        rm_feat = self.rm_done_feat if done else self.rm_state_features[(rm_id,u_id)]
        rm_obs = {'features': next_obs,'rm-state': rm_feat}
        return gym.spaces.flatten(self.observation_dict, rm_obs)

    def _get_rm_experience(self, rm_id, rm, u_id, obs, action, next_obs, env_done, true_props, info):
        rm_obs = self._get_observation(obs, rm_id, u_id, False)
        next_u_id, rm_rew, rm_done = rm.step(u_id, true_props, info, self.add_rs, env_done)
        done = rm_done or env_done
        rm_next_obs = self._get_observation(next_obs, rm_id, next_u_id, done)
        return rm_obs,action,rm_rew,rm_next_obs,done

    def _get_qrm_experience(self, obs, action, next_obs, env_done, true_props, info):
        """
        Returns a list of counterfactual experiences generated per each RM state.
        Format: [..., (obs, action, r, new_obs, done), ...]
        """
        experiences = []
        for rm_id, rm in enumerate(self.reward_machines):
            for u_id in rm.get_states():
                exp = self._get_rm_experience(rm_id, rm, u_id, obs, action, next_obs, env_done, true_props, info)
                experiences.append(exp)
        return experiences                
