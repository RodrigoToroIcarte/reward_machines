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
    def __init__(self, env, rm_files, use_reward_shaping, rs_gamma):
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
            - use_reward_shaping: when True, it will add shape the rewards in the RM as explained in the paper.
            - rs_gamma: Auxiliary value (<1) to shape the rewards if needed

        NOTES
        --------------------
            - Do not use reward shaping in the evaluation environments
        """
        super().__init__(env)

        # Loading the reward machines
        self.rm_files = rm_files
        self.reward_machines = []
        self.num_rm_states = 0
        for rm_file in rm_files:
            rm = RewardMachine(rm_file, use_reward_shaping, rs_gamma)
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

        #self.last_true_props = "" # <- HERE!!!!
        # Adding the ltl goal to the observation
        return self._get_observation(self.obs, self.current_rm_id, self.current_u_id, False)

    def step(self, action):
        # executing the action in the environment
        next_obs, original_reward, env_done, info = self.env.step(action)

        # getting the output of the detectors and saving information for generating counterfactual experiences
        true_props = self.env.get_events()
        qrm_experience = self.get_qrm_experience(self.obs, action, next_obs, env_done, true_props, info)
        self.obs = next_obs

        #c_id = self.current_u_id
        # update the RM state
        self.current_u_id, rm_rew, rm_done = self.current_rm.step(self.current_u_id, true_props, info)
        #if c_id != self.current_u_id:
        #    print(c_id, "->", self.current_u_id)

        # returning the result of this action
        done = rm_done or env_done
        rm_obs = self._get_observation(next_obs, self.current_rm_id, self.current_u_id, done)

        # adding qrm experience to the information
        info["qrm-experience"] = qrm_experience
        #self.last_true_props = true_props # <- HERE!!!!

        return rm_obs, rm_rew, done, info

    def _get_observation(self, next_obs, rm_id, u_id, done):
        rm_feat = self.rm_done_feat if done else self.rm_state_features[(rm_id,u_id)]
        rm_obs = {'features': next_obs,'rm-state': rm_feat}
        return gym.spaces.flatten(self.observation_dict, rm_obs)

    def get_qrm_experience(self, obs, action, next_obs, env_done, true_props, info):
        """
        Returns a list of counterfactual experiences generated per each RM state.
        Format: [..., (obs, action, r, new_obs, done), ...]
        """
        experiences = []
        for rm_id, rm in enumerate(self.reward_machines):
            for u_id in rm.get_states():
                #if (rm_id,u_id) != (self.current_rm_id,self.current_u_id):
                #    if ("c" in self.last_true_props and u_id == 0) or ("d" in self.last_true_props and u_id == 1): 
                #        continue # <- HERE!!!!
                rm_obs = self._get_observation(obs, rm_id, u_id, False)
                next_u_id, rm_rew, rm_done = rm.step(u_id, true_props, info)
                done = rm_done or env_done
                rm_next_obs = self._get_observation(next_obs, rm_id, next_u_id, done)
                experiences.append((rm_obs,action,rm_rew,rm_next_obs,done))
        return experiences                

