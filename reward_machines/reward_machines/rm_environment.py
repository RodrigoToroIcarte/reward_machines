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
        s_low  = float(env.observation_space.low[0])
        s_high = float(env.observation_space.high[0])
        self.observation_space = spaces.Box(low=s_low, high=s_high, shape=(flatdim,), dtype=np.float32)

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
        return self.get_observation(self.obs, self.current_rm_id, self.current_u_id, False)

    def step(self, action):
        # executing the action in the environment
        next_obs, original_reward, env_done, info = self.env.step(action)

        # getting the output of the detectors and saving information for generating counterfactual experiences
        true_props = self.env.get_events()
        self.crm_params = self.obs, action, next_obs, env_done, true_props, info
        self.obs = next_obs

        # update the RM state
        self.current_u_id, rm_rew, rm_done = self.current_rm.step(self.current_u_id, true_props, info)

        # returning the result of this action
        done = rm_done or env_done
        rm_obs = self.get_observation(next_obs, self.current_rm_id, self.current_u_id, done)

        return rm_obs, rm_rew, done, info

    def get_observation(self, next_obs, rm_id, u_id, done):
        rm_feat = self.rm_done_feat if done else self.rm_state_features[(rm_id,u_id)]
        rm_obs = {'features': next_obs,'rm-state': rm_feat}
        return gym.spaces.flatten(self.observation_dict, rm_obs)           


class RewardMachineWrapper(gym.Wrapper):
    def __init__(self, env, add_crm, add_rs, gamma, rs_gamma):
        """
        RM wrapper
        --------------------
        It adds crm (counterfactual experience) and/or reward shaping to *info* in the step function

        Parameters
        --------------------
            - env(RewardMachineEnv): It must be an RM environment
            - add_crm(bool):   if True, it will add a set of counterfactual experiences to info
            - add_rs(bool):    if True, it will add reward shaping to info
            - gamma(float):    Discount factor for the environment
            - rs_gamma(float): Discount factor for shaping the rewards in the RM
        """
        super().__init__(env)
        self.add_crm = add_crm
        self.add_rs  = add_rs
        if add_rs:
            for rm in env.reward_machines:
                rm.add_reward_shaping(gamma, rs_gamma)

    def get_num_rm_states(self):
        return self.env.num_rm_states

    def reset(self):
        self.valid_states = None # We use this set to compute RM states that are reachable by the last experience (None means that all of them are reachable!) 
        return self.env.reset()

    def step(self, action):
        # RM and RM state before executing the action
        rm_id = self.env.current_rm_id
        rm    = self.env.current_rm
        u_id  = self.env.current_u_id

        # executing the action in the environment
        rm_obs, rm_rew, done, info = self.env.step(action)

        # adding crm if needed
        if self.add_crm:
            crm_experience = self._get_crm_experience(*self.crm_params)
            info["crm-experience"] = crm_experience
        elif self.add_rs:
            # Computing reward using reward shaping
            _, _, _, rs_env_done, rs_true_props, rs_info = self.crm_params
            _, rs_rm_rew, _ = rm.step(u_id, rs_true_props, rs_info, self.add_rs, rs_env_done)
            info["rs-reward"] = rs_rm_rew

        return rm_obs, rm_rew, done, info

    def _get_rm_experience(self, rm_id, rm, u_id, obs, action, next_obs, env_done, true_props, info):
        rm_obs = self.env.get_observation(obs, rm_id, u_id, False)
        next_u_id, rm_rew, rm_done = rm.step(u_id, true_props, info, self.add_rs, env_done)
        done = rm_done or env_done
        rm_next_obs = self.env.get_observation(next_obs, rm_id, next_u_id, done)
        return (rm_obs,action,rm_rew,rm_next_obs,done), next_u_id

    def _get_crm_experience(self, obs, action, next_obs, env_done, true_props, info):
        """
        Returns a list of counterfactual experiences generated per each RM state.
        Format: [..., (obs, action, r, new_obs, done), ...]
        """
        reachable_states = set()
        experiences = []
        for rm_id, rm in enumerate(self.reward_machines):
            for u_id in rm.get_states():
                exp, next_u = self._get_rm_experience(rm_id, rm, u_id, obs, action, next_obs, env_done, true_props, info)
                reachable_states.add((rm_id,next_u))
                if self.valid_states is None or (rm_id,u_id) in self.valid_states:
                    # We only add experience that are possible (i.e., it is possible to reach state u_id given the previous experience)
                    experiences.append(exp)

        self.valid_states = reachable_states
        return experiences


class HierarchicalRMWrapper(gym.Wrapper):
    """
    HRL wrapper
    --------------------
    It extracts options (i.e., macro-actions) for each edge on the RMs. 
    Each option policy is rewarded when the current experience would have cause a transition through that edge.

    Methods
    --------------------
        - __init__(self, env, r_min, r_max, use_self_loops):
            - In addition of extracting the set of options available, it initializes the following attributes:
                - self.option_observation_space: space of options (concatenation of the env features and the one-hot encoding of the option id)
                - self.option_action_space: space of actions wrt the set of available options
            - Parameters:
                - env(RewardMachineEnv): It must be an RM environment.
                - r_min(float):          Reward given to the option policies when they failed to accomplish their goal.
                - r_max(float):          Reward given to the option policies when they accomplished their goal.
                - use_self_loops(bool):  When true, it adds option policies for each self-loop in the RM
                - add_rs(bool):    if True, it will add reward shaping to info
                - gamma(float):    Discount factor for the environment
                - rs_gamma(float): Discount factor for shaping the rewards in the RM
        - get_valid_options(self):
            - Returns the set of valid options in the current RM state.
        - get_option_observation(self, option_id):
            - Returns the concatenation of the env observation and the one-hot encoding of the option.
        - reset(self):
            - Resets the RM environment (as usual).
        - step(self,action):
            - Executes action in the RM environment as usual, but saves the relevant information to compute the experience that will update the option policies.
        - did_option_terminate(self, option_id):
            - Returns True if the last action caused *option* to terminate.
        - get_experience(self):
            - Returns the off-policy experience necessary to update all the option policies.
    """

    def __init__(self, env, r_min, r_max, use_self_loops, add_rs, gamma, rs_gamma):
        self.r_min = r_min
        self.r_max = r_max
        super().__init__(env)

        # Adding reward shaping (if needed)
        self.add_rs  = add_rs
        if add_rs:
            for rm in env.reward_machines:
                rm.add_reward_shaping(gamma, rs_gamma)

        # Extracting the set of options available (one per edge in the RM)
        if use_self_loops:
            # This version includes options for self-loops!
            self.options = [(rm_id, u1, u2) for rm_id, rm in enumerate(env.reward_machines) for u1 in rm.delta_u
                            for phi, u2 in rm.delta_u[u1].items()]
        else:
            # This version does not include options for the self-loops!
            self.options = [(rm_id, u1, u2) for rm_id, rm in enumerate(env.reward_machines) for u1 in rm.delta_u
                            for phi, u2 in rm.delta_u[u1].items() if u1 != u2]

        self.num_options = len(self.options)
        self.valid_options   = {}
        self.option_features = {}
        for option_id in range(len(self.options)):
            # Creating one-hot representation for each option
            rm_id,u1,u2 = self.options[option_id]
            opt_features = np.zeros(self.num_options)
            opt_features[option_id] = 1
            self.option_features[(rm_id,u1,u2)] = opt_features
            # Adding the set of valid options per RM state
            if (rm_id,u1) not in self.valid_options:
                self.valid_options[(rm_id,u1)] = []
            self.valid_options[(rm_id,u1)].append(option_id)

        # Defining the observation and action space for the options
        env_obs_space = env.observation_dict['features']
        self.option_observation_dict = spaces.Dict({'features': env_obs_space, 'option': spaces.Box(low=0, high=1, shape=(self.num_options,), dtype=np.uint8)})
        flatdim = gym.spaces.flatdim(self.option_observation_dict)
        s_low  = float(env_obs_space.low[0])
        s_high = float(env_obs_space.high[0])
        self.option_observation_space = spaces.Box(low=s_low, high=s_high, shape=(flatdim,), dtype=np.float32)
        self.option_action_space = env.action_space
        self.controller_observation_space = env.observation_space
        self.controller_action_space = spaces.Discrete(self.num_options)

    def get_number_of_options(self):
        return self.num_options

    def get_valid_options(self):
        return self.valid_options[(self.env.current_rm_id,self.env.current_u_id)]

    def get_option_observation(self, option_id, env_obs=None):
        if env_obs is None:
            env_obs = self.env.obs # using the current environment observation
        opt_feat = self.option_features[self.options[option_id]]
        opt_obs = {'features': env_obs,'option': opt_feat}
        return gym.spaces.flatten(self.option_observation_dict, opt_obs)    

    def reset(self):
        self.valid_states = None # We use this set to compute RM states that are reachable by the last experience (None means that all of them are reachable!) 
        return self.env.reset()

    def step(self, action):
        # RM and RM state before executing the action
        rm    = self.env.current_rm
        u_id  = self.env.current_u_id

        # executing the action in the environment
        rm_obs, rm_rew, done, info = self.env.step(action)

        # adding crm if needed
        if self.add_rs:
            # Computing reward using reward shaping
            _, _, _, rs_env_done, rs_true_props, rs_info = self.crm_params
            _, rs_rm_rew, _ = rm.step(u_id, rs_true_props, rs_info, self.add_rs, rs_env_done)
            info["rs-reward"] = rs_rm_rew

        return rm_obs, rm_rew, done, info

    def did_option_terminate(self, option_id):
        # Note: options terminate when the current experience changes the RM state
        rm_id, u1, _ = self.options[option_id]
        _, _, _, _, true_props, _ = self.crm_params
        un = self.env.reward_machines[rm_id].get_next_state(u1, true_props)
        return u1 != un

    def _get_option_experience(self, option_id, obs, action, next_obs, env_done, true_props, info):
        rm_id, u1, u2 = self.options[option_id]
        rm = self.env.reward_machines[rm_id]

        opt_obs = self.get_option_observation(option_id, obs)
        un, rm_rew, _ = rm.step(u1, true_props, info, self.add_rs, env_done)
        done = env_done or u1 != un
        opt_next_obs = self.get_option_observation(option_id, next_obs)

        # Computing the reward for the option
        opt_rew = rm_rew
        if u1 != u2 == un: 
            opt_rew += self.r_max  # Extra positive reward because the agent accomplished this option
        elif done: 
            opt_rew += self.r_min  # Extra negative reward because the agent failed to accomplish this option

        return opt_obs,action,opt_rew,opt_next_obs,done

    def get_experience(self):
        """
        Returns a list of counterfactual experiences generated for updating each option.
        Format: [..., (obs, action, r, new_obs, done), ...]
        """
        obs, action, next_obs, env_done, true_props, info = self.crm_params
        reachable_states = set()
        experiences = []
        for option_id in range(self.num_options):
            # Computing reachable states (for the next state)
            rm_id, u1, u2 = self.options[option_id]
            rm = self.env.reward_machines[rm_id]
            un, _, _ = rm.step(u1, true_props, info)
            reachable_states.add((rm_id,un))
            # Adding experience (if needed)
            if self.valid_states is None or (rm_id,u1) in self.valid_states:
                # We only add experience that are possible (i.e., it is possible to reach state u1 given the previous experience)
                exp = self._get_option_experience(option_id, obs, action, next_obs, env_done, true_props, info)
                experiences.append(exp)

        self.valid_states = reachable_states
        return experiences                
