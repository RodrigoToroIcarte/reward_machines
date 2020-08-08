"""
This code add event detectors to the Ant3 Environment
"""
import gym
import numpy as np
from gym.envs.mujoco.ant_v3 import AntEnv
from reward_machines.rm_environment import RewardMachineEnv

class WaterEnv(gym.Env):
    def __init__(self):

        self.action_space = spaces.Discrete(5) # noop, up, right, down, left
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size,grid_size,len(self.letter_types)+1), dtype=np.uint8)
        pass

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


    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        Returns:
            observation (object): the initial observation.
        """
        raise NotImplementedError

    def render(self, mode='human'):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        raise NotImplementedError















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
