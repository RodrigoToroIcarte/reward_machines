import os
import tempfile

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np

import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

from baselines.deepq.utils import ObservationInput

from baselines.common.tf_util import get_session
from baselines.deepq.models import build_q_func
from baselines.deepq.deepq import ActWrapper, load_act

from rl_agents.dhrm.build_graph import build_train
from rl_agents.dhrm.replay_buffer import ReplayBuffer

class ControllerDQN:
    """
    Wrapper for a DQN agent that learns a meta-controller for HRM.
    The main difference with a standard DQN is that the set of available actions depend on the current RM state.
    """

    def __init__(self,
          env,
          network='mlp',
          lr=5e-4,
          buffer_size=50000,
          exploration_epsilon=0.1,
          train_freq=1,
          batch_size=32,
          learning_starts=1000,
          target_network_update_freq=500,
          **network_kwargs
            ):
        """DQN wrapper to train option policies

        Parameters
        -------
        env: gym.Env
            environment to train on
        network: string or a function
            neural network to use as a q function approximator. If string, has to be one of the names of registered models in baselines.common.models
            (mlp, cnn, conv_only). If a function, should take an observation tensor and return a latent variable tensor, which
            will be mapped to the Q function heads (see build_q_func in baselines.deepq.models for details on that)
        lr: float
            learning rate for adam optimizer
        buffer_size: int
            size of the replay buffer
        exploration_epsilon: float
            value of random action probability
        train_freq: int
            update the model every `train_freq` steps.
        batch_size: int
            size of a batch sampled from replay buffer for training
        learning_starts: int
            how many steps of the model to collect transitions for before learning starts
        target_network_update_freq: int
            update the target network every `target_network_update_freq` steps.
        network_kwargs
            additional keyword arguments to pass to the network builder.
        """

        # Creating the network
        q_func = build_q_func(network, **network_kwargs)

        # capture the shape outside the closure so that the env object is not serialized
        # by cloudpickle when serializing make_obs_ph

        observation_space = env.controller_observation_space
        def make_obs_ph(name):
            return ObservationInput(observation_space, name=name)

        act, train, update_target, debug = build_train(
            make_obs_ph=make_obs_ph,
            q_func=q_func,
            num_actions=env.controller_action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=lr),
            grad_norm_clipping=10,
            scope="controller"
        )

        act_params = {
            'make_obs_ph': make_obs_ph,
            'q_func': q_func,
            'num_actions': env.controller_action_space.n,
        }

        act = ActWrapper(act, act_params)

        # Create the replay buffer
        replay_buffer = ReplayBuffer(buffer_size)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        # Variables that are used during learning
        self.act   = act
        self.train = train
        self.update_target = update_target
        self.replay_buffer = replay_buffer
        self.exp_epsilon   = exploration_epsilon
        self.train_freq    = train_freq
        self.batch_size    = batch_size
        self.learning_starts = learning_starts
        self.target_network_update_freq = target_network_update_freq
        self.num_actions   = env.controller_action_space.n
        self.t = 0

    def get_action(self, obs, valid_actions):
        kwargs = {}
        action = self.act(np.array(obs)[None], update_eps=self.exp_epsilon, action_mask=self._get_mask(valid_actions), **kwargs)[0]
        return action

    def add_experience(self, obs, action, rew, new_obs, done, valid_actions, gamma):
        self.replay_buffer.add(obs, action, rew, new_obs, float(done), self._get_mask(valid_actions), gamma)

    def learn(self):
        if self.t > self.learning_starts and self.t % self.train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            obses_t, actions, rewards, obses_tp1, dones, masks, gammas = self.replay_buffer.sample(self.batch_size)
            self.train(obses_t, actions, rewards, obses_tp1, dones, masks, gammas, np.ones_like(rewards))

    def update_target_network(self):
        if self.t > self.learning_starts and self.t % self.target_network_update_freq == 0:
            # Update target network periodically.
            self.update_target()

    def increase_step(self):
        self.t += 1

    def _get_mask(self, valid_actions):
        mask = np.zeros(self.num_actions)
        mask[valid_actions] = 1
        return mask 

