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

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput

from baselines.common.tf_util import get_session
from baselines.deepq.models import build_q_func
from baselines.deepq.deepq import ActWrapper, load_act


class OptionDQN:
    """
    Wrapper for a DQN agent that learns the policies for all the options
    """

    def __init__(self,
          env,
          gamma,
          total_timesteps,
          network='mlp',
          lr=5e-4,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          learning_starts=1000,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          **network_kwargs
            ):
        """DQN wrapper to train option policies

        Parameters
        -------
        env: gym.Env
            environment to train on
        gamma: float
            discount factor
        network: string or a function
            neural network to use as a q function approximator. If string, has to be one of the names of registered models in baselines.common.models
            (mlp, cnn, conv_only). If a function, should take an observation tensor and return a latent variable tensor, which
            will be mapped to the Q function heads (see build_q_func in baselines.deepq.models for details on that)
        total_timesteps: int
            number of env steps to optimizer for
        lr: float
            learning rate for adam optimizer
        buffer_size: int
            size of the replay buffer
        exploration_fraction: float
            fraction of entire training period over which the exploration rate is annealed
        exploration_final_eps: float
            final value of random action probability
        train_freq: int
            update the model every `train_freq` steps.
        batch_size: int
            size of a batch sampled from replay buffer for training
        learning_starts: int
            how many steps of the model to collect transitions for before learning starts
        target_network_update_freq: int
            update the target network every `target_network_update_freq` steps.
        prioritized_replay: True
            if True prioritized replay buffer will be used.
        prioritized_replay_alpha: float
            alpha parameter for prioritized replay buffer
        prioritized_replay_beta0: float
            initial value of beta for prioritized replay buffer
        prioritized_replay_beta_iters: int
            number of iterations over which beta will be annealed from initial value
            to 1.0. If set to None equals to total_timesteps.
        prioritized_replay_eps: float
            epsilon to add to the TD errors when updating priorities.
        param_noise: bool
            whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
        **network_kwargs
            additional keyword arguments to pass to the network builder.
        """

        q_func = build_q_func(network, **network_kwargs)

        # capture the shape outside the closure so that the env object is not serialized
        # by cloudpickle when serializing make_obs_ph

        observation_space = env.option_observation_space
        def make_obs_ph(name):
            return ObservationInput(observation_space, name=name)
        self.num_actions = env.option_action_space.n

        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=make_obs_ph,
            q_func=q_func,
            num_actions=self.num_actions,
            optimizer=tf.train.AdamOptimizer(learning_rate=lr),
            gamma=gamma,
            grad_norm_clipping=10,
            param_noise=param_noise, 
            scope="options"
        )

        act_params = {
            'make_obs_ph': make_obs_ph,
            'q_func': q_func,
            'num_actions': self.num_actions,
        }

        act = ActWrapper(act, act_params)

        # Create the replay buffer
        if prioritized_replay:
            replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
            if prioritized_replay_beta_iters is None:
                prioritized_replay_beta_iters = total_timesteps
            beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                           initial_p=prioritized_replay_beta0,
                                           final_p=1.0)
        else:
            replay_buffer = ReplayBuffer(buffer_size)
            beta_schedule = None
        # Create the schedule for exploration starting from 1.
        exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                     initial_p=1.0,
                                     final_p=exploration_final_eps)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        # Variables that are used during learning
        self.act   = act
        self.train = train
        self.update_target = update_target
        self.replay_buffer = replay_buffer
        self.beta_schedule = beta_schedule
        self.exploration   = exploration
        self.param_noise   = param_noise
        self.train_freq    = train_freq
        self.batch_size    = batch_size
        self.learning_starts = learning_starts
        self.target_network_update_freq = target_network_update_freq

        self.prioritized_replay       = prioritized_replay
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.prioritized_replay_beta_iters = prioritized_replay_beta_iters
        self.prioritized_replay_eps   = prioritized_replay_eps


    def get_action(self, obs, t, reset):
        kwargs = {}
        if not self.param_noise:
            update_eps = self.exploration.value(t)
            update_param_noise_threshold = 0.
        else:
            update_eps = 0.
            # Compute the threshold such that the KL divergence between perturbed and non-perturbed
            # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
            # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
            # for detailed explanation.
            update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(self.num_actions))
            kwargs['reset'] = reset
            kwargs['update_param_noise_threshold'] = update_param_noise_threshold
            kwargs['update_param_noise_scale'] = True
        action = self.act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
        return action


    def add_experience(self, obs, action, rew, new_obs, done):
        self.replay_buffer.add(obs, action, rew, new_obs, float(done))

    def learn(self, t):
        if t > self.learning_starts and t % self.train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if self.prioritized_replay:
                experience = self.replay_buffer.sample(self.batch_size, beta=self.beta_schedule.value(t))
                (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
            else:
                obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size)
                weights, batch_idxes = np.ones_like(rewards), None
            td_errors = self.train(obses_t, actions, rewards, obses_tp1, dones, weights)
            if self.prioritized_replay:
                new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                self.replay_buffer.update_priorities(batch_idxes, new_priorities)

    def update_target_network(self, t):
        if t > self.learning_starts and t % self.target_network_update_freq == 0:
            # Update target network periodically.
            self.update_target()
