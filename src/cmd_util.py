"""
Helpers for scripts like run_atari.py.
"""

import os
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import gym
from gym.wrappers import FlattenObservation, FilterObservation
from baselines import logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common import retro_wrappers
from baselines.common.wrappers import ClipActionsWrapper
from baselines.common.cmd_util import arg_parser

from reward_machines.rm_environment import RewardMachineWrapper, HierarchicalRMWrapper

def make_vec_env(env_id, env_type, num_env, seed, args, 
                 wrapper_kwargs=None,
                 env_kwargs=None,
                 start_index=0,
                 reward_scale=1.0,
                 flatten_dict_observations=True,
                 gamestate=None,
                 initializer=None,
                 force_dummy=False):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    wrapper_kwargs = wrapper_kwargs or {}
    env_kwargs = env_kwargs or {}
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = seed + 10000 * mpi_rank if seed is not None else None
    logger_dir = logger.get_dir()
    def make_thunk(rank, initializer=None):
        return lambda: make_env(
            env_id=env_id,
            env_type=env_type,
            args=args,
            mpi_rank=mpi_rank,
            subrank=rank,
            seed=seed,
            reward_scale=reward_scale,
            gamestate=gamestate,
            flatten_dict_observations=flatten_dict_observations,
            wrapper_kwargs=wrapper_kwargs,
            env_kwargs=env_kwargs,
            logger_dir=logger_dir,
            initializer=initializer
        )

    set_global_seeds(seed)
    if not force_dummy and num_env > 1:
        return SubprocVecEnv([make_thunk(i + start_index, initializer=initializer) for i in range(num_env)])
    else:
        return DummyVecEnv([make_thunk(i + start_index, initializer=None) for i in range(num_env)])

def make_env(env_id, env_type, args, mpi_rank=0, subrank=0, seed=None, reward_scale=1.0, gamestate=None, flatten_dict_observations=True, wrapper_kwargs=None, env_kwargs=None, logger_dir=None, initializer=None):
    if initializer is not None:
        initializer(mpi_rank=mpi_rank, subrank=subrank)

    wrapper_kwargs = wrapper_kwargs or {}
    env_kwargs = env_kwargs or {}
    if ':' in env_id:
        import re
        import importlib
        module_name = re.sub(':.*','',env_id)
        env_id = re.sub('.*:', '', env_id)
        importlib.import_module(module_name)
    env = gym.make(env_id, **env_kwargs)

    # Adding RM wrappers if needed
    if args.alg.endswith("hrm") or args.alg.endswith("dhrm"):
        env = HierarchicalRMWrapper(env, args.r_min, args.r_max, args.use_self_loops)

    if args.use_rs or args.use_crm:
        env = RewardMachineWrapper(env, args.use_crm, args.use_rs, args.gamma, args.rs_gamma)

    if flatten_dict_observations and isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)

    env.seed(seed + subrank if seed is not None else None)
    env = Monitor(env,
                  logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank)),
                  allow_early_resets=True)

    if isinstance(env.action_space, gym.spaces.Box):
        env = ClipActionsWrapper(env)

    if reward_scale != 1:
        env = retro_wrappers.RewardScaler(env, reward_scale)

    return env

def common_arg_parser():
    """
    Create an argparse.ArgumentParser.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--env_type', help='type of environment, used when the environment type cannot be automatically determined', type=str)
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--alg', help='Algorithm', type=str, default='ppo2')
    parser.add_argument('--num_timesteps', type=float, default=1e6),
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default=None)
    parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco', default=None, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
    parser.add_argument('--save_video_length', help='Length of recorded video. Default: 200', default=200, type=int)
    parser.add_argument('--log_path', help='Directory to save learning curve data.', default=None, type=str)
    parser.add_argument('--play', default=False, action='store_true')
    # RM-related arguments
    parser.add_argument("--use_rs", help="Use reward shaping", action="store_true", default=False)
    parser.add_argument("--use_crm", help="Use counterfactual experience", action="store_true", default=False)
    parser.add_argument('--gamma', help="Discount factor", type=float, default=0.9)
    parser.add_argument('--rs_gamma', help="Discount factor used for reward shaping", type=float, default=0.9)
    parser.add_argument('--r_min', help="R-min reward used for training option policies in hrm", type=float, default=0.0)
    parser.add_argument('--r_max', help="R-max reward used for training option policies in hrm", type=float, default=1.0)
    parser.add_argument("--use_self_loops", help="Add option policies for self-loops in the RMs", action="store_true", default=False)
    return parser
