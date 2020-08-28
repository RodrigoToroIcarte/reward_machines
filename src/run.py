import sys
import re
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np

from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import parse_unknown_args
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module


# Importing our environments and auxiliary functions
import envs
from envs.water.water_world import Ball, BallAgent
from reward_machines.rm_environment import RewardMachineWrapper
from cmd_util import make_vec_env, make_env, common_arg_parser

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def train(args, extra_args):
    env_type, env_id = get_env_type(args)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    # Adding RM-related parameters
    alg_kwargs['use_rs']   = args.use_rs
    alg_kwargs['use_qrm']  = args.use_qrm
    alg_kwargs['gamma']    = args.gamma

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args)

    if alg in ['deepq', 'qlearning', 'hrl', 'hdrl']:
        env = make_env(env_id, env_type, args, seed=seed, logger_dir=logger.get_dir())
    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config)

        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, args.num_env or 1, seed, args, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)

        if env_type == 'mujoco':
            env = VecNormalize(env, use_tf=True)

    return env


def get_env_type(args):
    env_id = args.env

    if args.env_type is not None:
        return args.env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    library = 'rl_agents'
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join([library, alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['baselines', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs



def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])

    model, env = train(args, extra_args)

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    if args.play:
        logger.log("Running trained model")
        obs = env.reset()

        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))

        episode_rew = np.zeros(env.num_envs) if isinstance(env, VecEnv) else np.zeros(1)
        while True:
            if state is not None:
                actions, _, state, _ = model.step(obs,S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)

            obs, rew, done, _ = env.step(actions)
            episode_rew += rew
            env.render()
            done_any = done.any() if isinstance(done, np.ndarray) else done
            if done_any:
                for i in np.nonzero(done)[0]:
                    print('episode_rew={}'.format(episode_rew[i]))
                    episode_rew[i] = 0

    env.close()

    return model

if __name__ == '__main__':
    # python run.py --alg=ddpg --use_qrm --env=Ant-RM5-v0 --num_timesteps=1e6 --network=mlp --num_layers=4 --num_hidden=128 --activation=tf.nn.relu
    # python run.py --alg=ddpg --use_qrm --env=Ant-RM5-v0 --num_timesteps=1e8 --log_path=./results/ant_rm5/qrm --save_path=./results/ant_rm5/qrm/models --network=mlp --num_layers=4 --num_hidden=128
    # python run.py --alg=ddpg --env=Ant-RM5-v0 --num_timesteps=1e8 --log_path=./results/ant_rm5/ddpg --save_path=./results/ant_rm5/ddpg/models --network=mlp --num_layers=4 --num_hidden=128

    # python run.py --alg=ddpg --use_qrm --env=Half-Cheetah-RM5-v0 --reward_scale=5 --num_timesteps=1e8 --log_path=./results/cheetah_rm5/qrm --save_path=./results/cheetah_rm5/qrm/models --network=mlp --num_layers=2 --num_hidden=256 --batch_size=100 --activation=tf.nn.relu --actor_lr=1e-3 --normalize_observations=False
    # python run.py --alg=ddpg --env=Half-Cheetah-RM5-v0 --reward_scale=5 --num_timesteps=1e8 --log_path=./results/cheetah_rm5/ddpg --save_path=./results/cheetah_rm5/ddpg/models --network=mlp --num_layers=2 --num_hidden=256 --batch_size=100 --activation=tf.nn.relu --actor_lr=1e-3 --normalize_observations=False

    # python run.py --alg=ddpg --env=Half-Cheetah-RM5-v0 --reward_scale=5 --num_timesteps=3e6 --log_path=./results/cheetah_test/ddpg --save_path=./results/cheetah_test/ddpg/models --network=mlp --num_layers=2 --num_hidden=256 --batch_size=100 --activation=tf.nn.relu --actor_lr=1e-3 --normalize_observations=False
    # python run.py --alg=ddpg --use_qrm --env=Half-Cheetah-RM5-v0 --reward_scale=5 --num_timesteps=3e6 --log_path=./results/cheetah_test/qrm2 --save_path=./results/cheetah_test/qrm2/models --network=mlp --num_layers=2 --num_hidden=256 --batch_size=100 --activation=tf.nn.relu --normalize_observations=False

    # python run.py --alg=ddpg --use_qrm --env=Half-Cheetah-RM6-v0 --num_timesteps=3e6 --log_path=./results/cheetah_rm6/qrm --save_path=./results/cheetah_rm6/qrm/models --network=mlp --num_layers=2 --num_hidden=256 --batch_size=100 --activation=tf.nn.relu --normalize_observations=False
    # python run.py --alg=hdrl --env=Half-Cheetah-RM7-v0 --num_timesteps=3e6 --r_max=100 --log_path=./results/cheetah_rm7/hdrl --save_path=./results/cheetah_rm7/hdrl/models

    # python run.py --alg=deepq --env=Water-M0-v0 --num_timesteps=1e8 --network=mlp --num_layers=6 --num_hidden=64 --gamma=0.9
    # python run.py --alg=deepq --use_qrm --env=Water-M0-v0 --num_timesteps=1e8 --log_path=./results/water/0/qrm --network=mlp --num_layers=6 --num_hidden=64 --gamma=0.9 --lr=1e-5



    # python3 run.py --alg=qlearning --env=Craft-M0-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=./results/craft/0/qlearning
    # python3 run.py --alg=qlearning --env=Craft-M0-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=./results/craft/0/qlearning-rs --use_rs
    # python3 run.py --alg=qlearning --env=Craft-M0-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=./results/craft/0/qrm --use_qrm
    # python3 run.py --alg=qlearning --env=Craft-M0-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=./results/craft/0/qrm-rs --use_qrm --use_rs
    # python3 run.py --alg=hrl --env=Craft-M0-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=./results/craft/0/hrl

    # python3 run.py --alg=hdrl --env=Water-M0-v0 --num_timesteps=2e6 --log_path=./results/water/0/hdrl

    # python3.6 run.py --alg=qlearning --env=Office-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=./results/ql/office/0
    # python3.6 run.py --alg=qlearning --env=Office-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=./results/rs/office/0 --use_rs
    # python3.6 run.py --alg=qlearning --env=Office-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=./results/qrm/office/0 --use_qrm
    # python3.6 run.py --alg=qlearning --env=Office-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=./results/qrm-rs/office/0 --use_qrm --use_rs

    # python3.6 run.py --alg=hrl --env=Office-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=./results/hrl/office/0

    # python3.6 run.py --alg=deepq --env=Water-M0-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=./results/ql/water/M0/0 --save_path=./results/ql/water/M0/0/models
    # python3.6 run.py --alg=deepq --env=Water-M0-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=./results/qrm/water/M0/0 --save_path=./results/qrm/water/M0/0/models --use_qrm

    # python3.6 run.py --alg=deepq --env=Water-M0-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=./results/qrm/water/M0/0 --save_path=./results/qrm/water/M0/0/models --use_qrm


    # python3.6 run.py --alg=deepq --env=Water-M0-v0 --num_timesteps=1e5 --gamma=0.9 --use_qrm

    #import time
    #t_init = time.time()
    main(sys.argv)
    #print(time.time() - t_init)