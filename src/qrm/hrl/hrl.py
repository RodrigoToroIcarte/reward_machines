"""
Q-Learning based method
"""

import random
from baselines import logger


def get_qmax(Q,s,actions,q_init):
    if s not in Q:
        Q[s] = dict([(a,q_init) for a in actions])
    return max(Q[s].values())

def get_best_action(Q,s,actions,q_init):
    qmax = get_qmax(Q,s,actions,q_init)
    best = [a for a in actions if Q[s][a] == qmax]
    return random.choice(best)

def learn(env,
          network=None,
          seed=None,
          lr=0.1,
          total_timesteps=100000,
          epsilon=0.1,
          print_freq=10000,
          gamma=0.9,
          q_init=1.0,
          hrl_lr=0.1):
    """Train a tabular HRL method.

    Parameters
    -------
    env: gym.Env
        environment to train on
    network: string or a function
        This is just a placeholder to be consistent with the openai-baselines interface, but we don't really use state-approximation in tabular q-learning
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    lr: float
        learning rate
    total_timesteps: int
        number of env steps to optimizer for
    epsilon: float
        epsilon-greedy exploration
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    gamma: float
        discount factor
    q_init: float
        initial q-value for unseen states
    hrl_lr: float
        learning rate for the macro-controller
    """

    # Running Q-Learning
    step         = 0
    num_episodes = 0
    reward_total = 0
    actions      = list(range(env.action_space.n))
    Q_controller = {}   # Q-values for the meta-controller
    Q_options    = {}   # Q-values for the option policies
    option_s     = None # State where the option initiated
    option_id    = None # Id of the current option being executed
    option_rews  = []   # Rewards obtained by the current option
    while step < total_timesteps:
        s = tuple(env.reset())
        while True:
            # Selecting an option if needed
            if option_id is None:
                valid_options = env.get_valid_options()
                option_s    = s
                option_id   = random.choice(valid_options) if random.random() < epsilon else get_best_action(Q_controller,s,valid_options,q_init)
                option_rews = []

            # Selecting and executing an action
            a = random.choice(actions) if random.random() < epsilon else get_best_action(Q_options,tuple(env.get_option_observation(option_id)),actions,q_init)
            sn, r, done, info = env.step(a)
            sn = tuple(sn)

            # Saving the real reward that the option is getting
            option_rews.append(r)

            # Updating the option policies
            for _s,_a,_r,_sn,_done in env.get_experience():
                _s,_sn = tuple(_s), tuple(_sn)
                if _s not in Q_options: Q_options[_s] = dict([(b,q_init) for b in actions])
                if _done: _delta = _r - Q_options[_s][_a]
                else:     _delta = _r + gamma*get_qmax(Q_options,_sn,actions,q_init) - Q_options[_s][_a]
                Q_options[_s][_a] += lr*_delta

            # Updating the meta-controller if needed 
            # Note that this condition always hold if done is True
            if env.did_option_terminate(option_id):
                option_sn = sn
                option_reward = sum([_r*gamma**_i for _i,_r in enumerate(option_rews)])
                if done: _delta = option_reward - Q_controller[option_s][option_id]
                else:    _delta = option_reward + gamma**(len(option_rews)) * get_qmax(Q_controller,option_sn,env.get_valid_options(),q_init) - Q_controller[option_s][option_id]
                Q_controller[option_s][option_id] += hrl_lr*_delta
                option_id = None

            # Moving to the next state
            reward_total += r
            step += 1
            if step%print_freq == 0:
                logger.record_tabular("steps", step)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("total reward", reward_total)
                logger.dump_tabular()
                reward_total = 0
            if done:
                num_episodes += 1
                break
            s = sn
