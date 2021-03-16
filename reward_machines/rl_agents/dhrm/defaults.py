import tensorflow as tf

def water_environment():

    controller_kargs=dict(
          network='mlp',
          num_layers=2, 
          num_hidden=256, 
          activation=tf.nn.relu,
          lr=1e-3,
          exploration_epsilon=0.05,
          buffer_size=50000,
          train_freq=1,
          batch_size=32,
          learning_starts=100,
          target_network_update_freq=100
        )
    option_kargs=dict(
          network='mlp',
          num_layers=3, 
          num_hidden=1024, 
          activation=tf.nn.relu,
          lr=1e-5,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          learning_starts=1000,
          target_network_update_freq=100,
          prioritized_replay=False,
          param_noise=False
        )

    return dict(
        use_ddpg=False,
        gamma=0.9,
        controller_kargs=controller_kargs,
        option_kargs=option_kargs)


def half_cheetah_environment():

    controller_kargs=dict(
          network='mlp',
          num_layers=2, 
          num_hidden=64, 
          activation=tf.nn.relu,
          lr=1e-3,
          buffer_size=50000,
          exploration_epsilon=0.1,
          train_freq=1,
          batch_size=32,
          learning_starts=100,
          target_network_update_freq=100
        )
    option_kargs=dict(
          network='mlp',
          num_layers=2, 
          num_hidden=256, 
          activation=tf.nn.relu,
          nb_rollout_steps=100,
          reward_scale=1.0,
          noise_type='adaptive-param_0.2',
          normalize_returns=False,
          normalize_observations=False,
          critic_l2_reg=1e-2,
          actor_lr=1e-4,
          critic_lr=1e-3,
          popart=False,
          clip_norm=None,
          nb_train_steps=50, # per epoch cycle and MPI worker,  <- HERE!
          nb_eval_steps=100,
          buffer_size=1000000,
          batch_size=100, # per MPI worker
          tau=0.01,
          param_noise_adaption_interval=50
        )

    return dict(
        use_ddpg=True,
        gamma=0.99,
        controller_kargs=controller_kargs,
        option_kargs=option_kargs)
