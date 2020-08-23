import tensorflow as tf

def water_environment():

    controller_kargs=dict(
          network='mlp',
          num_layers=2, 
          num_hidden=64, 
          activation=tf.nn.relu,
          lr=1e-3,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          learning_starts=100,
          target_network_update_freq=100
        )
    option_kargs=dict(
          network='mlp',
          num_layers=6, 
          num_hidden=64, 
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
        gamma=0.9,
        controller_kargs=controller_kargs,
        option_kargs=option_kargs)

