import tensorflow as tf

def atari():
    return dict(
        network='conv_only',
        lr=1e-4,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=True,
        prioritized_replay_alpha=0.6,
        checkpoint_freq=10000,
        checkpoint_path=None,
        dueling=True
    )

def retro():
    return atari()

def water_environment():
    return dict(
        network='mlp',
        num_layers=3, 
        num_hidden=256, 
        activation=tf.nn.relu,
        lr=1e-5,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=1,
        batch_size=32,
        learning_starts=1000,
        target_network_update_freq=100,
        gamma=0.9,
        prioritized_replay=False,
        param_noise=False
    )
