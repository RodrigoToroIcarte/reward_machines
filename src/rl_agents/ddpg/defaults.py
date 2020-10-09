import tensorflow as tf

def half_cheetah_environment():
    return dict(
        network='mlp',
        num_layers=2, 
        num_hidden=256, 
        activation=tf.nn.relu,
        batch_size=100,
        gamma=0.99,
        normalize_observations=False
    )
