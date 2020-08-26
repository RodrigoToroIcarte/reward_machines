import tensorflow as tf

def grid_environment():
    return dict(
        lr=0.5,
        epsilon=0.1,
        print_freq=10000,
        q_init=1.0)
