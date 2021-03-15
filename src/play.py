"""
This code allows to manually play our environments
"""
import envs, gym, argparse
from envs.water.water_world import Ball, BallAgent

if __name__ == '__main__':

    # Examples
    # >>> python3 play.py --env Office-v0
    # >>> python3 play.py --env Craft-M0-v0
    # >>> python3 play.py --env Water-M0-v0

    # Getting params
    parser = argparse.ArgumentParser(prog="play", description='This code allows to manually play our environments.')
    parser.add_argument('--env', default='Office-v0', type=str, 
                        help='This parameter indicated which environment to use.')

    args = parser.parse_args()
    env = gym.make(args.env)
    env.render()