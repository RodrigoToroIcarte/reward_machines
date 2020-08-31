"""
This code allows to compute and evaluate optimal policies for the grid environments. These optimal values are used to normalize the rewards per task.
"""
import envs, gym, argparse
from envs.water.water_world import Ball, BallAgent

if __name__ == '__main__':

	# Examples
	# >>> python3 test_optimal_policies.py --env Office-v0
	# >>> python3 test_optimal_policies.py --env Craft-M0-v0

    # Getting params
    parser = argparse.ArgumentParser(prog="test_optimal_policies", description='This code allows to evaluate optimal policies for the grid environments.')
    parser.add_argument('--env', default='Office-v0', type=str, 
                        help='This parameter indicated which environment to use.')

    args = parser.parse_args()
    env = gym.make(args.env)
    arps = env.test_optimal_policies(num_episodes=1000, epsilon=0.1, gamma=0.9)
    print(args.env)
    print(arps)
    print(10000*sum(arps)/len(arps))