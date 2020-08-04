# reward_machines

This project studies Reward Machines — a type of finite state machine that supports the specification of reward functions while exposing reward function structure to the learner and supporting decomposition — and how to use them to speed up learning of optimal policies. Our approach, called Q-Learning for Reward Machines (QRM), decomposes the reward machine and uses off-policy q-learning to simultaneously learn subpolicies for the different components. A detailed description of Reward Machines and QMR can be found in the following paper ([link](http://proceedings.mlr.press/v80/icarte18a.html)):

    @inproceedings{tor-etal-icml18,
        author = {Toro Icarte, Rodrigo and Klassen, Toryn Q. and Valenzano, Richard and McIlraith, Sheila A.},
        title     = {Using Reward Machines for High-Level Task Specification and Decomposition in Reinforcement Learning},
        booktitle = {Proceedings of the 35th International Conference on Machine Learning (ICML)},
        year      = {2018},
        pages      = {2112--2121}
    }

**(New feature!)** We recently included an automated reward shaping approach to further improve the sample efficiency of QRM (called "qrm-rs"). This approach slightly changes the rewards on a given RM such that (i) optimal policies are preserved and (ii) the agent is encouraged to make progress towards solving the task. It works great on the *office* and *craft* domains but it does not help on the *water* domain. More details can be found in the following paper ([link](https://www.ijcai.org/proceedings/2019/840)):

    @inproceedings{cam-etal-ijcai19,
        author = {Camacho, Alberto and Toro Icarte, Rodrigo and Klassen, Toryn Q. and Valenzano, Richard and McIlraith, Sheila A.},
        title     = {LTL and Beyond: Formal Languages for Reward Function Specification in Reinforcement Learning},
        booktitle = {Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI)},
        year      = {2019},
        pages     = {6065--6073}
    }

This code is meant to be a clean and usable version of our approach. If you find any bugs or have questions about it, please let us know. We'll be happy to help you!

## Installation instructions

You might clone this repository by running:

    git clone https://bitbucket.org/RToroIcarte/qrm.git

QRM requires [Python3.5](https://www.python.org/) with three libraries: [numpy](http://www.numpy.org/), [tensorflow](https://www.tensorflow.org/), and (optionally) [pygame](https://www.pygame.org/news). We use pygame to visualize the Water World environment, but it is not required by QRM or any of our baselines.


## Running examples

To run QRM and our three baselines, move to the *src* folder and execute *run.py*. This code receives 4 parameters: The RL algorithm to use (which might be "dqn", "hrl", "hrl-rm", "qrm", or "qrm-rs"), the environment (which might be "office", "craft", or "water"), the map (which is an integer between 0 and 10), and the number of independent trial to run per map. For instance, the following command runs QRM one time over map 0 of the craft environment:

```
python3 run.py --algorithm="qrm" --world="craft" --map=0 --num_times=1
```

The results will be saved in the './tmp' folder. We also include three scripts that allow you to replicate the experiments from our paper. They are in the './scripts' folder. After running all the experiments, you might compute the average performance across the different maps and trials running *export_summary.py*:

```
python3 export_summary.py --world="office"
python3 export_summary.py --world="craft"
python3 export_summary.py --world="water"
```

The overall results will be saved in the './tmp/results' folder. Note that map 0 is excluded from the summary because we used it to tune the algorithm parameters.


## Acknowledgments

Our implementation of QRM is based on the DQN baseline code provided by [OpenAI](https://github.com/openai/baselines). We encourage you to check out their repository. They are doing really cool RL stuff too :)