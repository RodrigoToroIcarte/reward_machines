# Reward Machines

Reinforcement learning (RL) methods usually treat reward functions as black boxes. As such, these methods must extensively interact with the environment in order to discover rewards and optimal policies. In most RL applications, however, users have to program the reward function and, hence, there is the opportunity to treat reward functions as white boxes instead — to show the reward function's code to the RL agent so it can exploit its internal structures to learn optimal policies faster. In this project, we show how to accomplish this idea in two steps. First, we propose reward machines (RMs), a type of finite state machine that supports the specification of reward functions while exposing reward function structure. We then describe different methodologies to exploit such structures, including automated reward shaping, task decomposition, and counterfactual reasoning for data augmentation. Our experiments on tabular and continuous domains show the benefits of exploiting reward structure across different tasks and RL agents. 

A complete description of reward machines and our methods to exploit their internal structure can be found in the following paper ([link](https://arxiv.org/abs/2010.03950)):

    @article{tor-etal-arxiv20,
        author  = {Toro Icarte, Rodrigo and Klassen, Toryn Q. and Valenzano, Richard and McIlraith, Sheila A.},
        title   = {Reward Machines: Exploiting Reward Function Structure in Reinforcement Learning},
        journal = {arXiv preprint arXiv:2010.03950},
        year    = {2020}
    }

Our methods build on top of our two previous work on reward machines ([icml18](http://proceedings.mlr.press/v80/icarte18a.html), [ijcai19](https://www.ijcai.org/Proceedings/2019/840)):

    @inproceedings{tor-etal-icml18,
        author = {Toro Icarte, Rodrigo and Klassen, Toryn Q. and Valenzano, Richard and McIlraith, Sheila A.},
        title     = {Using Reward Machines for High-Level Task Specification and Decomposition in Reinforcement Learning},
        booktitle = {Proceedings of the 35th International Conference on Machine Learning (ICML)},
        year      = {2018},
        pages      = {2112--2121}
    }
    @inproceedings{cam-etal-ijcai19,
        author = {Camacho, Alberto and Toro Icarte, Rodrigo and Klassen, Toryn Q. and Valenzano, Richard and McIlraith, Sheila A.},
        title     = {LTL and Beyond: Formal Languages for Reward Function Specification in Reinforcement Learning},
        booktitle = {Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI)},
        year      = {2019},
        pages     = {6065--6073}
    }

We will provide more details about how to run our code shortly.