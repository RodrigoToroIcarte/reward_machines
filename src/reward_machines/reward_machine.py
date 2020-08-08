from reward_machines.reward_functions import *
from reward_machines.reward_machine_utils import evaluate_dnf, value_iteration
import time

class RewardMachine:
    def __init__(self, file, use_rs, gamma):
        # <U,u0,delta_u,delta_r>
        self.U  = []         # list of non-terminal RM states
        self.u0 = None       # initial state
        self.delta_u    = {} # state-transition function
        self.delta_r    = {} # reward-transition function
        self.terminal_u = -1  # All terminal states are sent to the same terminal state with id *-1*
        self._load_reward_machine(file)
        self.use_rs = use_rs # flag indicating whether (or not) to use reward shaping
        if self.use_rs:
            self.gamma    = gamma # this is the gamme from the environment
            self.rs_gamma = gamma # gamma that is used in the value iteration that compute the shaping potentials
            self.potentials = value_iteration(self.U, self.delta_u, self.delta_r, self.terminal_u, self.rs_gamma)
            for u in self.potentials:
                self.potentials[u] = -self.potentials[u]

    # Public methods -----------------------------------

    def reset(self):
        # Returns the initial state
        return self.u0

    def _get_next_state(self, u1, true_props):
        for u2 in self.delta_u[u1]:
            if evaluate_dnf(self.delta_u[u1][u2], true_props):
                return u2
        return self.terminal_u # no transition is defined for true_props

    def step(self, u1, true_props, s_info):
        """
        Emulates an step on the reward machine from state *u1* when observing *true_props*.
        The rest of the parameters are for computing the reward when working with non-simple RMs: s_info (extra state information to compute the reward).
        """

        # Computing the next state in the RM and checking if the episode is done
        assert u1 != self.terminal_u, "the RM was set to a terminal state!"
        u2 = self._get_next_state(u1, true_props)
        done = (u2 == self.terminal_u)
        # Getting the reward
        rew = self._get_reward(u1,u2,s_info)

        return u2, rew, done


    def get_counterfactual_experience(self, true_props, s1, a, s2):
        """
        This method generates one experience per each RM state
        """
        experiences = []
        for u1 in self.U:
            u2, rew, done = self.step(u1, true_props, s1, a, s2)
            experiences.append((u1,u2,rew,done))
        return experiences

    def get_states(self):
        return self.U

    def get_useful_transitions(self, u1):
        # This is an auxiliary method used by the HRL baseline to prune "useless" options
        return [self.delta_u[u1][u2].split("&") for u2 in self.delta_u[u1] if u1 != u2]


    # Private methods -----------------------------------

    def _get_reward(self,u1,u2,s_info):
        """
        Returns the reward associated to this transition.
        """
        # Getting reward from the RM
        reward = 0 # NOTE: if the agent falls from the reward machine it receives reward of zero
        if u1 in self.delta_r and u2 in self.delta_r[u1]:
            reward += self.delta_r[u1][u2].get_reward(s_info)
        # Adding the reward shaping (if needed)
        rs = 0.0
        if self.use_rs:
            rs = self.gamma * self.potentials[u2] - self.potentials[u1]
        # Returning final reward
        return reward + rs


    def _load_reward_machine(self, file):
        """
        Example:
            0      # initial state
            [2]    # terminal state
            (0,0,'!e&!n',ConstantRewardFunction(0))
            (0,1,'e&!g&!n',ConstantRewardFunction(0))
            (0,2,'e&g&!n',ConstantRewardFunction(1))
            (1,1,'!g&!n',ConstantRewardFunction(0))
            (1,2,'g&!n',ConstantRewardFunction(1))
        """
        # Reading the file
        f = open(file)
        lines = [l.rstrip() for l in f]
        f.close()
        # setting the DFA
        self.u0 = eval(lines[0])
        terminal_states = eval(lines[1])
        # adding transitions
        for e in lines[2:]:
            # Reading the transition
            u1, u2, dnf_formula, reward_function = eval(e)
            # terminal states
            if u1 in terminal_states:
                continue
            if u2 in terminal_states:
                u2  = self.terminal_u
            # Adding machine state
            self._add_state([u1,u2])
            # Adding state-transition to delta_u
            if u1 not in self.delta_u:
                self.delta_u[u1] = {}
            self.delta_u[u1][u2] = dnf_formula
            # Adding reward-transition to delta_r
            if u1 not in self.delta_r:
                self.delta_r[u1] = {}
            self.delta_r[u1][u2] = reward_function
        # Sorting self.U... just because... 
        self.U = sorted(self.U)

    def _add_state(self, u_list):
        for u in u_list:
            if u not in self.U and u != self.terminal_u:
                self.U.append(u)
