from reward_machines.reward_functions import *
from reward_machines.reward_machine_utils import evaluate_dnf, value_iteration
import time

class RewardMachine:
    def __init__(self, file):
        # <U,u0,delta_u,delta_r>
        self.U  = []         # list of non-terminal RM states
        self.u0 = None       # initial state
        self.delta_u    = {} # state-transition function
        self.delta_r    = {} # reward-transition function
        self.terminal_u = -1  # All terminal states are sent to the same terminal state with id *-1*
        self._load_reward_machine(file)
        self.known_transitions = {} # Auxiliary variable to speed up computation of the next RM state

    # Public methods -----------------------------------

    def add_reward_shaping(self, gamma, rs_gamma):
        """
        It computes the potential values for shaping the reward function:
            - gamma(float):    this is the gamma from the environment
            - rs_gamma(float): this gamma that is used in the value iteration that compute the shaping potentials
        """
        self.gamma    = gamma
        self.potentials = value_iteration(self.U, self.delta_u, self.delta_r, self.terminal_u, rs_gamma)
        for u in self.potentials:
            self.potentials[u] = -self.potentials[u]


    def reset(self):
        # Returns the initial state
        return self.u0

    def _compute_next_state(self, u1, true_props):
        """ Given the current state u1 and a proposition valuation, it returns the next state """
        for phi, next_state in self.delta_u[u1]:
            if evaluate_dnf(phi, true_props):
                # at most one formula should be satisfied by the valuation
                return next_state, phi
        return self.terminal_u, "no formula"

    def get_next_state(self, u1, true_props):
        if (u1,true_props) not in self.known_transitions:
            u2, phi = self._compute_next_state(u1, true_props)
            self.known_transitions[(u1,true_props)] = u2, phi
        return self.known_transitions[(u1,true_props)]

    def step(self, u1, true_props, s_info, add_rs=False, env_done=False):
        """
        Emulates an step on the reward machine from state *u1* when observing *true_props*.
        The rest of the parameters are for computing the reward when working with non-simple RMs: s_info (extra state information to compute the reward).
        """

        # Computing the next state in the RM and checking if the episode is done
        assert u1 != self.terminal_u, "the RM was set to a terminal state!"
        u2, phi = self.get_next_state(u1, true_props)
        done = (u2 == self.terminal_u)
        # Getting the reward
        rew = self._get_reward(u1, phi, u2, s_info, add_rs, env_done)

        return u2, rew, done


    def get_states(self):
        return self.U

    def get_useful_transitions(self, u1):
        # This is an auxiliary method used by the HRL baseline to prune "useless" options
        return [phi.split("&") for phi, u2 in self.delta_u[u1].items() if u1 != u2]


    # Private methods -----------------------------------

    def _get_reward(self, u1, phi, u2, s_info, add_rs, env_done):
        """
        Returns the reward associated to this transition.
        """
        # Getting reward from the RM
        reward = 0
        # NOTE: if the agent falls from the reward machine, phi == "no formula" and it receives reward of zero.
        if u1 in self.delta_r and phi in self.delta_r[u1]:
            assert u1 in self.delta_u and phi in self.delta_u[u1] and self.delta_u[u1][phi] == u2
            reward += self.delta_r[u1][phi].get_reward(s_info)
        # Adding the reward shaping (if needed)
        rs = 0.0
        if add_rs:
            un = self.terminal_u if env_done else u2 # If the env reached a terminal state, we have to use the potential from the terminal RM state to keep RS optimality guarantees
            rs = self.gamma * self.potentials[un] - self.potentials[u1]
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
            self.delta_u[u1][dnf_formula] = u2
            # Adding reward-transition to delta_r
            if u1 not in self.delta_r:
                self.delta_r[u1] = {}
            self.delta_r[u1][dnf_formula] = reward_function
        # Sorting self.U... just because... 
        self.U = sorted(self.U)

    def _add_state(self, u_list):
        for u in u_list:
            if u not in self.U and u != self.terminal_u:
                self.U.append(u)
