
def value_iteration(S,A,L,T,rm,gamma):
    """
    Standard value iteration to compute optimal policies for the grid environments.
    
    PARAMS
    ----------
    S:     List of states
    A:     List of actions
    L:     Labeling function (it is a dictionary from states to events)
    T:     Transitions (it is a dictionary from SxA -> S)
    rm:    Reward machine
    gamma: Discount factor 

    RETURNS
    ----------
    Optimal deterministic policy (dictionary maping from states (SxU) to actions)
    """
    U = rm.get_states() # RM states
    V = dict([((s,u),0) for s in S for u in U])
    V_error = 1

    # Computing the optimal value function
    while V_error > 0.0000001:
        V_error = 0
        for s1 in S:
            for u1 in U:
                q_values = []
                for a in A:
                    s2 = T[(s1,a)]
                    l  = '' if s2 not in L else L[s2]
                    u2, r, done = rm.step(u1, l, None)
                    if done: q_values.append(r)
                    else:    q_values.append(r+gamma*V[(s2,u2)])
                v_new = max(q_values)
                V_error = max([V_error, abs(v_new-V[(s1,u1)])])
                V[(s1,u1)] = v_new

    # Extracting the optimal policy
    policy = {}
    for s1 in S:
        for u1 in U:
            q_values = []
            for a in A:
                s2 = T[(s1,a)]
                l  = '' if s2 not in L else L[s2]
                u2, r, done = rm.step(u1, l, None)
                if done: q_values.append(r)
                else:    q_values.append(r+gamma*V[(s2,u2)])
            a_i = max((x,i) for i,x in enumerate(q_values))[1] # argmax over the q-valies
            policy[(s1,u1)] = A[a_i]

    return policy

