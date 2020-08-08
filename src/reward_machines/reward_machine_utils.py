def evaluate_dnf(formula,true_props):
    """
    Evaluates 'formula' assuming 'true_props' are the only true propositions and the rest are false. 
    e.g. evaluate_dnf("a&b|!c&d","d") returns True 
    """
    # ORs
    if "|" in formula:
        for f in formula.split("|"):
            if evaluate_dnf(f,true_props):
                return True
        return False
    # ANDs
    if "&" in formula:
        for f in formula.split("&"):
            if not evaluate_dnf(f,true_props):
                return False
        return True
    # NOT
    if formula.startswith("!"):
        return not evaluate_dnf(formula[1:],true_props)

    # Base cases
    if formula == "True":  return True
    if formula == "False": return False
    return formula in true_props

def value_iteration(U, delta_u, delta_r, terminal_u, gamma):
    """
    Standard value iteration approach. 
    We use it to compute the potential function for the automated reward shaping
    """
    V = dict([(u,0) for u in U])
    V[terminal_u] = 0
    V_error = 1
    while V_error > 0.0000001:
        V_error = 0
        for u1 in U:
            q_u2 = []
            for u2 in delta_u[u1]:
                if delta_r[u1][u2].get_type() != "constant": raise NotImplementedError("So far, we can only computer potential-based functions for simple reward machines")
                r = delta_r[u1][u2].get_reward(None, None, None)
                q_u2.append(r+gamma*V[u2])
            v_new = max(q_u2)
            V_error = max([V_error, abs(v_new-V[u1])])
            V[u1] = v_new
    return V

