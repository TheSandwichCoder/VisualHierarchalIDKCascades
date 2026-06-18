class SpecClassifierNode:
    def __init__(self, cost, idk_prob):
        self.c = cost
        self.d = idk_prob

class IdenClassifierNode:
    def __init__(self, cost, idk_prob, prob, classes):
        self.c = cost
        self.d = idk_prob
        self.class_prob = prob
        self.n_classes = classes

class GlobClassifierNode:
    def __init__(self, cost, idk_prob):
        self.c = cost
        self.d = idk_prob
        
# assumes classifiers are sorted
def get_specialized_cost(classifiers, c_det):
    # no more classifiers then return det
    if classifiers.empty():
        return c_det.cost, [c_det]

    best_cost, seq = get_specialized_cost(classifiers[1:], c_det)
    
    c_curr = classifiers[0]

    if c_curr.d < (best_cost - c_curr.cost) / best_cost:
        new_cost = c_curr.cost + best_cost * c_curr.d
        return new_cost, [c_curr] + c_det
    
    else:
        return best_cost, seq


def get_identifier_cost(identifier, spec_classifiers):
    total_cost = 0

    for c_i in range(identifier.n_classes):
        d_i = identifier.class_prob[c_i]

        total_cost += spec_classifiers[c_i].cost * d_i

    identifier.cost = total_cost
    return total_cost

# c_all comprises of c_iden and c_glob
def get_cascade_cost(c_all, c_det):
    # no more classifiers then return det
    if c_all.empty():
        return c_det.cost, [c_det]

    best_cost, seq = get_specialized_cost(c_all[1:], c_det)
    
    c_curr = c_all[0]

    if c_curr.d < (best_cost - c_curr.cost) / best_cost:
        new_cost = c_curr.cost + best_cost * c_curr.d
        return new_cost, [c_curr] + c_det
    
    else:
        return best_cost, seq