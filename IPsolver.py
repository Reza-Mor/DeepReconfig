import gurobipy as gp
from gurobipy import GRB
import itertools
import numpy as np


def get_data_adwords(v_size, adjacency_matrix):
    """
    pre-process the data for groubi for the adwords problem
    Reads data from the specfied file and writes the graph tensor into multu dict of the following form:
        combinations, ms= gp.multidict({
            ('u1','v1'):10,
            ('u1','v2'):13,
            ('u2','v1'):9,
            ('u2','v2'):3
        })
    """

    adj_dic = {}

    for v, u in itertools.product(range(v_size), range(v_size)):
        adj_dic[(v, u)] = adjacency_matrix[u, v]

    return gp.multidict(adj_dic)


def solve_adwords(v_size, adjacency_matrix, budgets):
    try:
        m = gp.Model("adwords")
        # m.Params.LogToConsole = 0
        m.Params.timeLimit = 30

        dic = get_data_adwords(u_size, v_size, adjacency_matrix)

        # add variable
        T = v_size
        vars = []
        for t in range(T):
             name = 'variables at time {}'.format(t)
             vars.append(m.addVars(2 * v_size, vtype="B", name=name))

        # set constraints
        m.addConstrs((x.sum(v, "*") <= 1 for v in range(v_size)), "V")
        m.addConstrs((x.prod(dic, "*", u) <= budgets[u] for u in range(u_size)), "U")

        m.addConstrs((x.prod(dic, "*", v) <= 1 for v in range(v_size)), "V")

        # set the objective
        m.setObjective(k, GRB.MINIMIZE)
        m.optimize()

        solution = np.zeros(v_size).tolist()
        for v in range(v_size):
            u = 0
            for nghbr_v in x.select(v, "*"):
                if nghbr_v.getAttr("x") == 1:
                    solution[v] = u + 1
                    break
                u += 1
        return m.objVal, solution

    except gp.GurobiError as e:
        print("Error code " + str(e.errno) + ": " + str(e))
    except AttributeError:
        print("Encountered an attribute error")


if __name__ == "__main__":

    # osbm exmaple:

    # {v_id : freq}
    # r_v = {0: [1, 2], 1: [0]}

    # # 3 genres (each column), 3 movies (each row)
    # movie_features = [
    #    [0.0, 0.0, 1.0],
    #    [0.0, 0.0, 1.0],
    #    [0.0, 1.0, 1.0]
    # ]

    # # user preferences  V by |genres|
    # preferences = np.array([
    #    [0.999, 0.4, 0.222],
    #    [1, 1, 1]
    # ])

    # adjacency_matrix = np.array([
    #   [1, 2, 0],
    #   [0, 1, 0],
    #   [4, 0, 0]
    # ])

    # print(solve_submodular_matching(3, 2, adjacency_matrix, r_v, movie_features, preferences, 3))

    # adwords exmaple:

    # V by U matrix
    adjacency_matrix = np.array([[1, 2, 0], [0, 1, 0], [4, 0, 0]])

    budgets = [3, 1, 4]
    print(solve_adwords(3, 3, adjacency_matrix, budgets))
Footer
© 2022 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
CORL/IPsolver.py at master · lyeskhalil/CORL