import gurobipy as gp
from gurobipy import GRB
import itertools
from networkx.algorithms import bipartite
import numpy as np
import timeit
import shelve
import re
import argparse
import numpy as np
import timeit
import os
from tqdm import tqdm

import signal

def set_timeout(seconds):
    def decorator(func):
        def handler(signum, frame):
            raise TimeoutError()

        def wrapped_func(*args, **kwargs):
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            except TimeoutError:
                print("Function execution timed out.")
            finally:
                signal.alarm(0)  # Reset the alarm
            return result

        return wrapped_func

    return decorator


def parse_fie(file):

    # Open the file and read its contents
    with open(file, "r") as f:
        lines = f.readlines()

    parsed_lines = []

    for line in lines:
        if  line.strip() and not (line.startswith("p") or line.startswith("c") or line.startswith("%") or line.startswith("0")):
            parsed_lines.append(list(map(int, line.strip().split())))

    # Convert the parsed lines to a numpy array
    CNF = np.array(parsed_lines)[:, :-1]

    # Print the resulting array
    return CNF


def get_dic(CNF):
    """
    get a dictionary of form {variable: [(clause in which the variable appears in, index of the variable in the clause)]}
    """
    dic = {}
    J = int(max(abs(np.min(CNF)), np.max(CNF))) # number of variables in the CNF
    for j in range(1, J + 1):
        dic[j] = []  
    for c in range(CNF.shape[0]):
        index = 0
        for literal in CNF[c]:
            dic[abs(literal)].append((c, index))
            index += 1
    return dic


def get_biadj_matrix(CNF):
    """
    returns A_ and A+ , two |variables| x |clauses| biadjancncy matrices. The former only stores positive edges and the later stores negative edges. 
    """
    V = int(max(abs(np.min(CNF)), np.max(CNF))) # number of variables in the CNF
    C = CNF.shape[0]                            # number of clauses in the CNF
    A_plus, A_minus = np.zeros((V,C)), np.zeros((V,C))
    for c in range(CNF.shape[0]):
        for v in CNF[c]:
            if v > 0:
                A_plus[v-1,c] = 1
            elif v < 0:
                A_minus[abs(v)-1,c] = 1
    return A_plus, A_minus

def find_reconfig_path(CNF, init, target, T=10):
    # Create a new model
    m = gp.Model("mip")
    m.Params.LogToConsole = 0
    J = int(max(abs(np.min(CNF)), np.max(CNF))) # number of literals in the CNF
    X = m.addMVar((J+1, T), vtype="B", name="x")
    D1 = m.addMVar((J+1, T-1), vtype="B", name="d1")
    D2 = m.addMVar((J+1, T-1), vtype="B", name="d2")

    # every clause is satisfied at every step
    for t in range(T):
        for clause in CNF:
            c = sum(list(map(lambda variable: X[variable, t] if variable> 0 else (1-X[-1*variable, t]), clause)))
            m.addConstr(c >= 1)

    # the distance between any two adjacent configs is at most one variable
    for t in range(1,T):
        m.addConstr(D1[1:, t-1] >= X[1:, t-1] - X[1:, t])
        m.addConstr(D2[1:, t-1] >= X[1:, t] - X[1:, t-1])
        m.addConstr(D1[1:, t-1] + D2[1:, t-1] <= 1)
        m.addConstr(D1[1:, t-1] + D2[1:, t-1] <= (X[1:, t-1] - X[1:, t]) * (X[1:, t-1] - X[1:, t]))
    m.addConstr((D1[1:, ]+ D2[1:, ]).sum(axis = 0) <= 1)

    # must start and end at the the initial and target configuration
    m.addConstr(X[1:, 0] == init)
    m.addConstr(X[1:, T-1] == target)


    m.setObjective(0, sense=gp.GRB.MINIMIZE)
    m.optimize()

    if m.status == gp.GRB.OPTIMAL:
        return 

def generate_reconfig_path(CNF, T):
    # Create a new model
    m = gp.Model("mip")
    m.Params.LogToConsole = 0
    m.setParam('TimeLimit', 60*8)

    J = int(max(abs(np.min(CNF)), np.max(CNF))) # number of literals in the CNF
    X = m.addMVar((J+1, T), vtype="B", name="x")
    D1 = m.addMVar((J+1, T-1), vtype="B", name="d1")
    D2 = m.addMVar((J+1, T-1), vtype="B", name="d2")  
    W1 = m.addMVar(J+1, vtype="B", name="w1")
    W2 = m.addMVar(J+1, vtype="B", name="w1")
    d = m.addVar(0, 20, vtype="INTEGER", name="k")

    # every clause is satisfied at every step
    for t in range(T):
        for clause in CNF:
            c = sum(list(map(lambda variable: X[variable, t] if variable> 0 else (1-X[-1*variable, t]), clause)))
            m.addConstr(c >= 1)

    # the distance between any two adjacent configs is at most one variable
    for t in range(1,T):
        m.addConstr(D1[1:, t-1] >= X[1:, t-1] - X[1:, t])
        m.addConstr(D2[1:, t-1] >= X[1:, t] - X[1:, t-1])
        m.addConstr(D1[1:, t-1] + D2[1:, t-1] <= 1)
        m.addConstr(D1[1:, t-1] + D2[1:, t-1] <= (X[1:, t-1] - X[1:, t]) * (X[1:, t-1] - X[1:, t]))
    m.addConstr((D1[1:, ]+ D2[1:, ]).sum(axis = 0) <= 1)
    
    m.addConstr(W1[1:, ] >= X[1:, 0] - X[1:, T-1])
    m.addConstr(W2[1:, ] >= X[1:, T-1] - X[1:, 0])
    m.addConstr(W1[1:, ] + W2[1:, ] <= 1)
    m.addConstr(W1[1:, ] + W2[1:, ] <= (X[1:, 0] - X[1:, T-1]) * (X[1:, 0] - X[1:, T-1]))
    m.addConstr((W1[1:, ]+ W2[1:, ]).sum() >= d)

    
    #m.setObjective(0, sense=gp.GRB.MINIMIZE)
    m.setObjective(d, sense=gp.GRB.MAXIMIZE)
    m.optimize()

    if m.status == gp.GRB.OPTIMAL:
        return d.X, X.X[1:,]

    else:
        #print("No solution found.")
        return 0, None

def remove_duplicates(array, d):
    new_array = np.ndarray((array.shape[0], int(d)))
    column = 0
    for c1, c2 in zip(range(array.shape[1] - 1), range(1, array.shape[1])):
        if np.all(array[:,c1], array[:,c2]):
            column += 2
        else: 
            column += 1
        new_array[:,column] = c1
    return new_array

def remove_duplicate_consecutive_columns(A):
    _, indices = np.unique(A, axis=1, return_index=True)
    unique_columns = A[:, np.sort(indices)]
    return unique_columns

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    description=""
    )
    parser.add_argument(
        "--SAT_instance",
        type=str,
        default= "../datasets/SAT/CBS_k3_n100_m403_b10/CBS_k3_n100_m403_b10_0.cnf", #"../datasets/SAT/uf20-91",
        help="the path to the file where the SAT instances are stored - refer to https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=0,
        help="size of the dataset- only take the first data_size instances from the SAT_dataset. Take all of the dataset if not specified.",
    )
    args = parser.parse_args()

Exists = os.path.exists('../datasets/SAT/reconfig')
if not Exists:
   # Create a new directory because it does not exist
   os.makedirs('../datasets/SAT/reconfig')

db = shelve.open('../datasets/SAT/reconfig/CBS_k3_n100_m403_b10_0')

# open the SAT dataset
sat_instance = args.SAT_instance
CNF = parse_fie(sat_instance)
size = args.dataset_size
i = 0
print(CNF)
A_p, A_m = get_biadj_matrix(CNF)
print(np.where(A_p > 0))
print(A_p[39, 0])
#d = get_dic(CNF)
#print(d)
"""
print('Generating SAT Reconfiguration Dataset...')
while i < size:
    # checking if it is a file
    T = 10
    start= timeit.default_timer()
    # reconfig path is a numpy array with each i-th row and the j-th column represents
    # the value for the i-the variable truth value at time j
    path_length, reconfig_path_with_duplicates = generate_reconfig_path(CNF, T)
    stop = timeit.default_timer()
    if path_length != 0:
        reconfig_path = remove_duplicate_consecutive_columns(reconfig_path_with_duplicates)
        dict = {
                "time": stop - start,
                #"reconfig_path_with_duplicate": reconfig_path_with_duplicate,
                "path_length": path_length,
                "reconfig_path": reconfig_path,
                }
        print(dict)
        db[filename] = dict

        start= timeit.default_timer()
        find_reconfig_path(CNF, reconfig_path[:,0], reconfig_path[:,-1], T=10)
        stop = timeit.default_timer()
        print("time for finding: {}".format(stop - start))
    i += 1
"""
db.close()

