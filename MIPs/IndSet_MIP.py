import gurobipy as gp
from gurobipy import GRB
import itertools
from networkx.algorithms import bipartite
import numpy as np
import timeit
import shelve
import re
import argparse


def solve_independet_set_matrix(biadjacency_matrix, U, V):
    
    # Create a new model
    m = gp.Model("mip")
    m.Params.LogToConsole = 0
    T = U + V  #number of time steps
    N = U + V# number of nodes

    # Create variables
    X_v = m.addMVar(V, T, vtype="B", name="x_v")
    X_u = m.addMVar(U, T, vtype="B", name="x_u")
    delta_v = m.addMVar(V, T, vtype="I", name="delta_v")
    delta_u = m.addMVar(U, T, vtype="I", name="delta_u")

    #absolute values of deltas
    delta_v1 = m.addMVar(V, T, vtype="I", name="delta_v1")
    delta_v2 = m.addMVar(V, T, vtype="I", name="delta_v2")
    delta_u1 = m.addMVar(U, T, vtype="I", name="delta_u1")
    delta_u2 = m.addMVar(U, T, vtype="I", name="delta_u2")
    m.addConstr(delta_v1 - delta_v2 == delta_v)
    m.addConstr(delta_u1 - delta_u2 == delta_u)

    k = m.addVar(0, N, vtype="INTEGER", name="k")
    
    for t in range(T):
        for v, u in zip(np.nonzero(biadjacency_matrix)):
            m.addConstr(X_v[v, t] + X_u[u, t] <= 1, name='2')

        m.addConstr((X_u[:,t] + X_v[:,t]).sum() >= U - k, name="3")

        if t != 0:
            m.addConstr(X_u[:,t] == X_u[:,t-1] + delta_u[:,t], name="4 a")
            m.addConstr(X_v[:,t] == X_v[:,t-1] + delta_v[:,t], name="4 b")
            
    # write constraint four
    m.addConstr(delta_v[:0] == 0)
    m.addConstr(delta_u[:0] == 0)

    for t in range(T):
        # note: delta_v1[:0] - delta_v2[:0] = abs(delta_v[:0])
        m.addConstr((delta_v1[:t] - delta_v2[:t]).sum() + (delta_u1[:t] - delta_u2[:t]).sum() == 1) 
    
    #delta is one of 1,0,-1
    m.addConstr(delta_v >= -1)
    m.addConstr(delta_v <= 1)
    m.addConstr(delta_u >= -1)
    m.addConstr(delta_u <= 1)

    m.addConstr(X_u[:, 0] == 1, name='5 a')
    m.addConstr(X_u[:, T] == 0, name='5 b')
    m.addConstr(X_v[:, 0] == 0, name='5 c')
    m.addConstr(X_v[:, T] == 1, name='5 d')

    m.setObjective(k, GRB.MINIMIZE)
    m.optimize()
    
    if m.Status == 2: #an optimal solution is available
        return k.X


def solve_independet_set(biadjacency_matrix, U, V, solution_file, runtime_file):
    
    # Create a new model
    m = gp.Model("mip")
    m.Params.LogToConsole = 0
    T = U + V  #number of time steps
    N = U + V# number of nodes

    # Create variables
    x_v = m.addVars(V, T, vtype="B", name="x_v")
    x_u = m.addVars(U, T, vtype="B", name="x_u")
    k = m.addVar(0, N, vtype="INTEGER", name="k")
    
    for t in range(T):
        for v, u in zip(np.nonzero(biadjacency_matrix)):
            m.addConstr(x_v[v, t] + x_u[u, t] <= 1, name='2')

    m.addConstrs((x_u.sum("*", t) + x_v.sum("*", t) >= N/2 - k for t in range(T)), "3")

    #for t in range(T):
    #    for v in range(V):
    #        m.addConstr(x_v[v, t] <= 1, name='4 a')
    #        m.addConstr(x_v[v, t] >= -1, name='4 b')
    #    for u in range(U):
    #        m.addConstr(x_u[u, t] <= 1, name='4 c')
    #        m.addConstr(x_u[u, t] >= -1, name='4 d')

    # write constraint four

    for u in range(U):
        m.addConstr(x_u[u, 0] == 1, name='5 c')
        m.addConstr(x_u[u, T] == 0, name='5 d')
    for v in range(V):
        m.addConstr(x_v[v, 0] == 0, name='5 a')
        m.addConstr(x_v[v, T] == 1, name='5 b')

    m.setObjective(k, GRB.MINIMIZE)

    search_time = 0
    solved = 0

    start= timeit.default_timer() #time.time()
    m.optimize()
    stop = timeit.default_timer() #time.time()

    if m.Status == 2: #an optimal solution is available
        search_time = stop - start
        solved = 1
        runtime_file.write('Time: {} \n'.format(search_time))
        print("x_v: ", x_v)
        print("x_u: ", x_u)
        print("k: ", k)
        print("\n")
        #for v in x.values():
        #    if v.X == 1:
        #        s = extract_square_brackets(v.VarName)
        #        solution_file.write("({},{})".format(s[0], s[1]))
        #solution_file.write("\n")
    else:
        runtime_file.write('Encountered an error. Status: {} \n'.format(m.Status))
    
    return search_time, solved

def extract_square_brackets(text):
    pattern = r'\[(.*?)\]'
    matches = re.findall(pattern, text)
    return matches[0].split(',')

def write_solution(x, solution_file, print=False):
    for v in x.values():
        if v.X == 1:
            s = extract_square_brackets(v.VarName)
            solution_file.write("flow {} is migrated at time step {} \n".format(s[0], s[1]))
            if print:
                print("flow {} is migrated at time step {} \n".format(s[0], s[1]))
    solution_file.write("\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    description="Generating the secondary structures dataset"
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="datasets/flows/dataset_1",
        help="the file name to store the dataset",
    )
    parser.add_argument(
        "--solution_file",
        type=str,
        default="MIP_indSet_solutions.txt",
        help="file to write the solutions to",
    )
    parser.add_argument(
        "--runtime_file",
        type=str,
        default="MIP_indSet_runtimes.txt",
        help="file to write the runtimes to",
    )
    args = parser.parse_args()
    db = shelve.open(args.dataset_file)
    dataset_size = sh['dataset_size']
    runtime_file = open(args.runtime_file, "a")
    solution_file = open(args.solution_file, "a") 
    
    total_time, total_instances_solved = 0, 0
    
    print('Solving the MIP on {}'.format(args.dataset_file))
    for i in range(0,  dataset_size):
        try:
            # load the graph
            start_1= timeit.default_timer()
            dic = sh[str(i)]
            G = dic['graph'] 
            u, v = dic['size']
            biadj_matrix = bipartite.biadjacency_matrix(G, np.array(range(u)), u + np.array(range(v))).todense()
            search_time, solved = solve_independet_set(biadj_matrix, u, v)
            end_1= timeit.default_timer()
            runtime_file.write("Total time: {} \n".format(end_1 - start_1))
            total_time += search_time
            total_instances_solved += solved
        except EOFError:
            break
    
    avg_time = total_time/total_instances_solved if total_instances_solved != 0 else 0
    runtime_file.write('avg_time solving {} instances in {}: {}\n\n'.format(total_instances_solved, args.dataset_file, avg_time))
    runtime_file.close()
    solution_file.close()
    db.close()
