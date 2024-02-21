#from xmlrpc.client import boolean
import networkx as nx
from baseline_algorithms.general_bipartite import realize
#from ..MIPs.IndSet_MIP import solve_independet_set_matrix
from networkx.algorithms import bipartite
import numpy as np
import pickle 
import time
import argparse
import json
import shelve
import os
from tqdm import tqdm


import gurobipy as gp
from gurobipy import GRB

def solve_independet_set_matrix(biadjacency_matrix, U, V):
    
    # Create a new model
    m = gp.Model("mip")
    m.Params.LogToConsole = 0
    T = U + V  #number of time steps
    N = U + V# number of nodes

    # Create variables
    X_v = m.addMVar((V, T), vtype="B", name="x_v")
    X_u = m.addMVar((U, T), vtype="B", name="x_u")
    delta_v = m.addMVar((V, T), vtype="I", name="delta_v")
    delta_u = m.addMVar((U, T), vtype="I", name="delta_u")

    #absolute values of deltas
    delta_v1 = m.addMVar((V, T), vtype="I", name="delta_v1")
    delta_v2 = m.addMVar((V, T), vtype="I", name="delta_v2")
    delta_u1 = m.addMVar((U, T), vtype="I", name="delta_u1")
    delta_u2 = m.addMVar((U, T), vtype="I", name="delta_u2")
    m.addConstr(delta_v1 - delta_v2 == delta_v)
    m.addConstr(delta_u1 - delta_u2 == delta_u)

    k = m.addVar(0, U, vtype="INTEGER", name="k")
    
    edges_u = np.nonzero(biadjacency_matrix)[0]
    edges_v = np.nonzero(biadjacency_matrix)[1]

    for t in range(T):
        for v, u in zip(edges_v, edges_u):
            m.addConstr(X_v[v, t] + X_u[u, t] <= 1, name='2')

        m.addConstr((X_u[:,t] + X_v[:,t]).sum() >= U - k, name="3")

        if t != 0:
            m.addConstr(X_u[:,t] == X_u[:,t-1] + delta_u[:,t], name="4 a")
            m.addConstr(X_v[:,t] == X_v[:,t-1] + delta_v[:,t], name="4 b")
            
    # write constraint four
    m.addConstr(delta_v[:0] == 0)
    m.addConstr(delta_u[:0] == 0)

    for t in range(1,T):
        # note: delta_v1[:0] - delta_v2[:0] = abs(delta_v[:0])
        m.addConstr((delta_v1[:t] - delta_v2[:t]).sum() + (delta_u1[:t] - delta_u2[:t]).sum() == 1) 
    
    #delta is one of 1,0,-1
    m.addConstr(delta_v >= -1)
    m.addConstr(delta_v <= 1)
    m.addConstr(delta_u >= -1)
    m.addConstr(delta_u <= 1)

    m.addConstr(X_u[:, 0] == 1, name='5 a')
    m.addConstr(X_u[:, T-1] == 0, name='5 b')
    m.addConstr(X_v[:, 0] == 0, name='5 c')
    m.addConstr(X_v[:, T-1] == 1, name='5 d')

    m.setObjective(k, GRB.MINIMIZE)
    m.optimize()
    
    print('status: ', m.Status)
    if m.Status == 2: #an optimal solution is available
        print(k)
        return k.X


""""
See for more detail: https://bmarchand-perso.gitlab.io/bisr-dpw/
"""

def run_MIS(B,R,G):
    start = time.time()
    k=0
    while True:
        #try with k
        path = realize(B,R,G,k)
        if path:
            break
        k += 1
    end = time.time()
    return k, end-start, path

def run_MIP(biadjacency_matrix, U, V):
    start = time.time()
    k = solve_independet_set_matrix(biadjacency_matrix, U, V)
    end = time.time()
    path = []
    return k, end-start, path

def main(dataset, algorithm):
    try:
        os.path.isfile(dataset)
    except:
        print("The dataset file does not exist")
    
    print('running the {} algorithm on {}'.format(algorithm, dataset))

    runtimes = []
    k_dist = []
    paths = []
    paths_lens = []
    sh = shelve.open(dataset)
    dataset_size = sh['dataset_size']
    print(dataset_size)
    #for i in tqdm(range(0,  dataset_size)):
    for i in tqdm(range(0,  1)):
        try:
            # load the graph
            dic = sh[str(i)]
            G = dic['graph'] 
            u, v = dic['size']
            B, R = set(range(u)), set(u + np.array(range(v)))

            # solve the reconfiguration instance while always having at least |B|−k vertices
            if algorithm == 'm-MIS':
                k, run_time, P  = run_MIS(B,R,G)
            else:
                biadj_matrix = bipartite.biadjacency_matrix(G, np.array(range(u)), u + np.array(range(v))).todense()
                k, run_time, P = run_MIP(biadj_matrix, u, v)
            print('k: ', k)
            runtimes.append(run_time)
            k_dist.append(k)
            #paths.append(P)
            paths_lens.append(len(P))

            # write the k value obtained by the baseline m-MIS algorithm to the file
            dic['k'] = u - k
            sh[str(i)] = dic

        except EOFError:
            break
    
    # compute the average solution length found by the alorithm
    #print(paths)
    avg_sol_len = np.mean(paths_lens) #sum(list(map(lambda lst: len(lst), paths)))/len(paths)
    sol_len_std = np.std(paths_lens)
    max_sol_len = max(paths_lens)
    k_dist_non_zero = sum(list(map(lambda x: x!=0, k_dist)))

    dictionary = {
        'size': (u,v),
        'avg_runtime': sum(runtimes) / len(runtimes),
        'avg_k_dist': sum(k_dist) / len(k_dist),
        'avg_sol_len': avg_sol_len,
        'sol_len_std': sol_len_std,
        'max_sol_len': max_sol_len,
        'runtimes': runtimes,
        'k_dist': k_dist
    }

    sh['non_zero_k'] =  k_dist_non_zero
    

    if algorithm == 'm-MIS':
        outfile = '/home/moravejm/DeepReconfig/results/m_MIS_results.json'
    else:
        outfile = '/home/moravejm/DeepReconfig/results/MIP_results.json'

    with open(outfile, 'a') as f:
        json.dump(dictionary, f)
        f.write('\n')

    sh.close()
    f.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Running RNA baslines"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='/home/moravejm/DeepReconfig/datasets/RNA/dataset1_20by20',
        help="the file to load the dataset from",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default='MIP',
        help="algorithm to run, must be one of 'm-MIS' or 'MIP'",
    )
    args = parser.parse_args()
    assert args.algorithm in ['m-MIS','MIP']
    if args.dataset == None:
        print('must specify a file to read from')
    else:
        main(args.dataset, args.algorithm)