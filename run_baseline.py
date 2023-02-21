#from xmlrpc.client import boolean
import networkx as nx
from baseline_algorithms.general_bipartite import realize
from networkx.algorithms import bipartite
import numpy as np
import pickle 
import time
import argparse
import json
import shelve
import os
from tqdm import tqdm
""""
See for more detail: https://bmarchand-perso.gitlab.io/bisr-dpw/
"""

def main(dataset, write_k):
    try:
        os.path.isfile(dataset)
    except:
        print("The dataset file does not exist")
    
    print('running the baseline algorithm on {}'.format(dataset))

    runtimes = []
    k_dist = []
    paths = []
    paths_lens = []
    #with open(dataset, "rb") as input_file:
    sh = shelve.open(dataset)
    dataset_size = sh['dataset_size'] 
    for i in tqdm(range(0,  dataset_size)):
        try:
            # load the graph
            dic = sh[str(i)]
            G = dic['graph'] 
            u, v = dic['size']
            biadj_matrix = bipartite.biadjacency_matrix(G, np.array(range(u)), u + np.array(range(v))).todense()
            B, R = set(range(u)), set(u + np.array(range(v)))

            # solve the reconfiguration instance while always having at least |B|−k vertices
            start = time.time()
            k=0
            while True:
                #try with k
                P = realize(B,R,G,k)
                if P:
                    break
                k += 1
            end = time.time()

            #print("found k=", k)
            #print("found P=", P)
            #print('biadj_matrix: ', biadj_matrix)
            #print(np.sum(biadj_matrix, axis = 1))

            runtimes.append(end - start)
            k_dist.append(k)
            paths.append(P)
            paths_lens.append(len(P))

            # write the k value obtained by the baseline m-MIS algorithm to the file
            if write_k:
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
    
    with open('results/basline_results.json', 'a') as f:
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
        default='datasets/expert_dbCRW_AND_entry_typeSequence_70by70',
        help="the file to load the dataset from",
    )
    parser.add_argument(
        "--write_k",
        type=bool,
        default=True,
        help="write the k value obtained by the baseline m-MIS algorithm for each graph with key 'k'",
    )
    args = parser.parse_args()
    if args.dataset == None:
        print('must specify a file to read from')
    else:
        main(args.dataset, args.write_k)