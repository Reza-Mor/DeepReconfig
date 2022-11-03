import json
import subprocess
import argparse
from tqdm import tqdm
import networkx as nx
import pickle
from networkx.algorithms import bipartite
import shelve

def create_bipartite_graph(structure_1, structure_2, max_graph_size):
    """
    the structures follow the Dot-Bracket notation (see https://www.tbi.univie.ac.at/RNA/documentation.html#api) 
    Dot-Bracket notation --> bipartite graph
    """

    stack_1 = []
    pairs_1 = []
    stack_2 = []
    pairs_2 = []

    for i in range(len(structure_1)):
            if structure_1[i] == '(':
                stack_1.append(i)
            elif structure_1[i] == ')':
                pairs_1.append((stack_1.pop(), i))
            if structure_2[i] == '(':
                stack_2.append(i)
            elif structure_2[i] == ')':
                pairs_2.append((stack_2.pop(), i))

    # make nodes for each bond
    G = nx.Graph()
    V = 0
    U = 0

    if max_graph_size != None and max_graph_size < len(pairs_1):
        m = max_graph_size
    else:
        m = len(pairs_1)

    if max_graph_size != None and max_graph_size < len(pairs_2):
        n = max_graph_size
    else:
        n = len(pairs_2)

    for v in range(m):
        G.add_node(v, start=pairs_1[v][0], end=pairs_1[v][1])
        V += 1
    for u in range(n):
        G.add_node(u + m, start=pairs_2[u][0], end=pairs_2[u][1])
        U += 1
        # connect an edge if two nodes are crossing
        for v in range(m):
            if G.nodes[v]['start'] <= G.nodes[u + m]['start'] <= G.nodes[v]['end'] or G.nodes[v]['start'] <= G.nodes[u + m]['end'] <= G.nodes[v]['end']:
                G.add_edge(v,u + m)
                # print('added edge: [{},{}]'.format(pairs_1[v], pairs_2[u]))

    return G, V, U



"""
outputs a pkl dataset consisting of the RNA strand and the bipartite graph made from two secondary structures from the strand
The input file must be of json format outlined in https://rnacentral.org/ (download any of the datasets in json format)
There are two secondary structure created per RNA strand, see RNAsubopt -p https://www.tbi.univie.ac.at/RNA/RNAsubopt.1.html)
"""
def create_dataset(input_file, output_file, dataset_size, max_string_length, max_graph_size):
    print("Creating a dataset of {}by{} graphs (RNA string length: {})".format(max_graph_size, max_graph_size, max_string_length))
    i = 0
    data = []
    graph_size_avg = 0
    encoding = 'utf-8'
    db = shelve.open(output_file)
    # write meta data
    db['dataset_size'] = dataset_size
    db['dataset_name'] = input_file 
    db['max_string_length'] = max_string_length
    # we store the biggest graph size in case we use vairable sized graphs
    db['max_graph_size']= max_graph_size 
    with open(input_file) as f:
        datas = json.load(f)
        for data in tqdm(datas):
            if max_string_length != None:
                sequence = data['sequence'][:max_string_length] if len(data['sequence']) > max_string_length else data['sequence']
            else:
                sequence = data['sequence']
            # the command used to generate secondary bonds
            command = "echo '{}' | RNAsubopt -p 2".format(sequence)
            output = subprocess.check_output(command, shell=True).decode(encoding).split('\n')
            G , u, v = create_bipartite_graph(output[1], output[2], max_graph_size)
            graph_size_avg += v
            dict = {
                "sequence": output[0],
                "graph": G,
                "size": (u,v),
                "k": None
                }
            #pickle.dump(dict, out_file)
            db[str(i)] = dict
            if i >= dataset_size - 1:
                break
            i += 1
    db['avg_v_size'] = graph_size_avg/(i+1)
    print('avg_v_size: ', graph_size_avg/(i+1))
    print("Dataset of size {} created".format(i + 1))
    #out_file.close()
    f.close()
    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generating the secondary structures dataset"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="datasets/expert_dbCRW_AND_entry_typeSequence.json",
        help="The input file must be of json format outlined in https://rnacentral.org/ (download any of the datasets in json format)",
    )
    parser.add_argument(
        "--dataset_size", type=int, default=1, help="The dataset size (limited by the number of data points in input_file"
    )
    parser.add_argument(
        "--max_string_len", type=int, default=350, help="The maximum RNA string length (get RNA[:max_string_len], throw the extra '(' brackets away)"
    )
    parser.add_argument(
        "--max_graph_size", type=int, default=20, help="The maximum bipartite graph size (if inputted n, will get n by n graph)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="datasets/dataset1_20by20",
        help="the file name to store the dataset",
    )
    args = parser.parse_args()

    if args.max_string_len != None and args.max_string_len > 1000:
        print("WARNING: Computing the secondary structure grows exponentially with both sequence length and energy range")
        print("Computing a pair of secondary structure for a single string of length 1000 can take up to 10 seconds")

    create_dataset(args.input_file, args.output_file, args.dataset_size, args.max_string_len, args.max_graph_size)