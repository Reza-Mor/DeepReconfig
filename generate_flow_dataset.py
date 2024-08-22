from queue import PriorityQueue
import numpy as np
from numpy import random
import random
import networkx as nx
import numpy as np
import shelve
import argparse

def furthest_nodes(graph, start_node=0):
    n = graph.shape[0]
    distances = np.full(n, np.inf)
    previous = np.full(n, -1, dtype=int)
    unvisited = set(range(n))

    # Use Dijkstra's algorithm to find the shortest distances from the start node to all other nodes
    distances[start_node] = 0
    while unvisited:
        current = np.argmin(distances[list(unvisited)])
        current = list(unvisited)[current]
        unvisited.remove(current)
        for neighbor in range(n):
            if graph[current, neighbor] == 0:
                continue
            distance = distances[current] + graph[current, neighbor]
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current

    # Find the node with the maximum distance from the start node
    furthest = np.argmax(distances)

    # Trace the path from the furthest node back to the start node
    path = [furthest]
    while previous[furthest] != -1:
        furthest = previous[furthest]
        path.append(furthest)
    path.reverse()

    # Return the list of nodes from closest to furthest 
    return path

def Astar(graph, start, end):
    rows, cols = np.shape(graph)
    distances = [float('inf') for _ in range(rows)]
    distances[start] = 0
    previous = [-1 for _ in range(rows)]
    queue = PriorityQueue()
    queue.put((0, start))
    
    while not queue.empty():
        current_distance, current_node = queue.get()
        if current_node == end:
            return distances[end], previous
        for neighbor in range(cols):
            if graph[current_node, neighbor] == 0:
                continue
            distance = distances[current_node] + graph[current_node, neighbor]
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                queue.put((distance, neighbor))
                
    return float('inf'), previous

def get_path(previous, end):
    path = [end]
    while previous[end] != -1:
        end = previous[end]
        path.append(end)
    return path[::-1]

def generate_graph(max_capacity, min_capacity, graph_size, er_param, seed):
    graph = nx.fast_gnp_random_graph(graph_size, er_param, seed, directed=True)
    weights = np.random.randint(min_capacity, max_capacity + 1, (graph_size,graph_size), dtype=int)
    G = np.multiply(nx.adjacency_matrix(graph).todense(), weights)
    #graph = nx.from_numpy_matrix(A,create_using=nx.MultiDiGraph())
    return G

def generate_init_flows(G, avg_utilization, min_flow_size, max_flow_size, start, end, db, seed):
    
    # generate the flow paths in the intial configuration
    random.seed(seed)
    #G_init = G.copy()
    flow_id = 0
    init_flows = {}
    G_temp = G.copy()
    capacity_used = 0
    network_capacity = G.sum()
    edges = {}
    global longest_flow_length
    longest_flow_length = 0
    dist = 0

    while (dist != float('inf') or max_flow_size != min_flow_size): #avg_link_utilization <= utilization:
        # we set the flow size to go from 
        flow_size = random.randint(min_flow_size, max_flow_size)
        search_graph = np.where(G_temp - flow_size<0, 0, G_temp - flow_size)
        #randomness = np.random.randint(0, 2, size=search_graph.shape)
        dist, previous = Astar(search_graph, start, end)
        path = get_path(previous, end)

        if dist != float('inf'):
            
            if len(path) > longest_flow_length:
                longest_flow_length = len(path)
            for i in range(1, len(path)):
                edge = path[i-1], path[i]
                if edge not in edges.keys():
                    edges[edge] = G_temp[edge]
                G_temp[edge]  = G_temp[edge] - flow_size
            capacity_used += flow_size * (len(path) -1)

            #if edge in edges.keys():
            #    capacity += A[edge]
            #print('flow size: {}, capacity used: {}, network capacity: {}'.format(flow_size, capacity_used, network_capacity))
            #avg_link_utilization = capacity_used / network_capacity
            init_flows[flow_id] = path + [flow_size]
            flow_id += 1
        else:
            max_flow_size = max(min_flow_size, max_flow_size//2)
    
    link_capacity = sum(edges.values())
    assert link_capacity != 0, "no path from source to target (no flow could be added)"
        
    
    while capacity_used/link_capacity > avg_utilization:
        # randomly remove flow from the flows
        flow = random.choice(list(init_flows.keys()))
        f = init_flows.pop(flow)
        path = f[:-1]
        flow_size = f[-1]
        flow_id -= 1
        capacity_used -= flow_size * (len(path) -1)
        for i in range(1, len(path)):
            edge = path[i-1], path[i]
            G_temp[edge]  = G_temp[edge] + flow_size
            # if the edge is empty of flows
            if G_temp[edge] == edges[edge]:
                link_capacity -= edges[edge]
                edges.pop(edge)

    print('average link utilization: {}'.format(capacity_used/link_capacity))
    #print('total network capacity: {}, link_capacity : {}'.format(network_capacity, link_capacity))
    print('number of flows in each configuration: {}'.format(flow_id))
    db['num_flows'] = flow_id
    
    #print('longest flow length: ', longest_flow_length)
    init_flows = dict((key, value) for (key, value) in zip(range(len(init_flows)), init_flows.values()))
    #print('number of flows in each configuration: {}'.format(len(init_flows)))

    G_induced = np.where(G == G_temp, 0, G_temp)
    #G_induced2 = np.where(G- G_temp == G, 0, G)

    db['config_0'] = init_flows #, G_induced #, G_induced2

    return init_flows
    
def generate_configurations(G, init_flows, num_configs, db, seed):
    # generate the flow paths in the target configuration
    random.seed(seed)
    num_flows = len(init_flows)
    global longest_flow_length

    for j in range(1, num_configs):
        flow_ids = random.sample(range(num_flows), num_flows)
        
        target_flows = {}
        G_temp = G.copy()

        for flow in flow_ids:
            flow_size = init_flows[flow][-1]
            search_graph = np.where(G_temp - flow_size < 0, 0, G_temp - flow_size)
            dist = float('inf')
            while dist == float('inf'):
                randomness = np.random.randint(0, 2, size=search_graph.shape)
                dist, previous = Astar(search_graph* randomness, start, end)
                path = get_path(previous, end)

                if len(path) > longest_flow_length:
                    longest_flow_length = len(path)

            if dist != float('inf'):
                for i in range(1, len(path)):
                    edge = path[i-1], path[i]
                    G_temp[edge]  = G_temp[edge] - flow_size
                target_flows[flow] = path + [flow_size]
            else:
                print('could not find a target flow path for flow {}'.format(flow))
        
        G_induced = np.where(G == G_temp, 0, G_temp)
        #G_induced = np.where(G- G_temp == 0, 0, G_temp)
        #G_induced2 = np.where(G- G_temp == G, 0, G)
        #print(G_induced)

        db['config_{}'.format(j)] = target_flows #, G_induced #, G_induced2
        #print('target_flows: ', target_flows)

def compute_distance(dict1, dict2):
    same_flows = 0
    for key in dict1:
        if dict1[key] == dict2[key]:
            same_flows += 1
    return same_flows
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Generating the flows dataset"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default='/home/reza-m/DeepReconfig/datasets/flows/100_0.5ER_0.4util', #"~/DeepReconfig/datasets/flows/dataset_10",
        help="the file name to store the dataset",
    )
    parser.add_argument(
        "--min_link_capacity",
        type=int,
        default=5,
        help="",
    )
    parser.add_argument(
        "--max_link_capacity",
        type=int,
        default=5,
        help="",
    )
    parser.add_argument(
        "--min_flow_size",
        type=int,
        default=1,
        help="",
    )
    parser.add_argument(
        "--max_flow_size",
        type=int,
        default=1,
        help="",
    )
    parser.add_argument(
        "--graph_size",
        type=int,
        default=100,
        help="number of nodes in the graph",
    )
    parser.add_argument(
        "--num_configs_train",
        type=int,
        default=100,
        help="number of flow migration configurations to generate",
    )
    parser.add_argument(
        "--num_configs_test",
        type=int,
        default=10,
        help="number of flow migration configurations to generate",
    )
    parser.add_argument(
        "--avg_link_utilization",
        type=float,
        default=0.4,
        help="avrage link utilization in the configurations",
    )
    parser.add_argument(
        "--er_parameter",
        type=float,
        default=0.5,
        help="the ER parameter deciding the density of the graph",
    )
    args = parser.parse_args()
    
    assert args.max_flow_size <= args.min_link_capacity
    assert 0 < args.avg_link_utilization < 1
    
    print('Generating flows datasets with the following args: {}'.format(args))

    G = generate_graph(args.max_link_capacity, args.min_link_capacity, args.graph_size, args.er_parameter, 0)
    
    db_train, db_test = shelve.open(args.output_file + '_train'),  shelve.open(args.output_file + '_test')
    db_train['G'] , db_test['G']= G, G
    db_train['num_configs'], db_test['num_configs'] = args.num_configs_train, args.num_configs_test
    db_train['max_capacity'], db_test['max_capacity'] = args.max_link_capacity, args.max_link_capacity
    db_train['min_capacity'], db_test['min_capacity'] = args.min_link_capacity, args.min_link_capacity
    db_train['min_flow_size'], db_test['min_flow_size'] = args.min_flow_size, args.min_flow_size
    db_train['max_flow_size'], db_test['max_flow_size'] = args.max_flow_size, args.max_flow_size
    db_train['avg_link_utilization'], db_test['avg_link_utilization'] = args.avg_link_utilization, args.avg_link_utilization
    db_train['er_parameter'], db_test['er_parameter'] = args.er_parameter, args.er_parameter
    db_train['graph_size'], db_test['graph_size']= args.graph_size, args.graph_size

    start = 0 #source node
    end = furthest_nodes(G)[-1] #sink node

    init_flows = generate_init_flows(G, args.avg_link_utilization, args.min_flow_size, args.max_flow_size, start, end, db_train, 0)
    
    generate_configurations(G, init_flows, args.num_configs_train, db_train, 0)
    generate_configurations(G, init_flows, args.num_configs_test, db_test, 1)

    db_train['num_flows'], db_test['num_flows'] = len(init_flows), len(init_flows)
    # check if this has to be different train and test datasets
    db_train['longest_flow_length'], db_test['longest_flow_length']= longest_flow_length, longest_flow_length

    db_train.close()
    db_test.close()
    #for i in range(len(configs)):
    #    print('configuration {}: {}'.format(i, configs[i]))
    #    print('{} flows in configurations 0 and {} have the same path'.format(compute_distance(configs[0], configs[i]), i))
