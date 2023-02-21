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

def generate_init_flows(G, avg_utilization, min_flow_size, max_flow_size, start, end, seed):
    
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
        dist, previous = Astar(search_graph, start, end)
        path = get_path(previous, end)

        if dist != float('inf'):
            #print('path: ', path)
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
    while capacity_used/link_capacity > avg_utilization:
        f = init_flows.pop(flow_id-1)
        path = f[:-1]
        flow_size = f[-1]
        flow_id -= 1
        capacity_used -= flow_size
        for i in range(1, len(path)):
            edge = path[i-1], path[i]
            G_temp[edge]  = G_temp[edge] + flow_size

    print('link utilization: {}'.format(capacity_used/link_capacity))
    print('total network capacity: {}, link_capacity : {}'.format(network_capacity, link_capacity))
    print('number of flows in each configuration: {}'.format(flow_id))
    print('longest flow length: ', longest_flow_length)

    G_induced = np.where(G- G_temp == 0, 0, G_temp)

    print("G: ", G)
    print("G_induced: ", G_induced)
    print("init_flows: ", init_flows)
    db['config_0'] = init_flows , G_induced

    return init_flows
    
def generate_configurations(G, init_flows, num_configs):
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
            serach_graph = np.where(G_temp - flow_size < 0, 0, G_temp - flow_size)
            dist, previous = Astar(serach_graph, start, end)
            path = get_path(previous, end)

            if len(path) > longest_flow_length:
                longest_flow_length = len(path)

            if dist != float('inf'):
                for i in range(1, len(path)-1):
                    edge = path[i-1], path[i]
                    G_temp[edge]  = G_temp[edge] - flow_size
                target_flows[flow] = path + [flow_size]
            else:
                print('could not find a target flow path for flow {}'.format(flow))
        
        G_induced = np.where(G- G_temp == 0, 0, G_temp)

        db['config_{}'.format(j)] = target_flows , G_induced

def compute_distance(dict1, dict2):
    same_flows = 0
    for key in dict1:
        if dict1[key] == dict2[key]:
            same_flows += 1
    return same_flows
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Generating the secondary structures dataset"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="datasets/flows/dataset_1",
        help="the file name to store the dataset",
    )
    args = parser.parse_args()

    min_capacity = 10
    max_capacity = 10
    min_flow_size = 1
    max_flow_size = 5
    er_param = 0.7
    graph_size = 150
    num_configs = 2
    seed = 1
    avg_utilization = 0.8

    G = generate_graph(max_capacity, min_capacity, graph_size, er_param, seed)
    
    db = shelve.open(args.output_file)
    db['G'] = G
    db['num_configs'] = num_configs
    db['max_capacity'] = max_capacity
    db['min_capacity'] = min_capacity
    db['min_flow_size'] = min_flow_size
    db['max_flow_size'] = max_flow_size

    start = 0
    end = furthest_nodes(G)[-1]

    init_flows = generate_init_flows(G, avg_utilization, min_flow_size, max_flow_size, start, end, seed)
    generate_configurations(G, init_flows, num_configs)


    db['num_flows'] = len(init_flows)
    db['longest_flow_length'] = longest_flow_length
    db.close()

    #for i in range(len(configs)):
    #    print('configuration {}: {}'.format(i, configs[i]))
    #    print('{} flows in configurations 0 and {} have the same path'.format(compute_distance(configs[0], configs[i]), i))
