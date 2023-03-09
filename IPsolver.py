import gurobipy as gp
from gurobipy import GRB
import itertools
import numpy as np
import timeit
import shelve
import re
import argparse

def get_data_flows(adjacency_matrix, init_flows, target_flows):
    """
    pre-process the data for groubi for the flow migration problem
    
    Outputs:
    E1 = {edge: {flow_id: flow_size}} in initial flows
    E2 = {edge: {flow_id: flow_size}} in target flows
    E3 = {edge: capacity}

    Example of parameters:
    init_flows:  {0: [0, 8, 2, 1], 1: [0, 10, 2, 1], 2: [0, 15, 2, 1], 3: [0, 4, 3, 2, 1]}
    target_flows:  {1: [0, 8, 2, 1], 2: [0, 10, 2, 1], 0: [0, 15, 2, 1], 3: [0, 4, 3, 2, 1]}
    Where 0, 1, 2, 3 are flow ids. The last element in each list is the flow size. The rest of each list is the flow path.
    """
    E1, E2 , E3= {}, {}, {}

    n = adjacency_matrix.shape[0]
    for v, u in itertools.product(range(n), range(n)):
        if adjacency_matrix[v, u] != 0:                
            E1[(v, u)] = {}
            E2[(v, u)] = {}
            E3[(v, u)] = adjacency_matrix[v, u]
    
    for flow_id in init_flows.keys():
        dic = {flow_id: init_flows[flow_id][-1]}
        
        init_path = init_flows[flow_id][:-1]
        for i in range(1, len(init_path)):
            edge = init_path[i-1], init_path[i]
            E1[edge].update(dic)
            

        target_path = target_flows[flow_id][:-1]
        for j in range(1, len(target_path)):
            edge = target_path[j-1], target_path[j]
            E2[edge].update(dic)
        
    return E1, E2 , E3

def solve_flows(E1, E2, E3, num_flows, minimize_rounds):
    try:
        # Create a new model
        m = gp.Model("mip")
        m.Params.LogToConsole = 0
        F = num_flows #number of flows
        R = num_flows  #number of rounds

        # Create variables
        x = m.addVars(F, R, vtype="B", name="x")
        if minimize_rounds:
            y = m.addVars(R, vtype="B", name="y")

        d = {}
        for edge in E1.keys():
            for flow in range(num_flows):
                d[edge, flow] = m.addVar(vtype=GRB.CONTINUOUS, name="d{},{}".format(edge, flow))
        
        m.addConstrs((x.sum(f, "*") == 1 for f in range(F)), "3")

        if minimize_rounds:
            m.addConstrs((x.sum("*", k) <= y[k] * num_flows for k in range(R)), "4")
        
        #E1 = {edge: {flow_id: flow_size}}
        for e in E1.keys():
            for k in range(R):
                sum = 0
                #for flow_id in range(F):

                for flow_id in E1[e].keys():
                    inner_sum = 0
                    for t in range(k-1):
                        inner_sum += x[flow_id,t]
                    c1_fe = E1[e][flow_id]

                for flow_id in E2[e].keys():
                    inner_sum_2 = 0
                    for t in range(k-1):
                        inner_sum_2 += x[flow_id,t]
                    c2_fe = E2[e][flow_id]
                    
                    sum += c2_fe * inner_sum_2 + c1_fe * (1 - inner_sum)
                
                m.addConstr(d[e,k] == sum, name='6')

        for e in E3.keys():
            for k in range(R):
                sum_2 = 0
                #for flow_id in range(F):
                    #t = E1[e][flow_id] if flow_id in E1[e].keys() else 0
                for flow_id in E2[e].keys():
                    sum_2 += x[flow_id, k] * E2[e][flow_id]
                m.addConstr(d[e,k] + sum_2 <= E3[e], name='5')

        obj = 0

        if minimize_rounds:
            # minimize the number of rounds for full migration
            for k in range(R):
                obj += y[k]
            m.setObjective(obj, GRB.MINIMIZE)
        else:
            # maximize the number of flows migrated by the end of R rounds
            for f in range(F):
                for r in range(R):
                    obj += x[f, r]
                    m.setObjective(obj, GRB.MAXIMIZE)
            # stop as soon as found
            m.params.BestObjStop = F

        m.setParam('TimeLimit', 5*60) # time limit of 5 minutes
        m.optimize()

        #print(m.Status)
        #print(m.objVal) 
        
    except gp.GurobiError as e:
        print("Error code " + str(e.errno) + ": " + str(e))
    except AttributeError:
        print("Encountered an attribute error")
    
    return m.Status, x

def extract_square_brackets(text):
    pattern = r'\[(.*?)\]'
    matches = re.findall(pattern, text)
    return matches[0].split(',')

def write_solution(x, solution_file, print= False):
    for v in x.values():
        if v.X == 1:
            s = extract_square_brackets(v.VarName)
            solution_file.write("flow {} is migrated at time step {}".format(s[0], s[1]))
            if print:
                print("flow {} is migrated at time step {}".format(s[0], s[1]))

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
        "--minimize_rounds",
        type=bool,
        default=False,
        help="solve the mip to minimize the rounds. If false, maximize the number of flows migrated by the end of R rounds",
    )
    parser.add_argument(
        "--solution_file",
        type=str,
        default="MIP_flows_solutions.txt",
        help="file to write the solutions to",
    )
    parser.add_argument(
        "--runtime_file",
        type=str,
        default="MIP_flows_runtimes.txt",
        help="file to write the runtimes to",
    )
    args = parser.parse_args()
    db = shelve.open(args.dataset_file)
    num_configs = db['num_configs']
    num_flows = db['num_flows']
    
    runtime_file = open(args.runtime_file, "a")
    solution_file = open(args.solution_file, "a")
    runtime_file.write("{} \n\n".format(args.dataset_file))
    solution_file.write("{} \n\n".format(args.dataset_file))

    total_time = 0
    num_solved = 0
    print('solving the MIP...')
    for i1 in range(0, num_configs-1):
        init_config, _, adj_matrix1 = db['config_{}'.format(i1)]
        target_config, _, adj_matrix2= db['config_{}'.format(i1 + 1)]
        adj_matrix = np.maximum(adj_matrix1, adj_matrix1)
        E1, E2 , E3 = get_data_flows(adj_matrix, init_config, target_config)
        start = timeit.default_timer()
        status, x = solve_flows(E1, E2, E3, num_flows, args.minimize_rounds)
        stop = timeit.default_timer()
        
        if status == 2: #an optimal solution is available
            runtime_file.write('Time solving instance {}: {} \n'.format(i1, stop - start))
            write_solution(x, solution_file, print = False)
            num_solved += 1
        else:
            runtime_file.write('Encountered an error solving instance {}. Status: {} \n'.format(i1, status))
        total_time += stop - start

    avg_time = total_time/(num_configs-1)
    runtime_file.write('avg_time solving {} instances in {}: {}'.format(num_solved, args.dataset_file, avg_time))
    runtime_file.close()
    solution_file.close()
    db.close()
