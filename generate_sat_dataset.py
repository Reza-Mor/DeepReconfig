from ortools.sat.python import cp_model
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
import random
import matplotlib.pyplot as plt

class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.solutions = []
        self.__solution_count = 0

    def on_solution_callback(self) -> None:
        self.__solution_count += 1
        solution = []
        for v in self.__variables:
            solution.append(self.value(v))
            #print(f"{v}={self.value(v)}", end=" ")
        #print()
        self.solutions.append(np.array(solution))

    @property
    def solution_count(self) -> int:
        return self.__solution_count
    
    def get_all_solutions(self):
        return self.solutions
    
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

def find_all_solutions(CNF):
    # Create the SAT model
    model = cp_model.CpModel()
    
    vars= []
    V = int(max(abs(np.min(CNF)), np.max(CNF))) # number of variables in the CNF
    for i in range(V): #change this to V
        vars.append(model.NewBoolVar('x{}'.format(i)))
    
    # Create the constraints.
    for row in CNF:
        clause_vars = []
        for var in row:
            if var > 0:
                clause_vars.append(vars[var-1])
            elif var < 0:
                clause_vars.append(vars[abs(var)-1].Not())
        model.AddBoolOr(clause_vars)

    # Create a solver
    solver = cp_model.CpSolver()
    solution_printer = VarArraySolutionPrinter(vars)
    # Enumerate all solutions.
    solver.parameters.enumerate_all_solutions = True
    # Solve.
    status = solver.Solve(model, solution_printer)
    
    print(f"Status = {solver.status_name(status)}")
    print(f"Number of solutions found: {solution_printer.solution_count}")
    #print(f"Solutions found: {solution_printer.solutions}")
    return solution_printer.solutions

def check_feasibility(CNF, V, value_assignment):
    """
    Given a CNF instance and a value assignment to the variables in the CNF, 
    returns if the assignment satisfies the SAT instance and the percent of satisfied cluases
    """
    for variable in range(1, V+1):
        CNF = np.where(CNF == variable, value_assignment[variable-1], CNF)
        CNF = np.where(CNF == -1*variable, 1 - value_assignment[variable-1], CNF)
    
    C = np.where(CNF.sum(axis = 1) > 0, True, False) #values assignment of each clause
    percent_of_satisfiable_clauses = C.sum() / C.shape[0]
    satisfiable = percent_of_satisfiable_clauses == 1.0
    return percent_of_satisfiable_clauses, satisfiable

def create_sat_dataset(CNF, solutions, size, output_path):
    random.seed(0)
    assert size > len(solutions), "the size of the dataset should be larger than the number of satisfying assignments to the SAT"
    
    db = shelve.open(output_path)
    db['CNF'] = CNF
    
    V = int(max(abs(np.min(CNF)), np.max(CNF))) # number of variables in the CNF
    
    for s in range(len(solutions)):
        db['config_{}'.format(s)] = (solutions[s], 1.0, True)
    
    for i in range(size - len(solutions)):
        random_solution = random.choice(solutions) #pick a random solution
        num_vars = random.randrange(1, (V+1)) #pick a random number of variables to flip
        random_var_indices = random.sample(list(random_solution), num_vars)
        new_config = random_solution.copy()
        new_config[random_var_indices] = 1 - random_solution[random_var_indices] # flip the truth values
        percent_of_satisfiable_clauses, satisfiable = check_feasibility(CNF, V, new_config)
        db['config_{}'.format(i + s)] = (new_config, percent_of_satisfiable_clauses, satisfiable)
        
    db.close()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description=""
    )
    parser.add_argument(
        "--SAT_instance",
        type=str,
        default= "datasets/SAT/CBS_k3_n100_m403_b10/CBS_k3_n100_m403_b10_0.cnf", #"../datasets/SAT/uf20-91",
        help="the path to the file where the SAT instances are stored - refer to https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=0,
        help="size of the dataset- only take the first data_size instances from the SAT_dataset. Take all of the dataset if not specified.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        #default="datasets/SAT/reconfig/CBS_k3_n100_m403_b10_0_reconfig",
        default="datasets/SAT/reconfig/toy_reconfig",
        help="the file name to store the dataset",
    )
    args = parser.parse_args()

    """
    if not os.path.exists('../datasets/SAT/reconfig'):
        # Create a new directory because it does not exist
        os.makedirs('../datasets/SAT/reconfig')

    # open the SAT dataset
    sat_instance = args.SAT_instance
    #CNF = parse_fie(sat_instance)
    
    CNF = np.array([[1, -2, 4],
                    [-1, 2, 3],
                    [-1, -2, 3],
                    [-1, 2, 4],
                    [1, 3, 4],
                    [-1, -2, 4],
                    [-5, 1, 4],
                    [-5, 2, 4],
                    [-5, 2, -3],
                    [5, 1, 3],
                    [5, 3, 2],
                    [5, 3, 4],
                    [5, -4, 1],
                    ])
    
    size = args.dataset_size
    i = 0
    solutions = find_all_solutions(CNF)
    #create_sat_dataset(CNF, solutions, 1000000, args.output_path)
    create_sat_dataset(CNF, solutions, 1000, args.output_path)
    """
    
    lst = []
    #db = shelve.open('../datasets/SAT/reconfig/CBS_k3_n100_m403_b10_0_reconfig')
    db = shelve.open('../datasets/SAT/reconfig/toy_reconfig')
    for k in db.keys():
        print(k)
    for config in db['configurations']:
        _, percent_of_satisfiable_clauses, _ = lst.append(percent_of_satisfiable_clauses)
    plt.hist(lst)
    plt.savefig('../datasets/SAT/reconfig/CBS_k3_n100_m403_b10_0_reconfig_hist.jpg')
    db.close()
    

    """
    A_p, A_m = get_biadj_matrix(CNF)
    print(np.where(A_p > 0))
    print(A_p[39, 0])
    main()
    """
