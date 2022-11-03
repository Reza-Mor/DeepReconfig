import matplotlib.pyplot as plt
import numpy as np
import json
import os

dataset = 'expert_dbCRW_AND_entry_typeSequence_10by10'
directory = 'plots/{}'.format(dataset)

if not os.path.exists(directory):
    os.makedirs(directory)

file = 'results/{}.json'.format(dataset)
Action_Length = []
checkpoints = []

for line in open(file, 'r'):
    dict = json.loads(line)

    # fig for energy 
    plt.figure()
    plt.title("Energy vs Time, checkpoint: {}".format(dict['chkpt']))
    plt.plot([i for i in range(len(dict['Energy']))], dict['Energy'])
    plt.xlabel('Time Step')
    plt.ylabel('Energy')
    # plt.legend()
    plt.savefig(directory + '/{}_Energy.png'.format(dict['chkpt']))

    # fig for number of LHS and RHS nodes selected
    plt.figure()
    plt.title("LHS vs RHS, checkpoint: {}".format(dict['chkpt']))
    plt.plot([i for i in range(len(dict['L_selected']))], dict['L_selected'], label="Number of left nodes selected")
    plt.plot([i for i in range(len(dict['R_selected']))], dict['R_selected'], label="Number of right nodes selected")
    plt.xlabel('Time Step')
    plt.ylabel('Number of nodes')
    plt.legend()
    plt.savefig(directory + '/{}_Nodes.png'.format(dict['chkpt']))

    Action_Length.append(len(dict["Actions"]))
    checkpoints.append(dict["chkpt"])


plt.figure()
plt.title("Number of Actions in rollout from checkpoint i")
plt.plot(checkpoints, Action_Length)
plt.xlabel('checkpoint')
plt.ylabel('Number of actions to completion (or termination)')
plt.savefig(directory + '/Actions.png')
