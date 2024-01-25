"""
Example doc string
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import csv

# from multiprocessing import Pool
# from torch_geometric.utils.convert import from_networkx #, to_networkx

import graph_utils
from opt_solve import solve_main
import data_file


graph_id = int(sys.argv[1])
instance_id = int(sys.argv[2])

NUM_NODES = data_file.NUM_NODES
NUM_AVG_CONN = data_file.NUM_AVG_CONN
NUM_AGENTS = data_file.NUM_AGENTS
SEQ_ACT = data_file.SEQ_ACT
MAX_SIM_TIME = data_file.MAX_SIM_TIME
HORIZON = data_file.HORIZON
REPLAN_INTERVAL = data_file.REPLAN_INTERVAL 
NUM_TEST_INSTANCES = data_file.NUM_TEST_INSTANCES
EVAL_INTERVAL = data_file.EVAL_INTERVAL
MPC_HORIZON = data_file.MPC_HORIZON
opt_val_dict = {}
min_yet = np.inf

init_bat = {}
max_bat = {}
for k in range(NUM_AGENTS):
  init_bat[k] = 15
  max_bat[k] = 15


save_path = f"./{NUM_NODES}N-{NUM_AGENTS}A-{MPC_HORIZON}T_G{graph_id}_MPC"
os.makedirs(f"{save_path}/test_instances/instance_{instance_id}", exist_ok=True)

def graph_opt_sol(prob_inst_id_and_G_copy_and_obj_lim):

  [prob_inst_id, G_copy, obj_lim, pol_n, p_sol] = prob_inst_id_and_G_copy_and_obj_lim

  nx.write_gpickle(G_copy, 
                   f"{save_path}/test_instances/instance_{prob_inst_id}/time_step_0.gpickle")

  eta = np.random.choice(np.asarray(list(range(1, 11))))

  for _ in range(MPC_HORIZON):

    agent_set, init_agent_occ_dict = graph_utils.get_agent_set_init_agent_dict(G_copy, NUM_NODES)


    agent_obj = {'list':agent_set, 'm_k_t':[0 for a in agent_set]}
    C = list(filter(lambda node: (G_copy.nodes[node]['station']), 
                    G_copy))

    x_hat = [[[0 for t in range(HORIZON+1)] for k in agent_set] 
             for i in G_copy]
    init_bat = {}
    max_bat = {}
    for k in agent_set:
      init_bat[k] = 15
      max_bat[k] = 15
      for t_ in range(HORIZON+1):
        x_hat[init_agent_occ_dict[k]][k][t_] = 1

    prev_node = init_agent_occ_dict[k]

    init_stn_dem = []

    for k in agent_set:
      init_stn_dem.append(G_copy.nodes[NUM_NODES + k]['stn_dem'])

    init_time = time.time()
    solved_obj = solve_main(G_copy, agent_obj, x_hat, init_bat, max_bat,
                            5, HORIZON, 1, 1, init_G=init_stn_dem,
                            opt_bound=obj_lim, par_s=p_sol)
    sol_time = (time.time() - init_time)

    # next_graph = G_copy

    x_sol = np.asarray([[[int(solved_obj.model.getVal(solved_obj.x[_i][_k][t_]) 
                              > 0.5) for t_ in range(HORIZON+1)] 
                              for _k in solved_obj.A] for _i in solved_obj.V])

    for node in list(G_copy.nodes())[:-1]:
      if not x_sol[node][0][0]:
        G_copy.nodes[node]['demand'] += 1
        G_copy.nodes[node]['num_occ_agents'] = 0
        G_copy.nodes[node]['num_agents_to_take_action'] = 0
        G_copy.nodes[node]['agents_occ'] = [-1]

      else:
        if node not in G_copy.neighbors(prev_node):
          print(ERROR)

        prev_node = node
        G_copy.nodes[node]['demand'] = 0
        G_copy.nodes[node]['num_occ_agents'] = 1
        G_copy.nodes[node]['num_agents_to_take_action'] = 1
        G_copy.nodes[node]['agents_occ'] = [0]
        G_copy.nodes[NUM_NODES]['agents_occ'] = [node]


    
    if (_ == eta) and (_+1 > 1):
      eta = eta + np.random.choice(np.asarray(list(range(1, 11))))
      G_copy = graph_utils.initiate_node_demands(G_copy, mpc_flag=True)

    nx.write_gpickle(G_copy, 
                   f"{save_path}/test_instances/instance_{prob_inst_id}/time_step_{_+1}.gpickle")



if __name__ == "__main__":

  try:

    G = nx.read_gpickle(f"{save_path}/test_instances/base_instance.gpickle")

  except:
    print(f"BASE INSTANCE NOT GENERATED!!!!!!!")
    exit()

  # G_c = graph_utils.initiate_node_demands(G)
  # G_c = graph_utils.agent_pos_random_reset(G_c, NUM_AGENTS)

  G_c = graph_utils.initiate_node_demands(G)
  G_c = graph_utils.agent_pos_random_reset(G_c, NUM_AGENTS)
  _, agent_occ_dict = graph_utils.get_agent_set_init_agent_dict(G_c, NUM_NODES)
  G_c = graph_utils.update_node_features(G_c, data_file.HORIZON, agent_occ_dict, None, None, None, True)

  o_lim = None
  pol_num = -1
  par_sol = None

  graph_opt_sol([instance_id, G_c, o_lim, pol_num, par_sol])


