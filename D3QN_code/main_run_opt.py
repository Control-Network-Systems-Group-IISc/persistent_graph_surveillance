"""
Example doc string
"""

import os
import sys
import time
import copy

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
opt_val_dict = {}
min_yet = np.inf

init_bat = {}
max_bat = {}
for k in range(NUM_AGENTS):
  init_bat[k] = HORIZON
  max_bat[k] = HORIZON


save_path = f"./{NUM_NODES}N-{NUM_AGENTS}A-{MAX_SIM_TIME}T_G{graph_id}"
os.makedirs(f"{save_path}/test_instances", exist_ok=True)

def graph_opt_sol(prob_inst_id_and_G_copy_and_obj_lim):

  [prob_inst_id, G_copy, G_multi, obj_lim, pol_n, p_sol] = prob_inst_id_and_G_copy_and_obj_lim

  agent_set, init_agent_occ_dict = graph_utils.get_agent_set_init_agent_dict(G_copy, NUM_NODES)

  nx.write_gpickle(G_copy, \
           f"{save_path}/test_instances/instance_{prob_inst_id}_single.gpickle")

  nx.write_gpickle(G_multi, \
            f"{save_path}/test_instances/instance_{prob_inst_id}_multi.gpickle")

  agent_obj = {'list':agent_set, 'm_k_t':[0 for a in agent_set]}
  # C = list(filter(lambda node: (G_copy.nodes[node]['station']), G_copy))

  x_hat = [[[0 for t in range(HORIZON+1)] for k in agent_set] 
           for i in G_copy]
  # init_bat = {}
  # max_bat = {}
  for k in agent_set:
    # init_bat[k] = HORIZON
    # max_bat[k] = HORIZON
    for t_ in range(HORIZON+1):
      x_hat[init_agent_occ_dict[k]][k][t_] = 1

  init_stn_dem = []

  for k in agent_set:
    init_stn_dem.append(G_multi.nodes[NUM_NODES + k]['stn_dem'])

  init_time = time.time()
  solved_obj = solve_main(G_copy, agent_obj, x_hat, init_bat, max_bat,
                          5, HORIZON, 1, 1, init_G=init_stn_dem,
                          opt_bound=obj_lim, par_s=p_sol)
  sol_time = (time.time() - init_time)

  x_sol = np.asarray([[[int(solved_obj.model.getVal(solved_obj.x[_i][_k][t_]) 
                            > 0.5) for t_ in range(HORIZON+1)] 
                            for _k in solved_obj.A] for _i in solved_obj.V])

  obj_val = solved_obj.model.getVal(solved_obj.obj)

  opt_val_dict[prob_inst_id] = obj_val

  if obj_lim is None:

    with open(\
      f"{save_path}/test_instances/opt_sol_instance_{prob_inst_id}.csv", 'a',
          newline='') as f:
      spamwriter = csv.writer(f, delimiter=',', quotechar='|',
                              quoting=csv.QUOTE_MINIMAL)
      spamwriter.writerow(['obj', opt_val_dict[prob_inst_id]])
      spamwriter.writerow(['sol_time', sol_time])
      spamwriter.writerow(['time', 'agent', 'node'])
      for _t in range(HORIZON+1):
        for k in solved_obj.A:
          for _i in solved_obj.V:
            if x_sol[_i][k][_t] > 0.5:
              spamwriter.writerow([_t+1, k, _i])

  else:

    with open(\
    f"{save_path}/eval_data/opt_sol_instance_pol_{pol_n}_inst_{prob_inst_id}.csv",
    'a', newline='') as f:
      spamwriter = csv.writer(f, delimiter=',', quotechar='|',
                              quoting=csv.QUOTE_MINIMAL)
      spamwriter.writerow(['obj', opt_val_dict[prob_inst_id]])
      spamwriter.writerow(['sol_time', sol_time])
      spamwriter.writerow(['time', 'agent', 'node'])
      for _t in range(HORIZON+1):
        for k in solved_obj.A:
          for _i in solved_obj.V:
            if x_sol[_i][k][_t] > 0.5:
              spamwriter.writerow([_t+1, k, _i])

if __name__ == "__main__":

  try:

    # G = nx.read_gpickle(\
    #             f"{save_path}/test_instances/base_instance_single_step.gpickle")

    G_ = nx.read_gpickle(\
                f"{save_path}/test_instances/base_instance_multi_step.gpickle")

  except:
    print(f"BASE INSTANCE NOT GENERATED!!!!!!!")
    exit()

  # G_c = graph_utils.initiate_node_demands(G)
  # G_c = graph_utils.agent_pos_random_reset(G_c, NUM_AGENTS)

  G_c_ = graph_utils.initiate_node_demands(G_)
  G_c_ = graph_utils.agent_pos_random_reset(G_c_, NUM_AGENTS)
  _, agent_occ_dict = graph_utils.get_agent_set_init_agent_dict(G_c_, NUM_NODES)
  G_c_ = graph_utils.update_node_features(G_c_, data_file.HORIZON,
                                          agent_occ_dict, None, None, None,
                                          True)

  G_c = graph_utils.convert_to_single_time_step(copy.deepcopy(G_c_))

  # print(f"num_nodes: {len(G_c.nodes())}")


  # exit()

  o_lim = None
  pol_num = -1
  par_sol = None

  graph_opt_sol([instance_id, G_c, G_c_, o_lim, pol_num, par_sol])


