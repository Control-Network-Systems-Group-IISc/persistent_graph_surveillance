"""
Example doc string
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import copy

# from multiprocessing import Pool
# from torch_geometric.utils.convert import from_networkx #, to_networkx

import graph_utils
import data_file


graph_id = int(sys.argv[1])

NUM_NODES = data_file.NUM_NODES
NUM_AVG_CONN = data_file.NUM_AVG_CONN
NUM_AGENTS = data_file.NUM_AGENTS
SEQ_ACT = data_file.SEQ_ACT
MAX_SIM_TIME = data_file.MAX_SIM_TIME
HORIZON = data_file.HORIZON
REPLAN_INTERVAL = data_file.REPLAN_INTERVAL 
NUM_TEST_INSTANCES = data_file.NUM_TEST_INSTANCES
EVAL_INTERVAL = data_file.EVAL_INTERVAL
NUM_STATIONS = data_file.NUM_STATIONS
opt_val_dict = {}
min_yet = np.inf

init_bat = {}
max_bat = {}
for k in range(NUM_AGENTS):
  init_bat[k] = HORIZON
  max_bat[k] = HORIZON


save_path = f"./{NUM_NODES}N-{NUM_AGENTS}A-{MAX_SIM_TIME}T_G{graph_id}"
os.makedirs(f"{save_path}/test_instances", exist_ok=True)

try:

  G = nx.read_gpickle(\
                f"{save_path}/test_instances/base_instance_single_step.gpickle")
  exit()

except:
  G__, init_agent_occ_dict, _ = graph_utils.gen_graph_and_agents( \
  NUM_NODES, NUM_STATIONS, NUM_AVG_CONN, 0.5, NUM_AGENTS, max_bat, init_bat)
  
  nx.write_gpickle(G__, \
                 f"{save_path}/test_instances/base_instance_multi_step.gpickle")

  G = graph_utils.convert_to_single_time_step(copy.deepcopy(G__))

  nx.write_gpickle(G, \
                f"{save_path}/test_instances/base_instance_single_step.gpickle")



  label_dict_edges = {}
  label_dict_nodes = {}

  G_plot = G__.subgraph(list(range(NUM_NODES)))

  for edge in G_plot.edges():
    label_dict_edges[edge] = f"{G_plot.edges[edge]['travel_time']}"

  for node in G_plot.nodes():
    label_dict_nodes[node] = f"{node}:{int(G_plot.nodes[node]['station'])}"

  pos = nx.kamada_kawai_layout(G_plot)   
  nx.draw(G_plot, pos, labels=label_dict_nodes, with_labels=True)
  nx.draw_networkx_edge_labels(G, pos, rotate=False, font_size=8, 
                               edge_labels=label_dict_edges)
  plt.savefig(f"{save_path}/plt.png", dpi=600)


# stn_node = graph_utils.get_station_node(G)

# paths = graph_utils.findPaths(G, stn_node, MAX_SIM_TIME)
