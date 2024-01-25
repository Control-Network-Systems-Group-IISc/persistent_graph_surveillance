"""Example Google style docstrings.
"""

import os
import numpy as np
import networkx as nx
from networkx.classes.function import path_weight
import random
# from torch_geometric.utils.convert import from_networkx, to_networkx
from matplotlib import pyplot as plt
import random
from itertools import combinations, groupby
import copy

import data_file

def gen_graph_and_agents(num_nodes, num_stations, num_connects_per_node,
                         prob_of_rewiring, num_agents, max_charge, init_charge):

  """Example Google style docstrings.
  """

  graph_ = nx.connected_watts_strogatz_graph(
      n=num_nodes, k=num_connects_per_node, p=prob_of_rewiring)
  # graph_ = gnp_random_connected_graph(
  #     n=num_nodes, p=prob_of_rewiring)
  # graph = nx.hexagonal_lattice_graph(10, 7)

  agent_set = range(num_agents)
  
  for edge in graph_.edges():
    graph_[edge[0]][edge[1]]['travel_time'] = np.random.choice(range(1, 4))

  for node in graph_.nodes():
    if (node, node) not in graph_.edges():
      graph_.add_edge(node, node, travel_time=1)

  valid_station_list_flag = False

  while not valid_station_list_flag:
    station_nodes_list = np.random.choice(list(graph_.nodes())[:num_nodes],
                                          size=num_stations, replace=False)
    if num_stations == 1:
      valid_station_list_flag = True
      break

    else:
      temp_var = 0
      for ind_, node in enumerate(station_nodes_list):
        if ind_ == 0:
          temp_list = copy.deepcopy(station_nodes_list[1:])
        elif ind_ == num_stations - 1:
          temp_list = copy.deepcopy(station_nodes_list[:-1])
        else:
          temp_list = copy.deepcopy(station_nodes_list)[:ind_]
          temp_list = temp_list + copy.deepcopy(station_nodes_list[ind_+1:])

        for node_ in graph_.neighbors(node):
          if node_ in temp_list:
            temp_var += 1
            break

        if temp_var:
          valid_station_list_flag = False
          break

      if temp_var:
        valid_station_list_flag = False

      else:
        valid_station_list_flag = True


  graph__ = copy.deepcopy(graph_)

  graph__.add_nodes_from([(len(list(graph__.nodes())) + _) for _ in agent_set])


  nx.set_node_attributes(graph__, False, "station")
  nx.set_node_attributes(graph__, False, "agent")
  # nx.set_node_attributes(graph, [], "agents_feat")
  nx.set_node_attributes(graph__, np.double(0), "demand")
  nx.set_node_attributes(graph__, 0, "priority")
  nx.set_node_attributes(graph__, 0, "num_occ_agents")
  nx.set_node_attributes(graph__, 0, "num_agents_to_take_action")
  nx.set_node_attributes(graph__, [-1], "agents_occ")
  nx.set_node_attributes(graph__, -1, "agents_plan_id")
  nx.set_node_attributes(graph__, -1, "agents_time_since_station_visit")
  nx.set_node_attributes(graph__, False, "active")
  nx.set_node_attributes(graph__, -1, "max_charge")
  nx.set_node_attributes(graph__, 0, "true_node")
  nx.set_node_attributes(graph__, 0, "stn_dem")


  # print(f"list of edges: {graph.edges()}")

  for node in graph__.nodes():
    graph__.nodes[node]['priority'] = 1
    graph__.nodes[node]['true_node'] = 1

  for station_node in station_nodes_list:
    graph__.nodes[station_node]['station'] = True
    # graph__.nodes[station_node]['agents_occ'] = []
    graph__.nodes[station_node]['num_occ_agents'] = num_agents
    graph__.nodes[station_node]['num_agents_to_take_action'] = num_agents

  num_nodes__ = len(list(graph__.nodes())) - num_agents

  for agent_node in range(num_nodes__, num_nodes__+num_agents):
    graph__.nodes[agent_node]['agent'] = True
    # graph__.nodes[agent_node]['agents_occ'] = [] # station_nodes_list[0]
    graph__.nodes[agent_node]['agents_plan_id'] = 0
    graph__.nodes[agent_node]['active'] = True
    graph__.nodes[agent_node]['max_charge'] = max_charge[agent_node-num_nodes__]
    graph__.nodes[agent_node]['agents_time_since_station_visit'] = 0
    graph__.nodes[agent_node]['true_node'] = 0

    
  graph__ = initiate_node_demands(graph__)
  graph__ = agent_pos_random_reset(graph__, num_agents)
  _, agent_occ_dict = get_agent_set_init_agent_dict(graph__, num_nodes)
  graph__ = update_node_features(graph__, 0, agent_occ_dict, None, None, None, 
                                 True)


  # graph = copy.deepcopy(graph__)
  
  '''graph = nx.Graph()

  graph.add_nodes_from(list(graph__.nodes()))

  for edge in graph__.edges():
    temp_num_extra_nodes = graph__[edge[0]][edge[1]]['travel_time']
    
    if temp_num_extra_nodes == 1:
      graph.add_edge(edge[0], edge[1])

    else:
      temp_list = [edge[0]]
      for _ in range(temp_num_extra_nodes-1):
        temp_list.append(len(list(graph.nodes())))
        graph.add_node(temp_list[-1])
      temp_list.append(edge[1])
      # graph.add_node(temp_list[-1])

      for n_id in range(len(temp_list)-1):
        graph.add_edge(temp_list[n_id], temp_list[n_id+1])

  for node in graph__.nodes():
    if (node, node) not in graph.edges():
      graph.add_edge(node, node)

  for edge in graph.edges():
    graph[edge[0]][edge[1]]['travel_time'] = 1 # np.random.choice(range(1, 5))

  graph.add_nodes_from([len(list(graph.nodes())) + _ for _ in agent_set])

  nx.set_node_attributes(graph, False, "station")
  nx.set_node_attributes(graph, False, "agent")
  # nx.set_node_attributes(graph, [], "agents_feat")
  nx.set_node_attributes(graph, np.double(0), "demand")
  nx.set_node_attributes(graph, 0, "priority")
  nx.set_node_attributes(graph, 0, "num_occ_agents")
  nx.set_node_attributes(graph, 0, "num_agents_to_take_action")
  nx.set_node_attributes(graph, [-1], "agents_occ")
  nx.set_node_attributes(graph, -1, "agents_plan_id")
  nx.set_node_attributes(graph, -1, "agents_time_since_station_visit")
  nx.set_node_attributes(graph, False, "active")
  nx.set_node_attributes(graph, -1, "max_charge")
  nx.set_node_attributes(graph, 0, "true_node")
  nx.set_node_attributes(graph, 0, "stn_dem")


  # print(f"list of edges: {graph.edges()}")

  for node in graph_.nodes():
    graph.nodes[node]['priority'] = 1
    graph.nodes[node]['true_node'] = 1

  for station_node in station_nodes_list:
    graph.nodes[station_node]['station'] = True
    # graph.nodes[station_node]['agents_occ'] = []
    graph.nodes[station_node]['num_occ_agents'] = num_agents
    graph.nodes[station_node]['num_agents_to_take_action'] = num_agents

  num_nodes__ = len(list(graph__.nodes())) - num_agents

  for agent_node in range(num_nodes__, num_nodes__+num_agents):
    graph.nodes[agent_node]['agent'] = True
    # graph.nodes[agent_node]['agents_occ'] = [] # station_nodes_list[0]
    graph.nodes[agent_node]['agents_plan_id'] = 0
    graph.nodes[agent_node]['active'] = True
    graph.nodes[agent_node]['max_charge'] = max_charge[agent_node-num_nodes__]
    graph.nodes[agent_node]['agents_time_since_station_visit'] = 0
    graph.nodes[node]['true_node'] = 0


  graph = initiate_node_demands(graph)
  graph = agent_pos_random_reset(graph, num_agents)
  _, agent_occ_dict = get_agent_set_init_agent_dict(graph, num_nodes)
  graph = update_node_features(graph, data_file.HORIZON, agent_occ_dict, None, 
                               None, None, True)'''

  return graph__, agent_occ_dict, agent_set#, agent_feat_dict



def convert_to_single_time_step(graph__):

  graph = nx.Graph()

  graph.add_nodes_from(list(graph__.nodes()))

  graph.remove_nodes_from([_ for _ in range(\
              data_file.NUM_NODES, data_file.NUM_NODES + data_file.NUM_AGENTS)])

  for node in graph.nodes():
    graph.nodes[node]['station'] = graph__.nodes[node]['station']
    graph.nodes[node]['agent'] = graph__.nodes[node]['agent']
    graph.nodes[node]['demand'] = graph__.nodes[node]['demand']
    graph.nodes[node]['priority'] = graph__.nodes[node]['priority']
    graph.nodes[node]['num_occ_agents'] = graph__.nodes[node]['num_occ_agents']
    graph.nodes[node]['num_agents_to_take_action'] = \
                                graph__.nodes[node]['num_agents_to_take_action']
    graph.nodes[node]['agents_occ'] = graph__.nodes[node]['agents_occ']
    graph.nodes[node]['agents_plan_id'] = graph__.nodes[node]['agents_plan_id']
    graph.nodes[node]['agents_time_since_station_visit'] = \
                          graph__.nodes[node]['agents_time_since_station_visit']
    graph.nodes[node]['active'] = graph__.nodes[node]['active']
    graph.nodes[node]['max_charge'] = graph__.nodes[node]['max_charge']
    graph.nodes[node]['true_node'] = graph__.nodes[node]['true_node']
    graph.nodes[node]['stn_dem'] = graph__.nodes[node]['stn_dem']

  # graph.add_nodes_from(list(graph__.nodes()))

  for edge in graph__.edges():
    temp_num_extra_nodes = graph__[edge[0]][edge[1]]['travel_time']
    
    if temp_num_extra_nodes == 1:
      graph.add_edge(edge[0], edge[1])

    else:
      temp_list = [edge[0]]
      for _ in range(temp_num_extra_nodes-1):
        temp_list.append(len(list(graph.nodes())))
        graph.add_nodes_from([(temp_list[-1], \
                            {\
                              'station': False,
                              'agent' : False,
                              'demand' : np.double(0),
                              'priority' : 0,
                              'num_occ_agents': 0,
                              'num_agents_to_take_action': 0,
                              'agents_occ' : [-1],
                              'agents_plan_id' : -1,
                              'agents_time_since_station_visit' : -1,
                              'active' : False,
                              'max_charge' : -1,
                              'true_node' : False,
                              'stn_dem' : 0
                            })])

      temp_list.append(edge[1])
      # graph.add_node(temp_list[-1])

      for n_id in range(len(temp_list)-1):
        graph.add_edge(temp_list[n_id], temp_list[n_id+1])

  for node in graph__.nodes():
    if (node, node) not in graph.edges():
      graph.add_edge(node, node)

  for edge in graph.edges():
    graph[edge[0]][edge[1]]['travel_time'] = 1 # np.random.choice(range(1, 5))

  for _ in range(data_file.NUM_NODES, \
                 data_file.NUM_NODES + data_file.NUM_AGENTS):
    graph.add_nodes_from([(len(list(graph.nodes())), \
                            {\
                              'station': False,
                              'agent' : False,
                              'demand' : np.double(0),
                              'priority' : 0,
                              'num_occ_agents': 0,
                              'num_agents_to_take_action': 0,
                              'agents_occ' : [-1],
                              'agents_plan_id' : -1,
                              'agents_time_since_station_visit' : -1,
                              'active' : False,
                              'max_charge' : -1,
                              'true_node' : False,
                              'stn_dem' : 0
                            })])

    last_node_index = len(graph.nodes()) - 1

    graph.nodes[last_node_index]['agent'] = graph__.nodes[_]['agent']
    graph.nodes[last_node_index]['agents_occ'] = graph__.nodes[_]['agents_occ']
    graph.nodes[last_node_index]['agents_plan_id'] = \
                                              graph__.nodes[_]['agents_plan_id']
    graph.nodes[last_node_index]['active'] = graph__.nodes[_]['active']
    graph.nodes[last_node_index]['max_charge'] = graph__.nodes[_]['max_charge']
    graph.nodes[last_node_index]['agents_time_since_station_visit'] = \
                             graph__.nodes[_]['agents_time_since_station_visit']
    graph.nodes[last_node_index]['true_node'] = graph__.nodes[_]['true_node']

  num_nodes = len(list(graph.nodes())) - data_file.NUM_AGENTS

  _, agent_occ_dict = get_agent_set_init_agent_dict(graph, num_nodes)
  graph = update_node_features(graph, 0, agent_occ_dict, None, None, None, True)

  # print(f"num_nodes: {len(graph.nodes())}")

  # graph.add_nodes_from([len(list(graph.nodes())) + _ for _ in agent_set])

  # nx.set_node_attributes(graph, False, "station")
  # nx.set_node_attributes(graph, False, "agent")
  # # nx.set_node_attributes(graph, [], "agents_feat")
  # nx.set_node_attributes(graph, np.double(0), "demand")
  # nx.set_node_attributes(graph, 0, "priority")
  # nx.set_node_attributes(graph, 0, "num_occ_agents")
  # nx.set_node_attributes(graph, 0, "num_agents_to_take_action")
  # nx.set_node_attributes(graph, [-1], "agents_occ")
  # nx.set_node_attributes(graph, -1, "agents_plan_id")
  # nx.set_node_attributes(graph, -1, "agents_time_since_station_visit")
  # nx.set_node_attributes(graph, False, "active")
  # nx.set_node_attributes(graph, -1, "max_charge")
  # nx.set_node_attributes(graph, 0, "true_node")
  # nx.set_node_attributes(graph, 0, "stn_dem")


  # print(f"list of edges: {graph.edges()}")

  # for node in graph__.nodes():
  #   graph.nodes[node]['station'] = graph__.nodes[node]['station']
  #   graph.nodes[node]['agent'] = graph__.nodes[node]['agent']
  #   graph.nodes[node]['demand'] = graph__.nodes[node]['demand']
  #   graph.nodes[node]['priority'] = graph__.nodes[node]['priority']
  #   graph.nodes[node]['num_occ_agents'] = graph__.nodes[node]['num_occ_agents']
  #   graph.nodes[node]['num_agents_to_take_action'] = \
  #                               graph__.nodes[node]['num_agents_to_take_action']
  #   graph.nodes[node]['agents_occ'] = graph__.nodes[node]['agents_occ']
  #   graph.nodes[node]['agents_plan_id'] = graph__.nodes[node]['agents_plan_id']
  #   graph.nodes[node]['agents_time_since_station_visit'] = \
  #                         graph__.nodes[node]['agents_time_since_station_visit']
  #   graph.nodes[node]['active'] = graph__.nodes[node]['active']
  #   graph.nodes[node]['max_charge'] = graph__.nodes[node]['max_charge']
  #   graph.nodes[node]['true_node'] = graph__.nodes[node]['true_node']
  #   graph.nodes[node]['stn_dem'] = graph__.nodes[node]['stn_dem']

  return graph



def agent_pos_random_reset(graph, num_agents):

  for node in graph.nodes():
    graph.nodes[node]["agents_occ"] = [-1]
    graph.nodes[node]["num_occ_agents"] = 0
    graph.nodes[node]["num_agents_to_take_action"] = 0

  for agent_ in range(num_agents):
    
    new_pos_node = np.random.choice(list(graph.nodes())[:-num_agents])

    for node in graph.nodes():
      if node == new_pos_node:
        if (len(graph.nodes[node]["agents_occ"]) == 1) and \
                                     (graph.nodes[node]["agents_occ"][0] == -1):
          graph.nodes[node]["agents_occ"] = [agent_]
        else:
          graph.nodes[node]["agents_occ"] = graph.nodes[node]["agents_occ"] +\
                                                                        [agent_]
        graph.nodes[node]["num_occ_agents"] += 1
        graph.nodes[node]["num_agents_to_take_action"] += 1

      if graph.nodes[node]["agent"]:
        graph.nodes[node]["agents_occ"] = [new_pos_node]
        graph.nodes[node]["num_occ_agents"] = 0
        graph.nodes[node]["num_agents_to_take_action"] = 0

        if graph.nodes[new_pos_node]['station']:
          graph.nodes[node]['stn_dem'] = 0
        else:
          stn_node = get_station_node(graph)
          min_steps = min([nx.shortest_path_length(graph, source=_, \
                                      target=new_pos_node) for _ in stn_node])
          graph.nodes[node]['stn_dem'] = np.random.choice(list(range(\
                                         min_steps, (3*data_file.HORIZON) + 1)))
       
  return graph


def update_node_features(graph, time_left, agent_occ_dict, updated_node_demands,
                         updated_node_num_occ_agents_occ,
                         updated_node_num_occ_agents_to_take_action, 
                         final_update):

  """
  Example Google style docstrings.
  """

  num_nodes_ = sum([1 if (not graph.nodes[_]['agent']) else 0 
                    for _ in graph.nodes()])

  for node in graph.nodes():
    graph.nodes[node]["agents_occ"] = [-1]

  for agent_id in agent_occ_dict.keys():
    graph.nodes[num_nodes_ + agent_id]['agents_occ'] = [agent_occ_dict[agent_id]]
    # print(f"### : {agent_occ_dict[agent_id]}")
    # print(f"**** : {graph.nodes[agent_occ_dict[agent_id]]['agents_occ']}")
    if graph.nodes[agent_occ_dict[agent_id]]['agents_occ'] == [-1]:
      graph.nodes[agent_occ_dict[agent_id]]['agents_occ'] = [agent_id]

    else:
      graph.nodes[agent_occ_dict[agent_id]]['agents_occ'] = \
        graph.nodes[agent_occ_dict[agent_id]]['agents_occ'] + [agent_id]


  if ('feature' in graph.nodes[0]) and updated_node_demands is not None:
    stn_node = get_station_node(graph)
    for node_num, node in enumerate(graph.nodes()):
      if not graph.nodes[node]['agent']:
        graph.nodes[node]['demand'] = updated_node_demands[node]
        graph.nodes[node]['num_occ_agents'] = updated_node_num_occ_agents_occ[\
                                                                      node]
        graph.nodes[node]['num_agents_to_take_action'] = \
           updated_node_num_occ_agents_to_take_action[node]

        if final_update and graph.nodes[node]['num_occ_agents'] == -1:
          graph.nodes[node]['num_occ_agents'] = 0

        graph.nodes[node]['feature'] = np.asarray(\
            [graph.nodes[node]['demand'],
             graph.nodes[node]['priority'],
             time_left \
             # graph.nodes[node]['num_agents_to_take_action']
            ] \
            + \
            [graph.nodes[num_nodes_+_]['stn_dem'] for _ in agent_occ_dict.keys()]
            + \
            [(1 - graph.nodes[node]['station']) * \
             min(path_weight(graph,
                             list(nx.shortest_path(graph, source=_, 
                                                   target=node)),
                             weight='travel_time') for _ in stn_node)]
            +
            [graph.nodes[node]['num_occ_agents']]
            +
            [path_weight(graph,
                         list(nx.shortest_path(graph, 
                                               source=agent_occ_dict[agent_id],
                                               target=node)),
                         weight='travel_time')
             for agent_id in agent_occ_dict.keys()]\
            )

  else:
    num_agn = sum([graph.nodes[_]['num_occ_agents'] for _ in graph.nodes()])
    stn_node = get_station_node(graph)

    # print(f"stn_node: {stn_node}, num_nodes: {len(graph.nodes())}")

    for node in graph.nodes():
      if not graph.nodes[node]['agent']:
        if final_update and graph.nodes[node]['num_occ_agents'] == -1:
          graph.nodes[node]['num_occ_agents'] = 0
        graph.nodes[node]['feature'] = np.asarray(\
          [graph.nodes[node]['demand'], graph.nodes[node]['priority'], \
           time_left
           # graph.nodes[node]['num_occ_agents'], \
           # graph.nodes[node]['num_agents_to_take_action']\
           \
          ] \
          + \
          [graph.nodes[num_nodes_+_]['stn_dem'] for _ in agent_occ_dict.keys()]
          + \
          [(1 - graph.nodes[node]['station']) * \
           min(path_weight(graph,
                           list(nx.shortest_path(graph, source=_, 
                                                 target=node)),
                           weight='travel_time') for _ in stn_node)]
          +
          [graph.nodes[node]['num_occ_agents']]
          + \
          [path_weight(graph,
                       list(nx.shortest_path(graph, 
                                             source=agent_occ_dict[agent_id],
                                             target=node)),
                       weight='travel_time')
           for agent_id in agent_occ_dict.keys()]
          )


  for node in graph.nodes():
    if graph.nodes[node]['agent']:
      graph.nodes[node]['feature'] = np.asarray(\
                                        [-1 for _ in graph.nodes[0]['feature']])
    else: # if not graph.nodes[node]['agent']:
      ...
      # graph.nodes[node]['feature'] = np.concatenate(\
      #   (np.asarray(nx.adjacency_matrix(graph).todense()[node,
      #                                                    :])[0][:num_nodes_], 
      #    graph.nodes[node]['feature']))

      # graph.nodes[node]['feature'] = np.concatenate(\
      #   (np.asarray(nx.attr_matrix(graph, edge_attr='travel_time', 
      #                              rc_order=list(range(data_file.NUM_NODES + \
      #                              data_file.NUM_AGENTS)))[node, \
      #                                                    :])[0][:num_nodes_], 
      #    graph.nodes[node]['feature']))

  if ('agent_feature' in graph.nodes[0]) and updated_node_demands is not None:
    for node in graph.nodes():
      if not graph.nodes[node]['agent']:
        graph.nodes[node]['agent_feature'] = np.asarray([-1 \
                        for _ in graph.nodes[0]['feature']] + [-1, -1, -1, -1])

      else:
        if final_update:
          graph.nodes[node]['active'] = True

        _agent_id = node - num_nodes_
        _agent_occ_node = agent_occ_dict[_agent_id]
        graph.nodes[node]['agent_feature'] = np.concatenate(\
         (graph.nodes[_agent_occ_node]['feature'],\
         # np.asarray([graph.nodes[node]['agents_plan_id']]), \
         # np.asarray([graph.nodes[node]['agents_time_since_station_visit']]), \
          np.asarray([graph.nodes[node]['stn_dem']]), \
          np.asarray([time_left]), \
          np.asarray(graph.nodes[node]['agents_occ']), \
          np.asarray([graph.nodes[node]['active']])))

  else:
    for node in graph.nodes():
      if not graph.nodes[node]['agent']:
        graph.nodes[node]['agent_feature'] = np.asarray(\
                     [-1 for _ in graph.nodes[0]['feature']] + [-1, -1, -1, -1])
      else:
        if final_update:
          graph.nodes[node]['active'] = True

        _agent_id = node - num_nodes_
        _agent_occ_node = agent_occ_dict[_agent_id]
        # print(f"@@@{_agent_id}, {_agent_occ_node}")
        graph.nodes[node]['agent_feature'] = np.concatenate(\
          (np.asarray(graph.nodes[_agent_occ_node]['feature']), \
          # np.asarray([0]), np.asarray([0]), \
           np.asarray([graph.nodes[node]['stn_dem']]), \
           np.asarray([data_file.HORIZON]), \
           np.asarray(graph.nodes[node]['agents_occ']), \
           np.asarray([graph.nodes[node]['active']])))

  # print(f"agent: {graph.nodes[num_nodes_]['agents_occ']}, node: {graph.nodes[graph.nodes[num_nodes_]['agents_occ'][0]]['agents_occ']}")

  return graph



def initiate_node_demands(graph, zero_dem_flag=False):

  temp_list_nodes = copy.deepcopy(list(graph.nodes())[data_file.NUM_NODES:])

  temp_list = []

  for node in temp_list_nodes:
    if not graph.nodes[node]['station']:
      temp_list.append(node)

  random.shuffle(temp_list)

  # zero_temp_list = temp_list[:data_file.num_nodes_zero_priority]
  high_temp_list = temp_list[:data_file.num_nodes_high_priority] 
  # low_temp_list = temp_list[data_file.num_nodes_high_priority:]

  if zero_dem_flag:
    for node in graph.nodes():
      if (not graph.nodes[node]['station']) and \
          (graph.nodes[node]['true_node']): # and (node not in zero_temp_list):

        if node in high_temp_list:
          graph.nodes[node]['demand'] = 0 # np.random.choice(10)
          graph.nodes[node]['priority'] = np.random.choice(list(range(5, 8)))

        else:
          graph.nodes[node]['demand'] = 0 # np.random.choice(10)
          graph.nodes[node]['priority'] = np.random.choice(list(range(1, 3)))
      
      else:
        graph.nodes[node]['demand'] = 0
        graph.nodes[node]['priority'] = 0

  else:
    for node in graph.nodes():
      if not (graph.nodes[node]['station'] or graph.nodes[node]['agent']) and \
          (graph.nodes[node]['true_node']): # and (node not in zero_temp_list):


        if node in high_temp_list:
          graph.nodes[node]['demand'] = np.random.choice(list(range(10, 20)))\
                                                          # np.random.choice(10)
          graph.nodes[node]['priority'] = np.random.choice(list(range(5, 8)))

        else:
          graph.nodes[node]['demand'] = np.random.choice(list(range(1, 5))) \
                                                          # np.random.choice(10)
          graph.nodes[node]['priority'] = np.random.choice(list(range(1, 3)))
        
        # graph.nodes[node]['demand'] = np.random.choice(list(\
                                                        # range(201, 301)))#,\
        #     # p=[7/33, 5/33, 4/33, 3/33, 1/33, 1/33, 3/33, 3/33, 3/33, 3/33])
        # graph.nodes[node]['priority'] = np.random.choice(list(range(1, 4)))
        #(node%9)
        # np.random.choice(list(range(1, 7)),\
                                    # p=[5/18, 3/18, 1/18, 1/18, 3/18, 5/18])
      else:
        graph.nodes[node]['demand'] = 0
        graph.nodes[node]['priority'] = 0

  return graph

def findPaths(G, u, n):
  if n == 0:
    return [[u]]

  paths = [[u]+path for neighbor in G.neighbors(u) \
           for path in findPaths(G, neighbor, n-1)] \
           # if u not in path]
  return paths

def plot_graph_dem_and_agent_pos(nx_graph, train_iter, time, path, flag=True,
                                 opt_dem=0, sol_dem=0):

  # nx_graph = to_networkx(pyg_graph)

  plt.clf()

  if flag:
    save_path = f"{path}" #/train_iter_{train_iter}"

  else:
    save_path = f"{path}/train_iter_{train_iter}"


  # label_dict_edges = {}
  label_dict_nodes = {}

  # for edge in nx_graph.edges():
  #   label_dict_edges[edge] = f"e:{nx_graph.edges[edge]['travel_time']}"

  for node in nx_graph.nodes():
    if nx_graph.nodes[node]['station']:
      label_dict_nodes[node] \
                           = f"{node}:a{nx_graph.nodes[node]['num_occ_agents']}"

    else:
      label_dict_nodes[node] = f"{node}:{nx_graph.nodes[node]['demand']}"


  pos = nx.kamada_kawai_layout(nx_graph)#, seed=3113794652)
  options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 0.9}
  nx.draw_networkx_nodes(nx_graph, pos, nodelist=nx_graph.nodes(),
                         node_color="tab:orange", **options)
  nx.draw_networkx_nodes(nx_graph, pos, nodelist=get_station_node(nx_graph),
                         node_color="tab:gray", **options)
  nx.draw_networkx_nodes(nx_graph, pos, nodelist=get_agent_occ_nodes(nx_graph),
                         node_color="tab:green", **options)

  nx.draw_networkx_labels(nx_graph, pos, labels=label_dict_nodes)
  nx.draw_networkx_edges(nx_graph, pos)
  # plt.axis('off')
  os.makedirs(save_path, exist_ok=True)
  sum_temp = sum([nx_graph.nodes[_]['demand'] for _ in nx_graph.nodes()])
  title = f"time_{time}_"
  title += f"demand_{sum_temp}, "
  title += f"opt_dem_{round(opt_dem, 1)}, sol_dem_{round(sol_dem, 1)}"
  plt.title(f"{title}")
  plt.tight_layout()

  if flag:
    plt.savefig(f"{save_path}/time_{time}_action_agent_{train_iter}.png",
                dpi=600)

  else:
    plt.savefig(f"{save_path}/time_{time}.png", dpi=600)


def get_station_node(graph):

  station_nodes = []

  for node in graph.nodes():
    if graph.nodes[node]['station']:
      station_nodes.append(node)
      # break

  return station_nodes

def get_agent_occ_nodes(graph):
  node_list = []
  for node in graph.nodes():
    if graph.nodes[node]['num_occ_agents']:
      node_list.append(node)

  return node_list


def get_agent_set_init_agent_dict(graph, num_nodes__):

  num_agents = sum([graph.nodes[node]['num_occ_agents'] \
                    for node in graph.nodes()])

  # print(f"num_agents: {num_agents}")

  num_nodes_ = len(list(graph.nodes())) - num_agents

  # print(f"num_nodes: {num_nodes_}")

  agent_set = list(range(num_agents))

  agent_occ_dict = {}

  for agent in agent_set:
    try:
      agent_occ_dict[agent] = graph.nodes[num_nodes_+agent]['agents_occ'][0]
    except IndexError:
      agent_occ_dict[agent] = graph.nodes[num_nodes_+agent]['agents_occ']

  return list(agent_set), agent_occ_dict



def gnp_random_connected_graph(n, p):
  """
  Generates a random undirected graph, similarly to an Erdős-Rényi 
  graph, but enforcing that the resulting graph is conneted
  """
  edges = combinations(range(n), 2)
  G = nx.Graph()
  G.add_nodes_from(range(n))
  if p <= 0:
    return G
  if p >= 1:
    return nx.complete_graph(n, create_using=G)
  for _, node_edges in groupby(edges, key=lambda x: x[0]):
    node_edges = list(node_edges)
    random_edge = random.choice(node_edges)
    G.add_edge(*random_edge)
    for e in node_edges:
      if random.random() < p:
        G.add_edge(*e)
  return G
