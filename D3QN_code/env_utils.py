
import copy
from torch_geometric.utils.convert import from_networkx
import torch
import time
import networkx as nx
from torch_geometric.data import Batch


import graph_utils
import data_file

def env_step(nx_graph, _plans, curr_time, current_plan_id, time_left,
             dqn_agent, agent_occ_dict, explore=True, beam_k_val=None,
             seq_flag=False):


  num_non_agent_nodes = sum([1 if (not nx_graph.nodes[_]['agent']) \
                                else 0 \
                                for _ in nx_graph.nodes()])
  number_of_agents = len(list(nx_graph.nodes())) - num_non_agent_nodes  

  if seq_flag:
    print_ = False
    temp_action_duration = 0
    charge_penalty = 0
    station_visit_penalty = 0
    list_of_to_nodes = []

    list_of_from_nodes = []

    is_this_max_q_val_list = []
    
    number_of_agents = len(list(agent_occ_dict.keys()))

    temp_agent_occ_dict = copy.deepcopy(agent_occ_dict)

    temp_node_occ_list = copy.deepcopy(custom_sort(get_node_occ_list(\
                                                          temp_agent_occ_dict)))
    temp_node_occ_list.reverse()
    prev_graph = copy.deepcopy(nx_graph)
    temp_nx_graph = copy.deepcopy(nx_graph)
    state_feat_list = []
    action_list = []
    reward_list = []
    next_state_feat_list = []
    temp_pyg_graph = from_networkx(temp_nx_graph)

    if current_plan_id not in list(_plans[0].keys()):
      for _ in range(number_of_agents):
        _plans[_][current_plan_id] = {}

    for _a_ in range(number_of_agents):
      # print(f"agent: {_}")
      state_feat_list.append(temp_pyg_graph)
      action_init_time = time.time()
      action_, (agent_id, from_node, to_node), num_total_action = \
                                dqn_agent.select_action(temp_nx_graph, 
                                                        state_feat_list[-1],
                                                        explore=explore,
                                                        k_val=beam_k_val)
      temp_action_duration += time.time() - action_init_time

      if data_file.IS_BATTERY_DYNAMICS:
        if min([nx.shortest_path_length(\
                                      temp_nx_graph, source=to_node, target=c_)
                if temp_nx_graph.nodes[c_]['station'] else num_non_agent_nodes
                for c_ in temp_nx_graph.nodes()]) > \
           (temp_nx_graph.nodes[num_non_agent_nodes + agent_id]['max_charge'] -\
            temp_nx_graph.nodes[num_non_agent_nodes + agent_id][\
                                        'agents_time_since_station_visit'] - 1):
          
          path_length = {}
          min_path_len = num_non_agent_nodes
          for c_ in temp_nx_graph.nodes():
            if temp_nx_graph.nodes[c_]['station']:
              path_length[c_] = min(list(nx.shortest_path_length(\
                                temp_nx_graph, 
                                source=list(temp_nx_graph.neighbors(from_node)),
                                target=c_)))
              min_path_len = min(min_path_len, path_length[c_])
              if min_path_len != path_length[c_]:
                continue
              for neigh in list(temp_nx_graph.neighbors(from_node)):
                neigh_path_len = nx.shortest_path_length(temp_nx_graph,\
                                                        source=neigh, target=c_)
                if min_path_len == neigh_path_len:
                  charge_override_to_node = nx.shortest_path(temp_nx_graph, 
                                                             source=neigh,
                                                             target=c_)[0]

          to_node = charge_override_to_node

          charge_penalty += data_file.CHARGE_PENALTY_PER_TIMESTEP

      _plans[agent_id][current_plan_id][curr_time] = to_node
      
      action_list.append(action_)

      is_this_max_q_val = [False for u_ in range(num_total_action)]
      is_this_max_q_val[int(action_)] = True
      is_this_max_q_val_list.append(torch.tensor(is_this_max_q_val))

      is_override, override_to_node, next_plan_id =  prev_plan_override(\
            temp_nx_graph, _plans, curr_time, current_plan_id, agent_id, \
            from_node, to_node, temp_agent_occ_dict)

      temp_nx_graph.nodes[num_non_agent_nodes+agent_id]['agents_plan_id'] =\
                                                                    next_plan_id

      temp_nx_graph.nodes[num_non_agent_nodes+agent_id]['active'] = False      

      if temp_nx_graph.nodes[override_to_node]['station']:
        temp_nx_graph.nodes[\
            num_non_agent_nodes+agent_id]['agents_time_since_station_visit'] = 0

        temp_nx_graph.nodes[\
           num_non_agent_nodes+agent_id]['stn_dem'] = 0

      else:
        temp_nx_graph.nodes[\
           num_non_agent_nodes+agent_id]['agents_time_since_station_visit'] += 1

        temp_nx_graph.nodes[\
           num_non_agent_nodes+agent_id]['stn_dem'] += 1
      
      # temp_nx_graph.nodes[\
      #      num_non_agent_nodes+agent_id]['stn_dem'] = temp_nx_graph.nodes[\
      #      num_non_agent_nodes+agent_id]['agents_time_since_station_visit']

      station_visit_penalty += temp_nx_graph.nodes[\
                                        num_non_agent_nodes+agent_id]['stn_dem']

      temp_nx_graph.nodes[\
                  num_non_agent_nodes+agent_id]['agents_occ'] = override_to_node



      list_of_to_nodes.append(int(override_to_node))
      list_of_from_nodes.append(int(from_node))

      temp_agent_occ_dict[agent_id] = int(override_to_node)

      updated_node_demands, updated_node_num_occ_agents_occ, \
      updated_num_agents_to_take_action = get_node_demands_and_num_agents(
          temp_nx_graph, int(from_node), int(override_to_node))

      if print_:
        ...
        # print(f"calculated update before update: {[updated_node_demands[node] \
        # for node in temp_nx_graph]}")

        # print(f"\n============================================\n")

        # print(f"\nfrom_node: {int(from_node)}\tto_node: {int(to_node)}")
        # print(f"neighbours of from node: {list(temp_nx_graph.neighbors( \
        # int(from_node)))}")
        # print(f"\tdemand\tnum of occ agents\tnum of active agents")
        # print(f"\nFROM NODE BEFORE UPDATE:")
        # print(f"\t{temp_nx_graph.nodes[int(from_node)]['demand']}\t{temp_nx_graph.nodes[int(from_node)]['num_occ_agents']}\t{temp_nx_graph.nodes[int(from_node)]['num_agents_to_take_action']}")
        # print(f"\nTO NODE BEFORE UPDATE:")
        # print(f"\t{temp_nx_graph.nodes[int(to_node)]['demand']}\t{temp_nx_graph.nodes[int(to_node)]['num_occ_agents']}\t{temp_nx_graph.nodes[int(to_node)]['num_agents_to_take_action']}")


      temp_nx_graph = graph_utils.update_node_features(temp_nx_graph, \
        time_left, temp_agent_occ_dict, \
        updated_node_demands, updated_node_num_occ_agents_occ, \
        updated_num_agents_to_take_action, False)



      if print_:
        ...
        # print(f"\nFROM NODE AFTER UPDATE:")
        # print(f"\t{temp_nx_graph.nodes[int(from_node)]['demand']}\t{temp_nx_graph.nodes[int(from_node)]['num_occ_agents']}\t{temp_nx_graph.nodes[int(from_node)]['num_agents_to_take_action']}")
        # print(f"\nTO NODE AFTER UPDATE:")
        # print(f"\t{temp_nx_graph.nodes[int(to_node)]['demand']}\t{temp_nx_graph.nodes[int(to_node)]['num_occ_agents']}\t{temp_nx_graph.nodes[int(to_node)]['num_agents_to_take_action']}\n")

        # print(f"\n============================================\n")

        # print(f"updated_node_demands before update: {[temp_nx_graph.nodes[node]['demand'] for node in temp_nx_graph]}\n")

        # print(f"****************************************************************")

      temp_pyg_graph = from_networkx(temp_nx_graph)
      next_state_feat_list.append(temp_pyg_graph)

      # _, (_, _, _), _ = dqn_agent.select_action(temp_nx_graph, 
      #                                            next_state_feat_list[-1], 
      #                                            explore=explore,
      #                                            k_val=beam_k_val)

    updated_node_demand_dict, updated_node_num_agents_to_take_action = \
      update_final_demands_and_num_agents_to_take_action(temp_nx_graph, \
      list_of_from_nodes, list_of_to_nodes)

    num_time_step_skip = temp_nx_graph[list_of_from_nodes[0]][list_of_to_nodes[0]]['travel_time']

    temp_nx_graph = graph_utils.update_node_features(temp_nx_graph, \
      time_left-num_time_step_skip, temp_agent_occ_dict, \
      updated_node_demand_dict, updated_node_num_occ_agents_occ, \
      updated_node_num_agents_to_take_action, True)

    temp_pyg_graph = from_networkx(temp_nx_graph)

    next_state_feat_list[-1] = temp_pyg_graph

    # _, (_, _, _), _ = dqn_agent.select_action(temp_nx_graph, 
    #                                           next_state_feat_list[-1], 
    #                                           explore=explore,
    #                                           k_val=beam_k_val)

    full_demand = get_final_demand(prev_graph, temp_nx_graph, time_left, list_of_from_nodes[0], list_of_to_nodes[0])

    # print(f"time_left: {time_left}, dqn_agent.policy_net: {dqn_agent.policy_net(Batch.from_data_list([temp_pyg_graph])).clone().detach().squeeze(0)}")

    for _ in range(number_of_agents):
      reward_list.append(\
                 torch.tensor([(-full_demand)/number_of_agents])) #  - charge_penalty - station_visit_penalty - station_visit_penalty

    if print_:
      ...
      print(f"====================================\n")

    return temp_nx_graph, temp_agent_occ_dict, full_demand, (state_feat_list,\
           action_list, reward_list, next_state_feat_list, \
           is_this_max_q_val_list), _plans, temp_action_duration

  else:
    # print(f"here!!!!!")
    temp_nx_graph = copy.deepcopy(nx_graph)
    temp_pyg_graph = from_networkx(temp_nx_graph)

    temp_act_init_time = time.time()
    # action, from_node_list, to_nodes_list = dqn_agent.select_joint_action(\
    #                                                     temp_pyg_graph, explore)
    act_duration = time.time() - temp_act_init_time
    # action = 1


    to_nodes_list = []
    from_node_list = [temp_nx_graph.nodes[len(temp_nx_graph.nodes)-1]["agents_occ"][0]]
    # print(from_node_list)
    for f_node in from_node_list:
      temp_node = None
      temp_val = None
      for neigh_node in temp_nx_graph.neighbors(f_node):
        if temp_val is None:
          temp_val = temp_nx_graph.nodes[neigh_node]['priority'] * \
                                (temp_nx_graph.nodes[neigh_node]['demand'] + 1)
          temp_node = neigh_node
        else:
          if temp_val < (temp_nx_graph.nodes[neigh_node]['priority'] *\
                              (temp_nx_graph.nodes[neigh_node]['demand'] + 1)):
            temp_val = temp_nx_graph.nodes[neigh_node]['priority'] * \
                                (temp_nx_graph.nodes[neigh_node]['demand'] + 1)
            temp_node = neigh_node 
      to_nodes_list.append(temp_node)


    next_nx_graph = get_updated_graph(temp_nx_graph, to_nodes_list)
    agent_occ_dict = {}
    agent_occ_dict[0] = to_nodes_list[0]
    next_nx_graph = graph_utils.update_node_features(next_nx_graph, time_left-1,
                                                     agent_occ_dict, None, None,
                                                     None, final_update=True)   

    demand = get_graph_demand(next_nx_graph)

    reward = [-demand]

    temp_next_pyg_graph = from_networkx(next_nx_graph)

    # print(f"================================================================")

    # print(f"from nodes: {from_node_list}")
    # print(f"to_nodes: {to_nodes_list}")

    # for f_node, t_node in zip(from_node_list, to_nodes_list):
    #   print(f"neighbours of node {f_node} are \
    #           {list(next_nx_graph.neighbors(f_node))} and the to_node is {t_node}")

    #   # print(f"demand of from node: {next_nx_graph.nodes[f_node]['demand']}")
    #   print(f"old priority*demand of to node: {temp_nx_graph.nodes[t_node]['priority']*temp_nx_graph.nodes[t_node]['demand']}")

    #   # print(f"new demand of to node: {next_nx_graph.nodes[t_node]['demand']}")

    # print(f"updated demand: {demand}")

    # print(f"================================================================")


    return next_nx_graph, act_duration, demand, (temp_pyg_graph, action, 
                                                 reward, temp_next_pyg_graph)


def get_final_demand(pre_G, this_G, t_left, f_node, t_node):

  final_dem = 0

  num_time_step_skip = this_G[f_node][t_node]['travel_time']

  if num_time_step_skip > 1:
    for _ in range(1, num_time_step_skip):
      if t_left - _ >= 0:
        final_dem += sum([pre_G.nodes[n_]['priority'] * (pre_G.nodes[n_]['demand'] + _) for n_ in pre_G.nodes()])

  if t_left - num_time_step_skip >= 0:
    final_dem += sum([this_G.nodes[n_]['priority'] * \
                         (this_G.nodes[n_]['demand']) for n_ in this_G.nodes()])

  return final_dem


def prev_plan_override(nx_graph, __plans, _curr_time, curr_plan_id, _agent_id,
                       _from_node, _to_node, agent_occ_dict):

  
  num_non_agent_nodes = sum([1 if (not nx_graph.nodes[_]['agent'])\
                             else 0 \
                             for _ in nx_graph.nodes()])
  next_plan_id = nx_graph.nodes[\
                              num_non_agent_nodes + _agent_id]['agents_plan_id']
  if nx_graph.nodes[\
            num_non_agent_nodes + _agent_id]['agents_plan_id'] == curr_plan_id:
    _is_this_override = False
    _override_to_node = _to_node

  else:
    _is_this_override = True
    if _curr_time in __plans[_agent_id][nx_graph.nodes[\
                            num_non_agent_nodes + _agent_id]['agents_plan_id']]:
      _override_to_node = __plans[_agent_id][nx_graph.nodes[\
                 num_non_agent_nodes + _agent_id]['agents_plan_id']][_curr_time]

    else:
      _override_to_node = _from_node


  if nx_graph.nodes[_override_to_node]['station']:
    next_plan_id = curr_plan_id

  else:
    is_better_plan, next_plan_id = node_has_agent_with_better_plan(\
             nx_graph, _agent_id, _to_node, agent_occ_dict, num_non_agent_nodes)

  return _is_this_override, _override_to_node, next_plan_id


def node_has_agent_with_better_plan(_nx_graph, __agent_id, __to_node,
                                    _agent_occ_dict, num_non_agent_nodes):
  _node = __to_node
  _is_better_plan = False
  temp_plan_id = _nx_graph.nodes[\
                               num_non_agent_nodes+__agent_id]['agents_plan_id']
  _next_plan_id = temp_plan_id
  for _agent in range(len(_nx_graph.nodes()) - num_non_agent_nodes):
  # for _agent in _nx_graph.nodes[num_non_agent_nodes]['agents_occ']:
    if _agent != __agent_id:
      if _nx_graph.nodes[num_non_agent_nodes+_agent]['agents_occ'] == \
                  _nx_graph.nodes[num_non_agent_nodes+__agent_id]['agents_occ']:
        _next_plan_id = max(_next_plan_id, _nx_graph.nodes[\
                                  num_non_agent_nodes+_agent]['agents_plan_id'])
        _is_better_plan = True

  return _is_better_plan, _next_plan_id


  
def get_updated_graph(nx_graph, to_nodes):

  updated_graph = copy.deepcopy(nx_graph)
  updated_graph = wipe_agent_occ_data_and_increment_dem(updated_graph)
  updated_graph = update_agent_occ_and_correct_demand(updated_graph, to_nodes)

  return updated_graph



def update_agent_occ_and_correct_demand(nx_graph, to_nodes_list):

  for node in to_nodes_list:
    nx_graph.nodes[node]['num_occ_agents'] += 1
    nx_graph.nodes[node]['num_agents_to_take_action'] += 1
    nx_graph.nodes[node]['demand'] = 0

  return nx_graph



def wipe_agent_occ_data_and_increment_dem(nx_graph):

  for node in nx_graph.nodes():
    nx_graph.nodes[node]['num_occ_agents'] = 0
    nx_graph.nodes[node]['num_agents_to_take_action'] = 0
    nx_graph.nodes[node]['demand'] += 1

  return nx_graph


def get_node_demands_and_num_agents(graph, from_node, to_node):

  updated_node_demands = {}
  updated_node_num_occ_agents = {}
  updated_num_agents_to_take_action = {}

  num_time_step_skip = graph[from_node][to_node]['travel_time']
  
  for node in graph.nodes():
    updated_node_demands[node] = graph.nodes[node]['demand']
    updated_node_num_occ_agents[node] = graph.nodes[node]['num_occ_agents']
    updated_num_agents_to_take_action[node] = graph.nodes[node][\
                                                    'num_agents_to_take_action']

  # print(f"\n============================================\n")

  # print(f"\nfrom_node: {from_node}\tto_node: {to_node}")
  # print(f"neighbours of from node: {list(graph.neighbors(from_node))}")
  # print(f"\tdemand\tnum of occ agents\tnum of active agents")
  # print(f"\nFROM NODE BEFORE UPDATE:")
  # print(f"\t{updated_node_demands[from_node]}\t{updated_node_num_occ \
  # _agents[from_node]}\t{updated_num_agents_to_take_action[from_node]}")
  # print(f"\nTO NODE BEFORE UPDATE:")
  # print(f"\t{updated_node_demands[to_node]}\t{ \ 
  # updated_node_num_occ_agents[to_node]}\t{ \ 
  # updated_num_agents_to_take_action[to_node]}")
    

  if int(from_node) != int(to_node):
    updated_node_demands[from_node] = \
                         num_time_step_skip * demand_increment(graph, from_node)
    updated_node_num_occ_agents[from_node] -= 1

    updated_node_demands[to_node] = 0
    updated_node_num_occ_agents[to_node] += 1

  updated_num_agents_to_take_action[from_node] -= 1
  
  for node in graph.nodes():
    if updated_node_num_occ_agents[node]:
      updated_node_demands[node] = 0

  # print(f"neighbours of node {from_node} are {\
  # graph[int(from_node)]}. updated ")

  # print(f"\nFROM NODE AFTER UPDATE:")
  # print(f"\t{updated_node_demands[from_node]}\t{\
  # updated_node_num_occ_agents[from_node]}\t{\
  # updated_num_agents_to_take_action[from_node]}")
  # print(f"\nTO NODE AFTER UPDATE:")
  # print(f"\t{updated_node_demands[to_node]}\t{\
  # updated_node_num_occ_agents[to_node]}\t{\
  # updated_num_agents_to_take_action[to_node]}\n")

  # print(f"\n============================================\n")

  return updated_node_demands, updated_node_num_occ_agents,\
  updated_num_agents_to_take_action



# Function to efficiently sort a list with many duplicated values
# using the counting sort algorithm
def custom_sort(a_list):
  # create a new list to store counts of elements in the input list
  freq = {}
  for i in a_list:
    freq[i] = 0
  # using the value of elements in the input list as an index,
  # update their frequencies in the new list
  for i in a_list:
    freq[i] = freq[i] + 1
  # overwrite the input list with sorted order
  k = 0
  sorted_list_of_keys = list(freq.keys())
  sorted_list_of_keys.sort()
  for i in sorted_list_of_keys:
    while freq[i] > 0:
      a_list[k] = i
      freq[i] = freq[i] - 1
      k = k + 1
  return a_list


def demand_increment(nx_graph, node):
  if (nx_graph.nodes[node]['station'] or \
                        nx_graph.nodes[node]['agent']):
    return 0
  else:
    return 1


def get_node_occ_list(agent_occ_dict):
  node_occ_list = []

  for agent in list(agent_occ_dict.keys()):
    node_occ_list.append(agent_occ_dict[agent])
  return node_occ_list


def get_agent_in_node(agent_occ_dict, node):
  
  for agent in list(agent_occ_dict.keys()):
    if node == agent_occ_dict[agent]:
      return agent

  return None



def check_for_multiple_agents_in_node(agent_occ_dict, from_node):
  num_agents_in_node = 0
  for agent in list(agent_occ_dict.keys()):
    if agent_occ_dict[agent] == from_node:
      num_agents_in_node += 1
  return num_agents_in_node


def get_graph_demand(nx_graph):
  
  dem = 0

  for node in nx_graph.nodes():
    if not nx_graph.nodes[node]['agent']:
      dem += (nx_graph.nodes[node]['priority'] * nx_graph.nodes[node]['demand'])

  return dem


def update_final_demands_and_num_agents_to_take_action(nx_graph, from_nodes,
                                                       to_nodes):

  updated_node_demands = {}
  updated_node_num_agents_to_take_action = {}

  num_time_step_skip = \
                     nx_graph[from_nodes[0]][to_nodes[0]]['travel_time']

  for node in nx_graph.nodes():
    updated_node_demands[node] = nx_graph.nodes[node]['demand']
    updated_node_num_agents_to_take_action[node] = nx_graph.nodes[node][\
                                                               'num_occ_agents']
    if (node in from_nodes) or (node in to_nodes):
      continue

    else:
      updated_node_demands[node] = nx_graph.nodes[node]['demand'] \
                                   + (num_time_step_skip * demand_increment(nx_graph, node))

    if node in to_nodes:
      updated_node_demands[node] = 0

  return updated_node_demands, updated_node_num_agents_to_take_action
