# import math
# import random

import os
import sys

import datetime
import time

# import matplotlib
import matplotlib.pyplot as plt
# from collections import deque
# from itertools import count
import numpy as np
import copy
import networkx as nx
import csv

from multiprocessing import Pool

import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
from torch_geometric.utils.convert import from_networkx #, to_networkx
# from torch_geometric.nn import GCNConv

import graph_utils
import env_utils
from opt_solve import solve_main
import utils
import data_file
import main_run_opt


def eval_graph_pol(args):

  torch.set_num_threads(1)

  [graph_num, pol_num] = args

  opt_gap_list = []
  opt_gap_list_bs = []
  episode_loss = []

  BEAM_SEARCH = data_file.BEAM_SEARCH
  NUM_NODES = data_file.NUM_NODES
  NUM_AGENTS = data_file.NUM_AGENTS
  SEQ_ACT = data_file.SEQ_ACT
  if BEAM_SEARCH:
    BEAM_WIDTH = data_file.BEAM_WIDTH
  MAX_SIM_TIME = data_file.MAX_SIM_TIME
  HORIZON = data_file.HORIZON
  REPLAN_INTERVAL = data_file.REPLAN_INTERVAL 
  NUM_TEST_INSTANCES = data_file.NUM_TEST_INSTANCES
  EVAL_INTERVAL = data_file.EVAL_INTERVAL
  EVAL_FOLDER = data_file.EVAL_FOLDER
  opt_val_dict = {}
  init_dem_dict = {}
  min_yet = np.inf

  num_episodes = data_file.NUM_TRAIN_EPISODES

  list_of_iters_to_test = [0] # [(i_episode+1)*EVAL_INTERVAL*MAX_SIM_TIME \
                      # for i_episode in range(int(num_episodes/EVAL_INTERVAL))]

  # print(list_of_iters_to_test)

  save_path = f"./{NUM_NODES}N-{NUM_AGENTS}A-{MAX_SIM_TIME}T_G{graph_num}"

  os.makedirs(f"{save_path}/eval_data", exist_ok=True)

  with open(f"{save_path}/eval_data/eval_pol_{pol_num}_overall.csv",
                'a', newline='') as f:
    spamwriter = csv.writer(f, delimiter=',', quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['iter', 'avg_opt_gap', 'std-dev_opt_gap', 'avg_rl_sol_time'])

  if SEQ_ACT:
    from dqn_related_class_multiQout import DQN as Agent
  else:
    from ddpg_related_class import DDPG as Agent



  def update_sol(_nx_graph, partial_sol, time_):

    for node_ in _nx_graph.nodes():
      if not _nx_graph.nodes[node_]['agent']:
        partial_sol['agent_presence'][node_][0][time_] = \
                                      _nx_graph.nodes[node_]['num_occ_agents']
        partial_sol['demand'][node_][time_] = _nx_graph.nodes[node_]['demand']

    return partial_sol





  def evaluate_current_policy(next_graph_, dqn_agent_, next_agent_occ_dict_):

    sum_episode_demand = 0 # sum([next_graph_.nodes[node_]['demand'] * next_graph_.nodes[node_]['priority'] for node_ in next_graph_.nodes()])
    sum_action_time = 0

    par_sol = {}
    par_sol['agent_presence'] = {}
    par_sol['demand'] = {}
    for node_ in next_graph_.nodes():
      if not next_graph_.nodes[node_]['agent']:
        par_sol['agent_presence'][node_] = {}
        par_sol['agent_presence'][node_][0] = {}
        par_sol['demand'][node_] = {}
        for time_ in range(HORIZON+1):
          par_sol['agent_presence'][node_][0][time_] = None
          par_sol['demand'][node_][time_] = None

        

    plans = {}
    for agent_ in agent_set:
      plans[agent_] = {}

    par_sol = update_sol(next_graph_, par_sol, 0)
    
    if BEAM_SEARCH and SEQ_ACT:


      # num_neigh_of_station = len(next_graph_.neighbors( \
      #               graph_utils.get_station_node(next_graph_)[0]))

      init_graph_list = [next_graph_] # for _ in range(BEAM_WIDTH)]

      next_graph_nbs = copy.deepcopy(next_graph_)

      list_of_graphs_to_expand = copy.deepcopy(init_graph_list)
      list_of_occ_dicts = copy.deepcopy([next_agent_occ_dict_])

      temp_qvals = utils.get_all_qvals([next_graph_], dqn_agent_)
      max_k_qval_indices = utils.get_max_k_qval_indices(temp_qvals, BEAM_WIDTH)
      max_k_qval_indices = np.asarray([np.asarray([ind, index[1]]) \
                            for ind, index in enumerate(max_k_qval_indices)])
      all_qvals = np.asarray([temp_qvals[0] for _ in range(BEAM_WIDTH)])

      episode_demand_dict_prev = \
        {tuple(np.array([next_graph_.nodes[_]['feature'] \
          for _ in next_graph_.nodes()]).flatten()): sum( \
        [next_graph_.nodes[_]['demand'] for _ in next_graph_.nodes()])}

    for t__ in range(HORIZON):

      t_rem = HORIZON - t__

      current_plan_id = int(t__/HORIZON)
      plans[agent_][current_plan_id] = {}


      # print(f"time: {t__}")

      if BEAM_SEARCH and SEQ_ACT:

        episode_demand_dict_new = {}

        all_qvals = utils.get_all_qvals(list_of_graphs_to_expand, dqn_agent_)

        max_k_qval_indices = utils.get_max_k_qval_indices(all_qvals, BEAM_WIDTH)

        temp_graphs_list, temp_occ_dict_list, k_val_list = \
          utils.get_list_of_graphs_to_expand(list_of_graphs_to_expand, 
                                             list_of_occ_dicts,
                                             max_k_qval_indices)

        # temp_occ_dict_list = copy.deepcopy(list_of_occ_dicts)

        list_of_graphs_to_expand = []
        list_of_occ_dicts = []
        temp_index = 0

        # print(f"len(temp_graphs_list): {len(temp_graphs_list)}, \
        #   len(temp_occ_dict_list): {len(temp_occ_dict_list)}, \
        #   len(k_val_list): {len(k_val_list)}")

        # if t__ == 0:
          # init_graph = parent_child_relation(temp_graphs_list[-1])

        for temp_graph, occ_dict, k_val in zip(temp_graphs_list, \
                                             temp_occ_dict_list, k_val_list):
          # print(f"temp_index: {temp_index}")
        
          temp_graph_, temp_next_agent_occ_dict_, temp_demand_, \
            (_, _, _, _, _), _ = env_utils.env_step(temp_graph, t_rem, dqn_agent_,
                                                    occ_dict, explore=False,
                                                    beam_k_val=k_val, 
                                                    seq_flag=SEQ_ACT)


          temp_flag = False

          if len(list_of_graphs_to_expand) > 0:
            for gr in list_of_graphs_to_expand:
              if round(np.linalg.norm(np.array([gr.nodes[_]['feature'] \
                    for _ in gr.nodes()]).flatten() -\
                      np.array([temp_graph_.nodes[_]['feature'] \
                    for _ in temp_graph_.nodes()]).flatten()), 7) == 0: 
                   # nx.utils.graphs_equal(gr, temp_graph_):
                episode_demand_dict_new[tuple(np.array([gr.nodes[_]['feature'] \
                    for _ in gr.nodes()]).flatten())] = \
                min(episode_demand_dict_new[tuple(\
                  np.array([gr.nodes[_]['feature'] \
                    for _ in gr.nodes()]).flatten())], \
                episode_demand_dict_prev[tuple(\
                  np.array([temp_graph.nodes[_]['feature'] \
                    for _ in temp_graph.nodes()]).flatten())] + temp_demand_) 

                temp_flag = True
                break

            if temp_flag:
              # print(f"*****************")    
              temp_index += 1

            else:
              episode_demand_dict_new[tuple(\
                  np.array([temp_graph_.nodes[_]['feature'] \
                    for _ in temp_graph_.nodes()]).flatten())] = \
              episode_demand_dict_prev[tuple(\
                  np.array([temp_graph.nodes[_]['feature'] \
                    for _ in temp_graph.nodes()]).flatten())] + temp_demand_
              list_of_graphs_to_expand.append(temp_graph_)
              list_of_occ_dicts.append(temp_next_agent_occ_dict_)
              temp_index += 1

          else:
            episode_demand_dict_new[tuple(\
                  np.array([temp_graph_.nodes[_]['feature'] \
                    for _ in temp_graph_.nodes()]).flatten())] = \
            episode_demand_dict_prev[tuple(\
                  np.array([temp_graph.nodes[_]['feature'] \
                    for _ in temp_graph.nodes()]).flatten())] + temp_demand_

            list_of_graphs_to_expand.append(temp_graph_)
            list_of_occ_dicts.append(temp_next_agent_occ_dict_)

            temp_index += 1

        episode_demand_dict_prev = episode_demand_dict_new

        # print(f"len(list_of_graphs_to_expand): {len(list_of_graphs_to_expand)}")

        next_graph_nbs, next_agent_occ_dict_, demand_, (_, _, rew, _, _), plans, \
        action_duration = env_utils.env_step(next_graph_nbs, t_rem, dqn_agent_, 
                                             next_agent_occ_dict_, explore=False,
                                             seq_flag=SEQ_ACT)
            
        # demand_list.append(demand_)
        sum_episode_demand += -rew
        sum_action_time += action_duration

      else:

        if SEQ_ACT:

          next_graph_, next_agent_occ_dict_, demand_, (_, _, rew, _, _), plans,\
          action_duration = env_utils.env_step(next_graph_, plans, t__, 
                                               current_plan_id, t_rem, dqn_agent_,
                                               next_agent_occ_dict_, 
                                               explore=False, seq_flag=SEQ_ACT)

        else:

          next_graph_, action_duration, demand_, (_, _, _, _) = env_utils.env_step(\
            next_graph_, t_rem, dqn_agent_, None, explore=False, seq_flag=SEQ_ACT)
              
        # demand_list.append(demand_)
        sum_episode_demand += -sum(rew)
        sum_action_time += action_duration

        par_sol = update_sol(next_graph_, par_sol, t__+1)

    if BEAM_SEARCH:
      sum_episode_demand_bs = min([episode_demand_dict_prev[_] \
       for _ in list(episode_demand_dict_prev.keys())])

      sum_episode_demand_list = [sum_episode_demand_bs, sum_episode_demand]

    if BEAM_SEARCH:
      return sum_episode_demand_list, sum_action_time
    else:
      return sum_episode_demand, sum_action_time, par_sol


  time_for_opt_computation = {} # f"NOT_COMPUTED"

  graph_read_path = f"{NUM_NODES}N-{NUM_AGENTS}A-{MAX_SIM_TIME}T_G{graph_num}"
  # print(f"{graph_read_path}")

  G = nx.read_gpickle(f"{graph_read_path}/test_instances/instance_1.gpickle")

  agent_set, init_agent_occ_dict = graph_utils.get_agent_set_init_agent_dict(G, NUM_NODES)

  for i_ in range(1, NUM_TEST_INSTANCES+1):
    temp_graph_obj = nx.read_gpickle(f"{graph_read_path}/test_instances/instance_{i_}.gpickle")
    temp_opt_sol_time = None
    with open(f"{graph_read_path}/test_instances/opt_sol_instance_{i_}.csv") as f:
      reader = csv.reader(f)
      for row_ind, row in enumerate(reader):
        if row[0] == "obj":
          temp_val = sum([(temp_graph_obj.nodes[node_]['demand'] * \
                           temp_graph_obj.nodes[node_]['priority'])
                           for node_ in temp_graph_obj.nodes()]) + \
                           temp_graph_obj.nodes[NUM_NODES]['stn_dem']
          init_dem_dict[i_] = temp_val
          opt_val_dict[i_] = float(row[1]) #+ init_dem_dict[i_]
          # print(opt_val_dict[i_])

        if row[0] == "sol_time":
          time_for_opt_computation[i_] = float(row[1])

        if (i_ in opt_val_dict.keys()) and \
                                (i_ in time_for_opt_computation.keys()):
          break

    # sum_opt_obj_vals += temp_graph_obj
  avg_opt_sol_time = sum([time_for_opt_computation[i_] \
                  for i_ in time_for_opt_computation.keys()])/NUM_TEST_INSTANCES

  G_copy = graph_utils.initiate_node_demands(G)
  G_copy = graph_utils.update_node_features(G_copy, HORIZON,
                                            init_agent_occ_dict,
                                            None, None, None, True)
  pyg = from_networkx(G_copy)


  if SEQ_ACT:
    dqn_agent = Agent(G_copy, pyg, NUM_AGENTS=len(agent_set))
    
  else:
    dqn_agent = Agent(G_copy, NUM_AGENTS=len(agent_set))

  for iter_ in list_of_iters_to_test:

    # buff = torch.load(f"./trained_pols/{NUM_NODES}N-{NUM_AGENTS}A-{MAX_SIM_TIME}T_G{graph_num}/pol_{pol_num}/pol_iter_{iter_}.pt")
    # # print(buff)
    # dqn_agent.policy_net.load_state_dict(buff['state'])

    with open(f"{save_path}/eval_data/eval_pol_{pol_num}_iter_{iter_}.csv",
                'a', newline='') as f:
      spamwriter = csv.writer(f, delimiter=',', quotechar='|',
                              quoting=csv.QUOTE_MINIMAL)
      spamwriter.writerow(['opt_obj', 'RL_pol_obj'])


    rl_test_sol_obj = {}
    time_for_rl_sol_computation = {}
    # sum_rl_test_sol_obj = 0
    sum_rl_test_sol_obj_beam_search = 0
    # time_for_rl_sol_computation = 0
    time_for_rl_sol_computation_beamsearch = 0
    for i_ in range(1, NUM_TEST_INSTANCES+1):
      eval_graph = nx.read_gpickle(
                    f"{graph_read_path}/test_instances/instance_{i_}.gpickle")
      _, agent_occ_dict = graph_utils.get_agent_set_init_agent_dict(eval_graph,
                                                                    NUM_NODES)
      eval_graph = graph_utils.update_node_features(eval_graph, \
        HORIZON, agent_occ_dict, None, None, None, True)
      next_agent_occ_dict = copy.deepcopy(agent_occ_dict)
      print(f"testing_instance: {i_}\tgraph: {graph_num}\tpolicy: {pol_num}\titer: {iter_}", end="\r")

      if BEAM_SEARCH and SEQ_ACT:

        init_time = time.time()
        episode_dem_temp_list, act_time = evaluate_current_policy(eval_graph, \
                                                dqn_agent, next_agent_occ_dict)
        time_for_rl_sol_computation_beamsearch += \
                            (time.time() - init_time)/NUM_TEST_INSTANCES
        time_for_rl_sol_computation[i_] = act_time

        rl_test_sol_obj[i_] = episode_dem_temp_list[1]
        sum_rl_test_sol_obj_beam_search += episode_dem_temp_list[0]

      else:
        episode_dem_temp, act_time, partial_solution = evaluate_current_policy(\
                                    eval_graph, dqn_agent, next_agent_occ_dict)
        rl_test_sol_obj[i_] = float(episode_dem_temp) #+ init_dem_dict[i_]

        time_for_rl_sol_computation[i_] = act_time

        # main_run_opt.graph_opt_sol([i_, eval_graph,
        #                             rl_test_sol_obj[i_], pol_num,
        #                             partial_solution])

      with open(f"{save_path}/eval_data/eval_pol_{pol_num}_iter_{iter_}.csv",
                'a', newline='') as f:
        spamwriter = csv.writer(f, delimiter=',', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow([f'{opt_val_dict[i_]}', 
                             f'{float(episode_dem_temp)}'])
    if NUM_TEST_INSTANCES > 0:

      if BEAM_SEARCH and SEQ_ACT:
        avg_rl_opt_gap = sum([100*((rl_test_sol_obj[_]/opt_val_dict[_]) - 1)\
          for _ in range(1, NUM_TEST_INSTANCES+1)]) / NUM_TEST_INSTANCES
        opt_gap_list.append(avg_rl_opt_gap)
        sum_opt_val = sum([opt_val_dict[_] for _ in range(1, NUM_TEST_INSTANCES+1)])
        opt_gap_list_bs.append(100*\
          ((sum_rl_test_sol_obj_beam_search/sum_opt_val) - 1))

        # print(f"\n------------------------\ntime_for_opt_computation: \
        # {avg_opt_sol_time}s\t time_for greedy_rl_sol_computation: \
        # {time_for_rl_sol_computation}s\nGreedy OPT_GAP: {opt_gap_list[-1]}%\n \
        # time_for beam-search_rl_sol_computation: \
        # {time_for_rl_sol_computation_beamsearch}s\n \
        # Beam-search OPT_GAP: {opt_gap_list_bs[-1]}%\n \
        # ------------------------\n")

      else:
        avg_rl_opt_gap = sum([100*((rl_test_sol_obj[_]/opt_val_dict[_]) - 1)\
         for _ in range(1, NUM_TEST_INSTANCES+1)]) / NUM_TEST_INSTANCES

        stddev_opt_gap = np.std(np.asarray([100*((rl_test_sol_obj[_]/opt_val_dict[_]) - 1)\
         for _ in range(1, NUM_TEST_INSTANCES+1)]))

        avg_rl_sol_time = sum([time_for_rl_sol_computation[_]\
         for _ in range(1, NUM_TEST_INSTANCES+1)]) / NUM_TEST_INSTANCES

        opt_gap_list.append(avg_rl_opt_gap)

        with open(f"{save_path}/eval_data/eval_pol_{pol_num}_overall.csv",
                'a', newline='') as f:
          spamwriter = csv.writer(f, delimiter=',', quotechar='|',
                                  quoting=csv.QUOTE_MINIMAL)
          spamwriter.writerow([f'{iter_}', f'{avg_rl_opt_gap}', f'{stddev_opt_gap}',
                               f'{avg_rl_sol_time}'])


        for _ in range(1, NUM_TEST_INSTANCES+1):
          with open(f"{save_path}/eval_data/comp_time_log.csv",
                'a', newline='') as f:
            spamwriter = csv.writer(f, delimiter=',', quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow([f'{time_for_rl_sol_computation[_]}', f'{time_for_opt_computation[_]}']) 

        # print(f"\n------------------------\ntime_for_opt_computation: \
        # {avg_opt_sol_time}s\t time_for_rl_sol_computation: \
        # {avg_rl_sol_time}s\nOPT_GAP: {opt_gap_list[-1]}%\n \
        # ------------------------\n")
  # eval

  print('Complete')




if __name__ == "__main__":

  args_list = []

  g_num = str(sys.argv[1])
  p_num = str(sys.argv[2])

  args_list.append([g_num, p_num])
  eval_graph_pol(args_list[-1])

  #for g_num in range(2, data_file.NUM_GRAPHS+1):
  #  for p_num in range(1, data_file.NUM_TRAIN_POL+1):
  #    args_list.append([g_num, p_num])
  #    eval_graph_pol(args_list[-1])
      #exit()


  #pool = Pool(10)

  #pool.map(eval_graph_pol, args_list)

