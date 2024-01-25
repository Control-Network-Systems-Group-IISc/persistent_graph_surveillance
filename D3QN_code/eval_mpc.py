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
  NUM_TEST_INSTANCES = 5 # data_file.NUM_TEST_INSTANCES
  EVAL_INTERVAL = data_file.EVAL_INTERVAL
  EVAL_FOLDER = data_file.EVAL_FOLDER
  MPC_HORIZON = data_file.MPC_HORIZON
  opt_val_dict = {}
  init_dem_dict = {}
  min_yet = np.inf

  num_episodes = data_file.NUM_TRAIN_EPISODES

  list_of_iters_to_test = [300000]

  save_path = f"./{NUM_NODES}N-{NUM_AGENTS}A-{150}T_G{graph_num}_MPC"

  os.makedirs(f"{save_path}/eval_data/", exist_ok=True)

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


  def run_MPC(instance, next_graph_, dqn_agent_, next_agent_occ_dict_):

    # for node in next_graph_.nodes():
    #   if next_graph_.nodes[NUM_NODES]['agents_occ'] == node:
    #     next_graph_.nodes[node]['demand'] = 0
    #   else:
    #     next_graph_.nodes[node]['demand'] += 1

    saved_graph_read_path = f"{save_path}/test_instances/instance_{instance}"

    all_dem = 0

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

      init_graph_list = [next_graph_]

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

    for t__ in range(MPC_HORIZON):

      # print(f"t__: {t__}")

      if MPC_HORIZON - t__ <= HORIZON:
        t_rem = MPC_HORIZON - t__

      else:
        t_rem = HORIZON

      G = nx.read_gpickle(f"{saved_graph_read_path}/time_step_{t__}.gpickle")

      for node in next_graph_.nodes():
        next_graph_.nodes[node]['priority'] = G.nodes[node]['priority']


      next_graph_ = graph_utils.update_node_features(next_graph_, t_rem, next_agent_occ_dict_, None, None, None, True)

      prev_g = next_graph_
      
      current_plan_id = int(t__/HORIZON)
      # plans[agent_][current_plan_id] = {}

      if BEAM_SEARCH and SEQ_ACT:

        episode_demand_dict_new = {}

        all_qvals = utils.get_all_qvals(list_of_graphs_to_expand, dqn_agent_)

        max_k_qval_indices = utils.get_max_k_qval_indices(all_qvals, BEAM_WIDTH)

        temp_graphs_list, temp_occ_dict_list, k_val_list = \
          utils.get_list_of_graphs_to_expand(list_of_graphs_to_expand, 
                                             list_of_occ_dicts,
                                             max_k_qval_indices)

        list_of_graphs_to_expand = []
        list_of_occ_dicts = []
        temp_index = 0

        for temp_graph, occ_dict, k_val in zip(temp_graphs_list, \
                                             temp_occ_dict_list, k_val_list):
        
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

        next_graph_nbs, next_agent_occ_dict_, demand_, (_, _, rew, _, _), plans, \
        action_duration = env_utils.env_step(next_graph_nbs, t_rem, dqn_agent_, 
                                             next_agent_occ_dict_,
                                             explore=False,
                                             seq_flag=SEQ_ACT)
        sum_episode_demand += -rew
        sum_action_time += action_duration

      else:

        if SEQ_ACT:

          next_graph_, next_agent_occ_dict_, demand_, (_, _, rew, _, _), plans,\
          action_duration = env_utils.env_step(next_graph_, plans, t__, 
                                               current_plan_id, t_rem, dqn_agent_,
                                               next_agent_occ_dict_, 
                                               explore=False, seq_flag=SEQ_ACT, norm_feat_flag=False)

          # print(f"time: {t__}\tdem_sum: {-sum(rew)}")


        else:

          next_graph_, action_duration, demand_, (_, _, _, _) = env_utils.env_step(\
            next_graph_, t_rem, dqn_agent_, None, explore=False, seq_flag=SEQ_ACT)
              
        # sum_episode_demand += sum([next_graph_.nodes[node]['priority'] * next_graph_.nodes[node]['demand'] for node in next_graph_.nodes()])
        sum_episode_demand += copy.deepcopy(demand_) # sum([next_graph_.nodes[node]["priority"] * (prev_g.nodes[node]['demand'] + 1) * (1 - next_graph_.nodes[node]['num_occ_agents']) for node in next_graph_.nodes()]) #-sum(rew)
        sum_action_time += action_duration

        # par_sol = update_sol(next_graph_, par_sol, t__+1)

    # exit()

    if BEAM_SEARCH:
      sum_episode_demand_bs = min([episode_demand_dict_prev[_] \
       for _ in list(episode_demand_dict_prev.keys())])

      sum_episode_demand_list = [sum_episode_demand_bs, sum_episode_demand]

    if BEAM_SEARCH:
      return sum_episode_demand_list, sum_action_time
    else:
      return sum_episode_demand, sum_action_time, par_sol


  time_for_opt_computation = {} # f"NOT_COMPUTED"

  graph_read_path = save_path
  # print(f"{graph_read_path}")


  for i_ in range(1, NUM_TEST_INSTANCES+1):

    G = nx.read_gpickle(f"{graph_read_path}/test_instances/instance_{i_}/time_step_0.gpickle")

    prev_graph = G

    agent_set, init_agent_occ_dict = graph_utils.get_agent_set_init_agent_dict(G, NUM_NODES)

    opt_val_dict[i_] = 0

    for ti in range(1, MPC_HORIZON+1):

      temp_graph_obj = nx.read_gpickle(f"{graph_read_path}/test_instances/instance_{i_}/time_step_{ti}.gpickle")

      # print(f"time: {ti}\t node: {temp_graph_obj.nodes[NUM_NODES]['agents_occ']}")

      dem_sum = sum([temp_graph_obj.nodes[node]['priority'] * temp_graph_obj.nodes[node]['demand'] for node in temp_graph_obj.nodes()])

      # print(f"time: {ti}\tdem_sum: {dem_sum}")

      if prev_graph.nodes[NUM_NODES]['agents_occ'] == temp_graph_obj.nodes[NUM_NODES]['agents_occ']:
        print(f"SAME")

      prev_graph = temp_graph_obj
    
      opt_val_dict[i_] += dem_sum #sum([temp_graph_obj.nodes[node]['priority'] * temp_graph_obj.nodes[node]['demand'] for node in temp_graph_obj.nodes()])

    # exit()

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

    buff = torch.load(f"./trained_pols/{NUM_NODES}N-{NUM_AGENTS}A-{MAX_SIM_TIME}T_G{graph_num}/pol_{pol_num}/pol_iter_{iter_}.pt")
    # print(buff)
    dqn_agent.policy_net.load_state_dict(buff['state'])



    rl_test_sol_obj = {}
    time_for_rl_sol_computation = {}
    sum_rl_test_sol_obj_beam_search = 0
    time_for_rl_sol_computation_beamsearch = 0
    for i_ in range(1, NUM_TEST_INSTANCES+1):
      with open(f"{save_path}/eval_data/eval_inst_{i_}_pol_{pol_num}_iter_{iter_}.csv",
                  'a', newline='') as f:
        spamwriter = csv.writer(f, delimiter=',', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['opt_obj', 'RL_pol_obj'])
      eval_graph = nx.read_gpickle(
                    f"{graph_read_path}/test_instances/instance_{i_}/time_step_0.gpickle")
      _, agent_occ_dict = graph_utils.get_agent_set_init_agent_dict(eval_graph,
                                                                    NUM_NODES)
      eval_graph = graph_utils.update_node_features(eval_graph, \
        HORIZON, agent_occ_dict, None, None, None, True)
      next_agent_occ_dict = copy.deepcopy(agent_occ_dict)
      print(f"testing_instance: {i_}\tgraph: {graph_num}\tpolicy: {pol_num}\titer: {iter_}", end="\r")

      if BEAM_SEARCH and SEQ_ACT:

        init_time = time.time()
        episode_dem_temp_list, act_time = run_MPC(i_, eval_graph, \
                                                dqn_agent, next_agent_occ_dict)
        time_for_rl_sol_computation_beamsearch += \
                            (time.time() - init_time)/NUM_TEST_INSTANCES
        time_for_rl_sol_computation[i_] = act_time

        rl_test_sol_obj[i_] = episode_dem_temp_list[1]
        sum_rl_test_sol_obj_beam_search += episode_dem_temp_list[0]

      else:
        episode_dem_temp, act_time, partial_solution = \
           run_MPC(i_, eval_graph, dqn_agent, next_agent_occ_dict)
        rl_test_sol_obj[i_] = copy.deepcopy(float(episode_dem_temp)) #+ init_dem_dict[i_]

        time_for_rl_sol_computation[i_] = act_time

      with open(f"{save_path}/eval_data/eval_inst_{i_}_pol_{pol_num}_iter_{iter_}.csv",
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

      else:
        avg_rl_opt_gap = sum([100*((rl_test_sol_obj[_]/opt_val_dict[_]) - 1)\
         for _ in range(1, NUM_TEST_INSTANCES+1)]) / NUM_TEST_INSTANCES

        stddev_opt_gap = np.std(np.asarray([100*((rl_test_sol_obj[_]/opt_val_dict[_]) - 1)\
         for _ in range(1, NUM_TEST_INSTANCES+1)]))

        avg_rl_sol_time = sum([time_for_rl_sol_computation[_]\
         for _ in range(1, NUM_TEST_INSTANCES+1)]) / NUM_TEST_INSTANCES

        opt_gap_list.append(avg_rl_opt_gap)

        with open(f"{save_path}/eval_data/eval_inst_{i_}_pol_{pol_num}_overall.csv",
                'a', newline='') as f:
          spamwriter = csv.writer(f, delimiter=',', quotechar='|',
                                  quoting=csv.QUOTE_MINIMAL)
          spamwriter.writerow([f'{iter_}', f'{avg_rl_opt_gap}', f'{stddev_opt_gap}', f'{avg_rl_sol_time}'])
  print('Complete')




if __name__ == "__main__":

  args_list = []

  g_num = str(sys.argv[1])
  p_num = str(sys.argv[2])

  args_list.append([g_num, p_num])
  eval_graph_pol(args_list[-1])

