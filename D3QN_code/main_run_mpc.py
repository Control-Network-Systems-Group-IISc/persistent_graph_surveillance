# import math
# import random

import os

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

# set up matplotlib
# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
#     from IPython import display

# plt.ion()

opt_gap_list = []
opt_gap_list_bs = []
episode_loss = []


GET_OPT = True
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
min_yet = np.inf

curr_date_time = f"{datetime.datetime.now().strftime(f'%Y%m%d_%H%M%S')}"
save_path = f"./{EVAL_FOLDER}"

if SEQ_ACT:
  from dqn_related_class_multiQout import DQN as Agent
else:
  from ddpg_related_class import DDPG as Agent

def plot_demands(show_result=False):
  # if demands:
  # fig = plt.figure()
  # fig1, fig2 = fig.subfigures(1, 2)
  # demands_t = torch.tensor(opt_gap_list, dtype=torch.float)
  if show_result:
    plt.title('Result N_a x Up-Bnd. on degree of nodes')
  else:
    plt.clf()
    plt.title('Training... N_a x Up-Bnd. on degree of nodes')

  plt.subplot(2, 1, 1)
  plt.xlabel(f'Episode/{EVAL_INTERVAL}')
  plt.ylabel('%Optim_Gap')
  plt.plot(opt_gap_list, color='green')
  # plt.hold(True)
  plt.plot(opt_gap_list_bs, color='blue')

  plt.subplot(2, 1, 2)
  plt.xlabel(f'Episode/{EVAL_INTERVAL}')
  plt.ylabel('TD_loss')
  plt.plot(episode_loss)

  plt.tight_layout()

  plt.savefig(f"{save_path}/learn_curve_{curr_date_time}.png", dpi=600)





def evaluate_current_policy(next_graph_, dqn_agent_, next_agent_occ_dict_):

  sum_episode_demand = 0
  sum_action_time = 0

  plans = {}
  for agent_ in agent_set:
    plans[agent_] = {}
  
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

  for t__ in range(HORIZON+1):

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

      next_graph_nbs, next_agent_occ_dict_, demand_, (_, _, _, _, _), plans, \
      action_duration = env_utils.env_step(next_graph_nbs, t_rem, dqn_agent_, 
                                           next_agent_occ_dict_, explore=False,
                                           seq_flag=SEQ_ACT)
          
      # demand_list.append(demand_)
      sum_episode_demand += demand_
      sum_action_time += action_duration

    else:

      if SEQ_ACT:

        next_graph_, next_agent_occ_dict_, demand_, (_, _, _, _, _), plans,\
        action_duration = env_utils.env_step(next_graph_, plans, t__, 
                                             current_plan_id, t_rem, dqn_agent_,
                                             next_agent_occ_dict_, 
                                             explore=False, seq_flag=SEQ_ACT)

      else:

        next_graph_, action_duration, demand_, (_, _, _, _) = env_utils.env_step(\
          next_graph_, t_rem, dqn_agent_, None, explore=False, seq_flag=SEQ_ACT)
            
      # demand_list.append(demand_)
      sum_episode_demand += demand_
      sum_action_time += action_duration

  if BEAM_SEARCH:
    sum_episode_demand_bs = min([episode_demand_dict_prev[_] \
     for _ in list(episode_demand_dict_prev.keys())])

    sum_episode_demand_list = [sum_episode_demand_bs, sum_episode_demand]

  if BEAM_SEARCH:
    return sum_episode_demand_list, sum_action_time
  else:
    return sum_episode_demand, sum_action_time


time_for_opt_computation = {} # f"NOT_COMPUTED"
# G, _, _ = graph_utils.gen_graph_and_agents(NUM_NODES, 1, 5, 0.5, NUM_AGENTS)

G = nx.read_gpickle(f"{save_path}/test_instances/instance_1.gpickle")


 # graph_utils.gen_graph_and_agents(NUM_NODES, 1, 5, 0.5, NUM_AGENTS)


agent_set, init_agent_occ_dict = graph_utils.get_agent_set_init_agent_dict(G, NUM_NODES)

for i_ in range(1, NUM_TEST_INSTANCES+1):
  temp_graph_obj = None
  temp_opt_sol_time = None
  with open(f"{save_path}/test_instances/opt_sol_instance_{i_}.csv") as f:
    reader = csv.reader(f)
    for row_ind, row in enumerate(reader):
      if row[0] == "obj":
        opt_val_dict[i_] = float(row[1])

      if row[0] == "sol_time":
        time_for_opt_computation[i_] = float(row[1])

      if (i_ in opt_val_dict.keys()) and (i_ in time_for_opt_computation.keys()):
        break

  # sum_opt_obj_vals += temp_graph_obj

avg_opt_sol_time = sum([time_for_opt_computation[i_] \
                       for i_ in time_for_opt_computation.keys()])/NUM_TEST_INSTANCES

num_episodes = 5100

G_copy = graph_utils.initiate_node_demands(G)
G_copy = graph_utils.update_node_features(G_copy, HORIZON, init_agent_occ_dict,
                                          None, None, None, True)
pyg = from_networkx(G_copy)

if SEQ_ACT:
  dqn_agent = Agent(G_copy, pyg, NUM_AGENTS=len(agent_set))

else:
  dqn_agent = Agent(G_copy, NUM_AGENTS=len(agent_set))

for i_episode in range(num_episodes):

  episode_this_state_list = []
  episode_action_list = []
  episode_return_list = []
  episode_reward_list = []
  episode_next_state_list = []
  episode_is_this_max_q_list = []

  next_graph = copy.deepcopy(G_copy)
  next_agent_occ_dict = copy.deepcopy(init_agent_occ_dict)

  next_graph = graph_utils.initiate_node_demands(next_graph)
  next_graph = graph_utils.update_node_features(next_graph, HORIZON,
                                                next_agent_occ_dict,
                                                None, None, None, True)

  explore = False if ((not i_episode) or (((i_episode+1) % EVAL_INTERVAL) \
                                                                == 0)) else True
  
  list_of_graphs_to_plot = []

  plans = {}

  for agent_ in agent_set:
    plans[agent_] = {}

  for rec_hor_iter in range(int(MAX_SIM_TIME/REPLAN_INTERVAL)):
    mpc_init_timestep = rec_hor_iter*REPLAN_INTERVAL
    # print(f"mpc_init_timestep: {mpc_init_timestep}")
    for t in range(mpc_init_timestep, mpc_init_timestep+HORIZON+1):

      t_left = (mpc_init_timestep+HORIZON) - t

      current_plan_id = rec_hor_iter # int(t/REPLAN_INTERVAL)

      plans[agent_][current_plan_id] = {}

      print(f"time {t} of episode {i_episode}...", end="\r")


      if SEQ_ACT:

        next_graph, next_agent_occ_dict, demand, (state_feat_list, \
        action_list, reward_list, next_state_feat_list, is_this_max_q), plans, \
        _ = env_utils.env_step(next_graph, plans, t, current_plan_id, t_left, dqn_agent, next_agent_occ_dict,
                               explore=explore, seq_flag=SEQ_ACT)


        episode_this_state_list.append(state_feat_list)
        episode_action_list.append(action_list)
        episode_reward_list.append(reward_list)
        episode_next_state_list.append(next_state_feat_list)
        episode_is_this_max_q_list.append(is_this_max_q)
        # Perform one step of the optimization (on the policy network)

      else:
        next_graph, _, _, (state, action, rew, next_state) = env_utils.env_step(\
          next_graph, t_left, dqn_agent, None, explore=explore, seq_flag=SEQ_ACT)
      
      for _ in range(1):
        if SEQ_ACT:
          dqn_agent.optimize_model()
        else:
          dqn_agent.update()

      done = True if (t == MAX_SIM_TIME) else False
      
      if SEQ_ACT:
        if done:
          episode_next_state_list[-1][-1] = None


          episode_return_list = episode_reward_list 
                                # utils.extract_episode_retrun_values(\
                                #                     episode_reward_list)

          # Store the transition in memory
          for i_trans in range(len(episode_return_list)):
            state_feat_list = episode_this_state_list[i_trans]
            action_list = episode_action_list[i_trans]
            ret_list = episode_return_list[i_trans]
            next_state_feat_list = episode_next_state_list[i_trans]
            is_this_max_q = episode_is_this_max_q_list[i_trans]
            for i_exp, ret in enumerate(ret_list):
              # print(f"ret: {ret}")
              dqn_agent.memory.push(state_feat_list[i_exp], \
            	              action_list[i_exp], torch.tensor([ret]), \
            	              next_state_feat_list[i_exp], is_this_max_q[i_exp])
          break

      else:
        # if done:
        #   next_state = None
        dqn_agent.memory.push(state, action, rew, next_state, done)





  if not explore:
    rl_test_sol_obj = {}
    time_for_rl_sol_computation = {}
    # sum_rl_test_sol_obj = 0
    sum_rl_test_sol_obj_beam_search = 0
    # time_for_rl_sol_computation = 0
    time_for_rl_sol_computation_beamsearch = 0
    for i_ in range(1, NUM_TEST_INSTANCES+1):
      eval_graph = nx.read_gpickle(
                    f"{save_path}/test_instances/instance_{i_}.gpickle")
      _, agent_occ_dict = graph_utils.get_agent_set_init_agent_dict(eval_graph,
                                                                    NUM_NODES)
      eval_graph = graph_utils.update_node_features(eval_graph, \
        HORIZON, agent_occ_dict, None, None, None, True)
      next_agent_occ_dict = copy.deepcopy(agent_occ_dict)
      print(f"testing for instance {i_}", end="\r")

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
        episode_dem_temp, act_time = evaluate_current_policy(eval_graph, \
                                                dqn_agent, next_agent_occ_dict)
        rl_test_sol_obj[i_] = episode_dem_temp

        time_for_rl_sol_computation[i_] = act_time
        #(act_time)/NUM_TEST_INSTANCES

        with open(f"{save_path}/test_instances/policy_eval_compare.csv", 'a',
                  newline='') as f:
          spamwriter = csv.writer(f, delimiter=',', quotechar='|',
                                  quoting=csv.QUOTE_MINIMAL)
          if i_episode == 0:
            spamwriter.writerow(['iter', 'opt_obj', 'RL_pol_obj'])
          spamwriter.writerow([f'{i_episode+1}', f'{opt_val_dict[i_]}', 
                               f'{episode_dem_temp}'])
    episode_loss.append(dqn_agent.loss)
    if NUM_TEST_INSTANCES > 0:

      if BEAM_SEARCH and SEQ_ACT:
        avg_rl_opt_gap = sum([100*((rl_test_sol_obj[_]/opt_val_dict[_]) - 1)\
          for _ in range(1, NUM_TEST_INSTANCES+1)]) / NUM_TEST_INSTANCES
        opt_gap_list.append(avg_rl_opt_gap)
        sum_opt_val = sum([opt_val_dict[_] for _ in range(1, NUM_TEST_INSTANCES+1)])
        opt_gap_list_bs.append(100*\
          ((sum_rl_test_sol_obj_beam_search/sum_opt_val) - 1))

        print(f"\n------------------------\ntime_for_opt_computation: \
        {avg_opt_sol_time}s\t time_for greedy_rl_sol_computation: \
        {time_for_rl_sol_computation}s\nGreedy OPT_GAP: {opt_gap_list[-1]}%\n \
        time_for beam-search_rl_sol_computation: \
        {time_for_rl_sol_computation_beamsearch}s\n \
        Beam-search OPT_GAP: {opt_gap_list_bs[-1]}%\n \
        ------------------------\n")

      else:
        avg_rl_opt_gap = sum([100*((rl_test_sol_obj[_]/opt_val_dict[_]) - 1)\
         for _ in range(1, NUM_TEST_INSTANCES+1)]) / NUM_TEST_INSTANCES

        avg_rl_sol_time = sum([time_for_rl_sol_computation[_]\
         for _ in range(1, NUM_TEST_INSTANCES+1)]) / NUM_TEST_INSTANCES

        opt_gap_list.append(avg_rl_opt_gap)

        print(f"\n------------------------\ntime_for_opt_computation: \
        {avg_opt_sol_time}s\t time_for_rl_sol_computation: \
        {avg_rl_sol_time}s\nOPT_GAP: {opt_gap_list[-1]}%\n \
        ------------------------\n")

    # episode_demands.append(sum(demand_list))
    plot_demands()



    # exit()

print('Complete')
plot_demands(show_result=True)
plt.ioff()
plt.show()






