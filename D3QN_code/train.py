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
import pickle

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

# set up matplotlib
# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
#     from IPython import display

# plt.ion()


def train_a_pol(args):
  torch.set_num_threads(1)
  [graph_num, pol_num] = args
  NUM_NODES = data_file.NUM_NODES
  NUM_AGENTS = data_file.NUM_AGENTS
  SEQ_ACT = data_file.SEQ_ACT
  MAX_SIM_TIME = data_file.MAX_SIM_TIME
  HORIZON = data_file.HORIZON
  REPLAN_INTERVAL = data_file.REPLAN_INTERVAL 
  NUM_TEST_INSTANCES = data_file.NUM_TEST_INSTANCES
  EVAL_INTERVAL = data_file.EVAL_INTERVAL
  EVAL_FOLDER = data_file.EVAL_FOLDER
  opt_val_dict = {}
  min_yet = np.inf

  # curr_date_time = f"{datetime.datetime.now().strftime(f'%Y%m%d_%H%M%S')}"
  save_path = f"./{NUM_NODES}N-{NUM_AGENTS}A-{MAX_SIM_TIME}T_G{graph_num}"


  if SEQ_ACT:
    from dqn_related_class_multiQout import DQN as Agent
  else:
    from ddpg_related_class import DDPG as Agent


  G = nx.read_gpickle(f"{save_path}/test_instances/instance_1_multi.gpickle")



  num_episodes = data_file.NUM_TRAIN_EPISODES

  G_copy = graph_utils.initiate_node_demands(G)
  G_copy = graph_utils.agent_pos_random_reset(G_copy, NUM_AGENTS)
  agent_set, init_agent_occ_dict = graph_utils.get_agent_set_init_agent_dict(G_copy, NUM_NODES)
  G_copy = graph_utils.update_node_features(G_copy, HORIZON, 
                                            init_agent_occ_dict,
                                            None, None, None, True)
  pyg = from_networkx(G_copy)

  if SEQ_ACT:
    dqn_agent = Agent(G_copy, pyg, NUM_AGENTS=len(agent_set))

  else:
    dqn_agent = Agent(G_copy, NUM_AGENTS=len(agent_set))




  pol_save_path = f"trained_pols/{save_path}/pol_{pol_num}"
  
  os.makedirs(f"{pol_save_path}", exist_ok=True)

  explore = True

  for i_episode in range(num_episodes):

    episode_this_state_list = []
    episode_action_list = []
    episode_return_list = []
    episode_reward_list = []
    episode_next_state_list = []
    episode_is_this_max_q_list = []

    next_graph = copy.deepcopy(G_copy)
    # next_agent_occ_dict = copy.deepcopy(init_agent_occ_dict)

    next_graph = graph_utils.initiate_node_demands(next_graph)
    next_graph = graph_utils.agent_pos_random_reset(next_graph, NUM_AGENTS)
    agent_set, next_agent_occ_dict = graph_utils.get_agent_set_init_agent_dict(\
                                                          next_graph, NUM_NODES)
    next_graph = graph_utils.update_node_features(next_graph, HORIZON,
                                                  next_agent_occ_dict,
                                                  None, None, None, True)

    
    
    list_of_graphs_to_plot = []

    plans = {}

    for agent_ in agent_set:
      plans[agent_] = {}

    for rec_hor_iter in range(int(MAX_SIM_TIME/REPLAN_INTERVAL)):
      mpc_init_timestep = rec_hor_iter*REPLAN_INTERVAL
      t = mpc_init_timestep
      t_left = (mpc_init_timestep+HORIZON) - t
      # print(f"mpc_init_timestep: {mpc_init_timestep}")
      # for t in range(mpc_init_timestep, mpc_init_timestep+HORIZON+1):
      while (t in list(range(mpc_init_timestep, mpc_init_timestep+HORIZON+1))):

        prev_location = next_graph.nodes[NUM_NODES]['agents_occ'][0]

        current_plan_id = rec_hor_iter # int(t/REPLAN_INTERVAL)

        plans[agent_][current_plan_id] = {}

        print(f"time {t} of episode {i_episode}...", end="\r")


        if SEQ_ACT:

          next_graph, next_agent_occ_dict, demand, (state_feat_list, \
          action_list, reward_list, next_state_feat_list, is_this_max_q), plans, \
          _ = env_utils.env_step(next_graph, plans, t, current_plan_id,
                                 t_left, dqn_agent, next_agent_occ_dict, 
                                 explore=explore, seq_flag=SEQ_ACT)


          episode_this_state_list.append(state_feat_list)
          episode_action_list.append(action_list)
          episode_reward_list.append(reward_list)
          episode_next_state_list.append(next_state_feat_list)
          episode_is_this_max_q_list.append(is_this_max_q)
          # Perform one step of the optimization (on the policy network)

        else:
          next_graph, _, _, (state, action, rew, next_state) = \
                            env_utils.env_step(next_graph, t_left, \
                            dqn_agent, None, explore=explore, seq_flag=SEQ_ACT)

        next_location = next_graph.nodes[NUM_NODES]['agents_occ'][0]

        num_time_step_skip = \
                   next_graph[prev_location][next_location]['travel_time']

        t += num_time_step_skip

        t_left = (mpc_init_timestep+HORIZON) - t

        for _ in range(num_time_step_skip):
          if SEQ_ACT:
            dqn_agent.optimize_model()
          else:
            dqn_agent.update()

        done = (t_left <= 0) # True if (t == MAX_SIM_TIME) else False
        
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


    explore = False if ((not i_episode) or (((i_episode+1) % EVAL_INTERVAL) \
                                                              == 0)) else True


    if not explore:
      torch.save({'state': dqn_agent.policy_net.state_dict()}, 
                 f"{pol_save_path}/pol_iter_{(i_episode+1)*MAX_SIM_TIME}.pt")

      with open(f'{pol_save_path}/latest_agent_obj.pkl', 'wb') as file: 
        pickle.dump(dqn_agent, file)

      explore = True
    

  print('Complete')



if __name__ == "__main__":

  args_list = []
  g_num = str(sys.argv[1])
  p_num = str(sys.argv[2])

  args_list.append([g_num, p_num])
  train_a_pol(args_list[-1])
  # for g_num in range(1, data_file.NUM_GRAPHS+1):
   # for p_num in range(1, data_file.NUM_TRAIN_POL+1):
    #  args_list.append([g_num, p_num])
     # train_a_pol(args_list[-1])
      # exit()

  # pool = Pool(10)
  # pool.map(train_a_pol, args_list)





