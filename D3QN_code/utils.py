""" utitlity functions for surveillance DQN implementation
"""

from collections import namedtuple
import copy
import numpy as np
import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Batch



Transition = namedtuple('Transition', ('state_feat', 'action', 'return_',
                                       'next_state_feat', 'is_this_max_q_act'))



def extract_episode_retrun_values(episode_reward_list):

  num_agents = len(episode_reward_list[0])
  # print(f"num_agents: {num_agents}")      
  temp_eps_rew_list = copy.deepcopy(episode_reward_list)
  # print(f"***:{temp_eps_rew_list}")
  temp_eps_rew_list = torch.from_numpy(np.vstack(np.asarray(temp_eps_rew_list).astype(np.float))).flatten()
  temp_eps_rew_list = torch.flip(temp_eps_rew_list, (0,))
  # print(f"temp_eps_rew_list: {temp_eps_rew_list}")
  temp_episode_return_list = [torch.tensor(temp_eps_rew_list[0])] # [[temp_eps_rew_list[-1]]]
  # print(f"==={temp_episode_return_list}")

  for rew in temp_eps_rew_list[1:]:
    temp_episode_return_list.append(temp_episode_return_list[-1]+torch.tensor(rew))

  # print(f"////{temp_episode_return_list}, len: {len(temp_episode_return_list)}")

  for ret_ind, ret in enumerate(temp_episode_return_list):
    # print(f"ret_ind: {ret_ind}")
    if ret_ind == 0:
      episode_return_list = []

    if not (ret_ind % num_agents):
      episode_return_list.append([temp_episode_return_list[ret_ind+num_agents-1]])

    else:
      episode_return_list[-1].append(temp_episode_return_list[ret_ind+num_agents-1-(ret_ind%num_agents)])



  # for 
  #   if rew_ind%num_agents == 0:
  #     temp_episode_return_list.append([])

  #   temp_episode_return_list[-1].append(temp_episode_return_list[-1][-1].flatten() + \
  #                                     torch.tensor(rew))


  # for reward_list in temp_eps_rew_list[1:]:
  #   temp_episode_return_list.append([temp_episode_return_list[-1][-1].flatten() + \
  #                                     torch.tensor(reward_list[-1])])
  #   for rew in reward_list[1:]:
  #   # print(f"----{temp_episode_return_list[-1]}\n--{torch.tensor(reward_list)}\n---{temp_episode_return_list[-1] +torch.tensor(reward_list)}")
  #     temp_episode_return_list.append(temp_episode_return_list[-1][-1].flatten() +\
  #                                   torch.tensor(rew))

  # episode_return_list = copy.deepcopy(temp_episode_return_list)
  episode_return_list.reverse()

  # print(f"episode_return_list: {episode_return_list}")

  return episode_return_list



def get_all_qvals(list_of_graphs, dqn_agent):

  list_of_qval_vecs = []

  for graph in list_of_graphs:
    temp_pyg_graph = from_networkx(graph)
    temp_q_vals = dqn_agent.policy_net(Batch.from_data_list(
        [temp_pyg_graph])).clone().detach().squeeze(0)

    list_of_qval_vecs.append(temp_q_vals.squeeze(0).numpy())

  return np.transpose(np.asarray(list_of_qval_vecs))


def get_max_k_qval_indices(array_of_qvals, num_max_qvals):


  # indices_of_max_k_q_vals = np.dstack(np.unravel_index(\
  #   np.argsort(array_of_qvals.ravel()), np.shape(array_of_qvals)))
  # print(f"array_of_qvals.shape[0] = {array_of_qvals.shape[0]}, max(array_of_qvals.shape[0]-num_max_qvals, num_max_qvals): {-num_max_qvals}")
  indices_of_max_k_q_vals = np.dstack(np.unravel_index(np.argsort(\
    array_of_qvals.ravel())[-num_max_qvals:], \
    np.shape(array_of_qvals))).squeeze(0)

  temp_copy_indices = copy.deepcopy(indices_of_max_k_q_vals)

  # print(f"------------------------")
  # print(f"array_of_qvals: {array_of_qvals}")

  temp_var = 0
  for ind_of_ind, ind in enumerate(temp_copy_indices):
    if array_of_qvals[ind[0], ind[1]] == -np.inf:
      indices_of_max_k_q_vals = np.delete(indices_of_max_k_q_vals, 
                                          ind_of_ind-temp_var, 0)
      temp_var += 1

  # indices_of_max_k_q_vals = indices_of_max_k_q_vals[:num_max_qvals]

  # print(f"indices_of_max_k_q_vals = {indices_of_max_k_q_vals}")

  # print(f"sorted qvals: {-np.sort(-array_of_qvals.ravel())}")

  # for ind in indices_of_max_k_q_vals:
  #   print(f"q_vals: {array_of_qvals[ind[0]][ind[1]]}")


  return indices_of_max_k_q_vals


def get_list_of_graphs_to_expand(list_of_init_graphs, list_of_init_occ_dicts,
                                 indices_of_max_k_q_vals):

  list_of_graphs_to_expand = []
  list_of_occ_dicts = []
  list_of_k_vals_for_graphs = []

  for index_ in indices_of_max_k_q_vals:

    # print(f"index_: {index_}")

    list_of_graphs_to_expand.append(list_of_init_graphs[index_[1]])
    list_of_occ_dicts.append(list_of_init_occ_dicts[index_[1]])
    list_of_k_vals_for_graphs.append(index_[0])

  # print(f"len(list_of_graphs_to_expand): {len(list_of_graphs_to_expand)}")
  return list_of_graphs_to_expand, list_of_occ_dicts, list_of_k_vals_for_graphs

