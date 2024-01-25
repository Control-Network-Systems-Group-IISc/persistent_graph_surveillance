"""Example Google style docstrings.
"""

import math
import random
import time
# import matplotlib.pyplot as plt
from collections import deque
# from itertools import count
import numpy as np
import pprint

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torch_geometric.utils.convert import from_networkx
import torch_geometric
# from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch

# import graph_utils
import utils

import data_file

# import multiprocessing as mp
# from multiprocessing import Pool

# def evaluate(network):
#     return value

class DQN():

  """ DQN main class
  """

  def __init__(self, nx_graph, pyg_graph, NUM_AGENTS, BATCH_SIZE=16, GAMMA=1,
               EPS_START=0.5, EPS_END=0.0001, EPS_DECAY=5000, TAU=1E-3, 
               LR=5E-4, MEM_CAP=50000, NUM_SAMP_REPS=1):

    self.batch_size = BATCH_SIZE
    self.gamma = GAMMA
    self.eps_start = EPS_START
    self.eps_end = EPS_END
    self.eps_decay = EPS_DECAY
    self.tau = TAU
    self.lr = LR
    self.mem_capacity = MEM_CAP
    self.num_sample_reps = NUM_SAMP_REPS

    self.nx_graph = nx_graph
    self.pyg_graph = pyg_graph

    self.num_agents = NUM_AGENTS
    self.num_nodes = self.pyg_graph.num_nodes-self.num_agents
    self.num_max_actions = int(
        self.num_agents * max([d-1 for n, d in self.nx_graph.degree()]))
    self.num_node_features = len(self.nx_graph.nodes[0]['feature'])
    self.num_agent_features = len(self.nx_graph.nodes[self.num_nodes]['agent_feature'])
    self.loss = 0

    # if GPU is to be used
    self.device = 'cpu' 
    # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # buff = torch.load(f"./20230808_165647/weights/pol_net.pt")
    self.policy_net = GNNDQNMultiQout( \
        self.nx_graph, self.pyg_graph, num_agents=self.num_agents)#.to(
            # self.device)
    self.target_net = GNNDQNMultiQout( \
        self.nx_graph, self.pyg_graph, num_agents=self.num_agents)#.to(
            # self.device)

    self.target_net.load_state_dict(self.policy_net.state_dict())

    self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr,
                                 amsgrad=True) #, weight_decay=0.001)#


    self.criterion = nn.SmoothL1Loss().double()

    # self.optimizer = optim.SGD(self.policy_net.parameters(), lr=self.lr,
    #                            momentum=0.9)

    self.memory = ReplayMemory(self.mem_capacity)

    self.steps_done = 0

    self.reset_n = 0
    self.explore_n = 1 - self.reset_n

    self.explore = True


  def get_batch_input(self, list_of_pyg_graphs):

    with torch.no_grad():

      batch_size_inp = len(list_of_pyg_graphs)

      graph_state = list_of_pyg_graphs.feature.float()

      graph_ = graph_state.view(batch_size_inp, 
                                self.num_nodes+self.num_agents,
                                self.num_node_features)[:, :-self.num_agents, :]

      agent_ = list_of_pyg_graphs.agent_feature.view(batch_size_inp, 
                                self.num_nodes+self.num_agents,
                                self.num_agent_features)[:, self.num_nodes:, :]

      global_aggr = graph_.view(batch_size_inp,
                                self.num_nodes * self.num_node_features)

      concat_layer = None

      num_acts = {}

      neighbour_list = []

      for i in range(batch_size_inp):
        num_acts[i] = self.num_max_actions
        concat_temp = None
        curr_node_list = []
        agent__ = agent_[i]

        for agent_feat in agent__:
          try:
            if int(agent_feat[-1]) > 0:
              node = int(agent_feat[-2])
              curr_node_list.append(int(node))
              
          except Exception as e:
            print(f"@@@@@@@@@\n{agent_feat}")
            print(f"eeeeeeeeee: {e}")
            print(error)

        temp_dict = {}

        for node in curr_node_list:
          temp_dict[int(node)] = [int(node)]
          for neigh_node in self.nx_graph.neighbors(int(node)):
            if int(node) != int(neigh_node):
              temp_dict[int(node)].append(neigh_node)

        neighbour_list.append(temp_dict[int(node)]+[-1 \
          for _ in range(self.num_max_actions - len(temp_dict[int(node)]))])


        for neigh_node in temp_dict[int(node)]:

          if concat_temp is None:
            concat_temp = graph_state[(i * \
                      (self.num_nodes + self.num_agents)) +\
                       int(node)].clone().detach()

          elif int(node) != int(neigh_node):
            concat_temp = torch.cat((concat_temp, graph_state[(i * \
                    (self.num_nodes + self.num_agents)) + neigh_node]), -1)

        for _ in range(self.num_max_actions - \
            int(len(concat_temp)/(self.num_node_features))):
          concat_temp = torch.cat((concat_temp, \
                    torch.zeros(self.num_node_features)))
          num_acts[i] += -1


        if concat_layer is None:
          concat_layer = concat_temp

        else:
          concat_layer = torch.cat((concat_layer, concat_temp))

    return [[global_aggr.view(batch_size_inp, self.num_nodes,\
              self.num_node_features).detach(),\
              concat_layer.view(batch_size_inp, self.num_max_actions,
                                self.num_node_features).detach(), 
             neighbour_list],
            num_acts]


  def select_action(self, nx_graph, graph, explore=True, k_val=None):

    ###### FIX ME FIX ME FIX ME FIX ME ########

    """
    Select the current node and next node to go.
    
    """
    did_explore = False
    sample = random.random()
    eps_threshold = ((self.eps_start - self.eps_end) * math.exp(
        -1. * self.steps_done / self.eps_decay)) + self.eps_end

    # eps_threshold = (base_eps_threshold) + (0.5*base_eps_threshold * math.exp(
    #         -1. * (self.steps_done%data_file.HORIZON) / (data_file.HORIZON/3)))

    # max(self.eps_end, self.eps_end + (self.eps_start - self.eps_end) *  \
    # (1 -  (self.steps_done / self.eps_decay))) \
    # self.eps_end + (self.eps_start - self.eps_end) * \
      # math.exp(-1. * self.steps_done / self.eps_decay)
    if explore:
      if self.steps_done > 10000 * data_file.HORIZON:
        eps_threshold = 10**(-6)
      self.steps_done += 1

    num_available_actions = []

    occ_nodes = []

    for agent_id in range(self.num_agents):
      if nx_graph.nodes[self.num_nodes+agent_id]['active']:
        occ_nodes.append(int(nx_graph.nodes[\
                                    self.num_nodes+agent_id]['agents_occ'][0]))
        node = occ_nodes[-1]
        if len(num_available_actions) == 0:
          num_available_actions.append(len(\
                             list(self.nx_graph.neighbors(int(node)))))
        else:
          num_available_actions.append(num_available_actions[-1] +\
                len(list(self.nx_graph.neighbors(int(node)))))

    total_num_action = num_available_actions[-1] #len(q_vals)
    # print(f"total_num_action: {total_num_action}")
    with torch.no_grad():
      self.policy_net.eval()
      q_vals = self.policy_net(self.get_batch_input(Batch.from_data_list(
                               [graph]))[0]).clone().detach().squeeze(0)

      # a_vals = s_q_vals[1 : total_num_action]
      # max_a_val = max(list(a_vals))
      # average_a_val = np.average(lsit(a_vals))
      # st_val = s_q_vals[0]

    if (sample > eps_threshold) or (not explore):

      if k_val is None:
        with torch.no_grad():
          # ######### naive
          # act_index = np.asarray([float(st_val + a_val - max_a_val) \
          #                                   for a_val in a_vals]).argmax()
          # ######### alternative
          # act_index = np.asarray([float(st_val + a_val - average_a_val) \
          #                                   for a_val in a_vals]).argmax()

          act_index = q_vals[:total_num_action].squeeze(0).max(0)[1].view(1, 1)


      else:
        with torch.no_grad():
          act_index = torch.tensor(np.asarray([k_val])).view(1, 1)

      # t.max(1) will return the largest column value of each row.
      # second column on max result is index of where max element was
      # found, so we pick action with the larger expected reward.
    else:
      did_explore = True
      act_index = torch.tensor([[np.random.choice(total_num_action)]],
                               device=self.device, dtype=torch.long)

    for agent_ind, node in zip(list(range(self.num_agents)), occ_nodes):
      if (int(act_index) > num_available_actions[agent_ind] - 1):
        continue

      else:
        act_curr_node = int(node)
        temp_neigh_list = [int(node)]

        for _ in list(self.nx_graph.neighbors(act_curr_node)):
          if _ != int(node):
            temp_neigh_list.append(_)

        if agent_ind == 0:
          act_next_node = temp_neigh_list[int(act_index)]
          break

        elif agent_ind > 0:
          act_next_node = temp_neigh_list\
                [int(act_index) - num_available_actions[agent_ind - 1] - 1]
          break

    action_node_pair = (agent_ind, act_curr_node, act_next_node)

    if q_vals[act_index] is -np.inf:
      print(f"ERROR!!!!")
      print(error)

    # print(f"-------------------------\n---------------------")

    return act_index, action_node_pair, num_available_actions[-1]

  def optimize_model(self):

    """
    Optimization step for Q-network
    """

    if len(self.memory) < self.batch_size: #self.mem_capacity/4:
    #self.num_sample_reps * self.batch_size:
      return


    all_init_time = time.time()

    self.policy_net.train()
    self.target_net.train()

    for _ in range(self.num_sample_reps):
      transitions = self.memory.sample(self.batch_size)# )
      # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
      # detailed explanation). This converts batch-array of Transitions
      # to Transition of batch-arrays.
      batch = utils.Transition(*zip(*transitions))

      # Compute a mask of non-final states and concatenate the batch elements
      # (a final state would've been the one after which simulation ended)

      non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state_feat)),
                                    device=self.device,
                                    dtype=torch.bool)

      [non_final_next_states, non_final_next_states_acts] = \
                                     self.get_batch_input(Batch.from_data_list(\
                           [s for s in batch.next_state_feat if s is not None]))

      action_batch = torch.cat(batch.action)
      return_batch = torch.cat(batch.return_)

      # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
      # columns of actions taken. These are the actions which would've been taken
      # for each batch state according to policy_net
      state_batch = self.get_batch_input(Batch.from_data_list(list(batch.state_feat)))[0]

      list_of_action_indices = []
      for act_num, action in enumerate(action_batch):
        list_of_action_indices.append(act_num*(self.num_max_actions) + int(action))

      # list_of_state_indices = [st_num*(1 + self.num_max_actions) for st_num in range(self.batch_size)]


      state_action_values = self.policy_net(state_batch).flatten()[list_of_action_indices]

      # action_values = state_and_act_all_vals.flatten()[list_of_action_indices]
      # state_values = state_and_act_all_vals.flatten()[list_of_state_indices]

      # state_action_values = []

      # for ind_, _ in 

      assert -np.inf not in list(state_action_values)

      # Compute V(s_{t+1}) for all next states.
      # Expected values of actions for non_final_next_states are computed based
      # on the "older" target_net; selecting their best reward with max(1)[0].
      # This is merged based on the mask, such that we'll have either the expected
      # state value or 0 in case the state was final.
      next_state_values = torch.zeros(
          self.batch_size, device=self.device, dtype=torch.float)
      with torch.no_grad():
        next_state_qall_act_q_vals_policy = self.policy_net(
            non_final_next_states)

        next_state_qall_act_q_vals = self.target_net(non_final_next_states)

        temp_var = 0
        for _ind, _var in enumerate(batch.next_state_feat):
          if _var is not None:
            all_act_q_vals_policy = next_state_qall_act_q_vals_policy[temp_var]
            all_act_q_vals = next_state_qall_act_q_vals[temp_var]
            next_state_values[_ind] = min(all_act_q_vals.squeeze(0)[\
                                                               np.asarray(list(\
                                              all_act_q_vals_policy.squeeze(0))\
                                              [:int(non_final_next_states_acts[\
                                                        temp_var])]).argmax()],\
                                                         all_act_q_vals_policy[\
                                               :int(non_final_next_states_acts[\
                                               temp_var])].squeeze(0).max(0)[1])
            temp_var += 1        

      # Compute the expected Q values
      expected_state_action_values = (return_batch) +\
                                     (next_state_values * self.gamma)
      
      loss_calc_init = time.time()                                     
      loss = self.criterion(state_action_values, expected_state_action_values)
      # l1_regularization = 0
      # for param in self.policy_net.parameters():
      #   l1_regularization += param.abs().sum()

      # loss += 0.1*l1_regularization

      loss_calc_time = time.time() - loss_calc_init
      self.loss = loss.clone().detach().numpy()

      # Optimize the model
      loss_back_init = time.time()
      loss.backward()
      # In-place gradient clipping
      loss_back_time = time.time() - loss_back_init

    torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
    self.optimizer.step()
    self.soft_target_update()
    for param in self.policy_net.parameters():
      param.grad = None

  def soft_target_update(self):

    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = self.target_net.state_dict()
    policy_net_state_dict = self.policy_net.state_dict()
    for key in policy_net_state_dict:
      target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + (
                    target_net_state_dict[key] * (1-self.tau))
    self.target_net.load_state_dict(target_net_state_dict)


class ReplayMemory(object):

  """Example Google style docstrings.
  """

  def __init__(self, capacity):
    self.memory = deque([], maxlen=capacity)

  def push(self, *args):
    """Save a transition"""
    self.memory.append(utils.Transition(*args))

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)


class EncoderRNN(nn.Module):

  """Example Google style docstrings.
  """

  def __init__(self, input_size, hidden_size):
    super(EncoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.input_size = input_size
    self.gru = nn.GRU(input_size, hidden_size,
                      bidirectional=True, batch_first=True)

  def forward(self, input_):

    with torch.no_grad():
      init_hidden_state = torch.zeros(2,
                                      input_.size(0),
                                      self.hidden_size)

    output, hidden = self.gru(input_, init_hidden_state)
    # print(f"##### hidden size: {hidden.size()}")
    return output, hidden.view(input_.size(0),
                               1, 2*self.hidden_size)


class BahdanauAttention(nn.Module):
  """Example Google style docstrings.
  """
  def __init__(self, hidden_size):
    super(BahdanauAttention, self).__init__()
    self.hidden_size = hidden_size
    self.Wa = nn.Linear(hidden_size, hidden_size)
    self.Ua = nn.Linear(hidden_size, hidden_size)
    self.Va = nn.Linear(hidden_size, 1)

  def forward(self, query, keys):

    # if query.size(0) > 2:
    # print(f"*****query size: {query.size()}")
    # print(f"*****keys size: {keys.size()}")

    scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
    scores = scores.squeeze(2).unsqueeze(1)

    weights = F.softmax(scores, dim=-1)
    context = torch.bmm(weights, keys)

    return context, weights

class AttnDecoderRNN(nn.Module):
  """Example Google style docstrings.
  """
  def __init__(self, hidden_size, output_size):
    super(AttnDecoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.attention = BahdanauAttention(2 * hidden_size)
    self.gru = nn.GRU(4 * hidden_size,
                      2 * hidden_size, batch_first=True)
    self.out = nn.Linear(2 * hidden_size, output_size)

  def forward(self, encoder_outputs, encoder_hidden, target_tensor):
    # batch_size = encoder_outputs.size(0)
    # decoder_input = torch.empty(batch_size, 1, 2 * self.hidden_size,
                                # dtype=torch.float).fill_(0)
    # decoder_hidden = encoder_hidden
    decoder_outputs = []
    attentions = []

    for i in range(target_tensor.size(1)):
      # Teacher forcing: Feed the target as the next input
      decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
      decoder_output, _, attn_weights, context = self.forward_step(
          decoder_input, encoder_hidden, encoder_outputs)

      decoder_outputs.append(decoder_output)
      attentions.append(attn_weights)

      if i == 0:
        init_context = context

    decoder_outputs = torch.cat(decoder_outputs, dim=1)
    # decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
    # attentions = torch.cat(attentions, dim=1)

    return decoder_outputs, init_context#, decoder_hidden, attentions


  def forward_step(self, input_, hidden, encoder_outputs):

    query = hidden#.permute(1, 0, 2)
    context, attn_weights = self.attention(query, encoder_outputs)
    # print(f"context size: {context.size()}")
    # print(f"input_ size: {input_.size()}")
    # print(f"hidden size: {hidden.size()}")


    input_gru = torch.cat((input_, context), dim=2)

    # print(f"input_gru size : {input_gru.size()}")

    output, hidden = self.gru(input_gru, hidden.permute(1, 0, 2))
    output = self.out(output)

    return output, hidden, attn_weights, context


class GNNDQNMultiQout(nn.Module):
  """Example Google style docstrings.
  """

  def __init__(self, nx_graph, pyg_graph, num_agents=1):
    super(GNNDQNMultiQout, self).__init__()

    self.nx_graph = nx_graph
    self.pyg_graph = pyg_graph
    self.num_agents = num_agents
    self.num_nodes = self.pyg_graph.num_nodes-self.num_agents
    # self.batch_size = batch_size

    # self.node_hidden_feat_dimension = 16
    # self.num_of_hops = 4
    self.num_max_actions = int(
        self.num_agents * max([d-1 for n, d in self.nx_graph.degree()]))
    self.num_node_features = len(self.nx_graph.nodes[0]['feature'])
    self.num_agent_features = len(self.nx_graph.nodes[\
                                      self.num_nodes]['agent_feature'])

    # print(f"#####{self.nx_graph.degree()}max_out_degree: { \
    # max([d-1 for n, d in self.nx_graph.degree()])}#####, \
    # self.num_max_actions: {self.num_max_actions}")

    # print(f"num_node_feat: {self.num_node_features}")

    self.cnn_layer_1_dimension = 512 #512 #
    self.cnn_layer_2_dimension = 512
    self.cnn_layer_3_dimension = 128 # 4096 #1 #
    self.cnn_layer_4_dimension = 32

    self.state_layer_1_dim = 512
    self.state_layer_2_dim = 512
    self.state_layer_3_dim = 256
    self.state_layer_4_dim = 64

    self.encode_hidden_size = 256

    self.decode_output_size = 1

    self.encoder = EncoderRNN(input_size=self.num_node_features,
                              hidden_size=self.encode_hidden_size)

    self.decoder = AttnDecoderRNN(hidden_size=self.encode_hidden_size,
                                  output_size=self.decode_output_size)

    temp_kernel_size = 2 * self.encode_hidden_size
    stride_ = 2 * self.num_nodes * (2 * self.encode_hidden_size)

    self.state_layer_1 = nn.Conv1d(in_channels=1,
                                   out_channels=self.state_layer_1_dim,
                                   kernel_size=temp_kernel_size,
                                   stride=stride_)

    temp_kernel_size = self.state_layer_1_dim
    stride_ = self.state_layer_1_dim

    self.state_layer_2 = nn.Conv1d(in_channels=1,
                                   out_channels=self.state_layer_2_dim,
                                   kernel_size=temp_kernel_size,
                                   stride=stride_)

    temp_kernel_size = self.state_layer_2_dim
    stride_ = self.state_layer_2_dim

    self.state_layer_3 = nn.Conv1d(in_channels=1,
                                   out_channels=self.state_layer_3_dim,
                                   kernel_size=temp_kernel_size,
                                   stride=stride_)

    temp_kernel_size = self.state_layer_3_dim
    stride_ = self.state_layer_3_dim

    self.state_layer_4 = nn.Conv1d(in_channels=1,
                                   out_channels=self.state_layer_4_dim,
                                   kernel_size=temp_kernel_size,
                                   stride=stride_)

    temp_kernel_size = 1
    stride_ = self.state_layer_4_dim

    self.state_layer_out = nn.Conv1d(in_channels=1,
                                     out_channels=1,
                                     kernel_size=temp_kernel_size,
                                     stride=stride_)

    temp_kernel_size = self.decode_output_size
                       # self.num_node_features * (self.num_nodes + 1)

    self.cnn_layer_1 = nn.Conv1d(in_channels=1,
                                 out_channels=self.cnn_layer_1_dimension, \
                                 kernel_size=temp_kernel_size,
                                 stride=temp_kernel_size)

    temp_kernel_size = self.cnn_layer_1_dimension

    self.cnn_layer_2 = nn.Conv1d(in_channels=1,
                                 out_channels=self.cnn_layer_2_dimension, \
                                 kernel_size=temp_kernel_size,
                                 stride=temp_kernel_size)

    temp_kernel_size = self.cnn_layer_2_dimension 

    self.cnn_layer_3 = nn.Conv1d(in_channels=1,
                                 out_channels=self.cnn_layer_3_dimension, \
                                 kernel_size=temp_kernel_size,
                                 stride=temp_kernel_size)

    temp_kernel_size = self.cnn_layer_3_dimension

    self.cnn_layer_4 = nn.Conv1d(in_channels=1,
                                 out_channels=self.cnn_layer_4_dimension, \
                                 kernel_size=temp_kernel_size,
                                 stride=temp_kernel_size)

    temp_kernel_size = self.cnn_layer_4_dimension

    self.cnn_layer_out = nn.Conv1d(in_channels=1,
                                   out_channels=1, \
                                   kernel_size=temp_kernel_size,
                                   stride=temp_kernel_size)

  # Called with either one element to determine next action, or a batch
  # during optimization. Returns tensor([[left0exp,right0exp]...]).
  def forward(self, input_list):

    # print(f"inp.size(): {inp.size()}")

    self.state_seq = input_list[0]
    self.neighbour_seq = input_list[1]
    self.neighbour_list = input_list[2]
    inp = self.state_seq

    self.encoder_out, self.encoder_hidden = self.encoder(self.state_seq)

    self.decoder_inp_seq = None

    for batch_ind, sub_list in enumerate(self.neighbour_list):
      temp_seq = None
      for neigh in sub_list:
        if neigh != -1:
          if temp_seq is None:
            temp_seq = self.encoder_out[batch_ind][:][neigh].view(1,\
                                                  2*self.encode_hidden_size)

          else:
            temp_seq = torch.cat((temp_seq, \
                                self.encoder_out[batch_ind][:][neigh].view(1,\
                                        2 * self.encode_hidden_size)), dim=0)

        else:
          if temp_seq is None:
            temp_seq = torch.zeros(1, 2*self.encode_hidden_size)

          else:
            temp_seq = torch.cat((temp_seq, torch.zeros(1, \
                                      2*self.encode_hidden_size)), dim=0)

      if self.decoder_inp_seq is None:
        self.decoder_inp_seq = temp_seq.view(1, self.num_max_actions,
                                             2 * self.encode_hidden_size)

      else:
        self.decoder_inp_seq = torch.cat((self.decoder_inp_seq, temp_seq.view(\
                  1, self.num_max_actions, 2*self.encode_hidden_size)), dim=0) 

    self.decoder_out, self.context = self.decoder(self.encoder_out,
                                                  self.encoder_hidden,
                                                  self.decoder_inp_seq)

    self.state_val = F.relu(self.state_layer_1(self.context)).flatten()

    self.state_val = self.state_layer_2(self.state_val.view(inp.size(dim=0),\
                                        1, self.state_layer_1_dim))

    self.state_val = F.relu(self.state_layer_3(self.state_val.view(\
                    inp.size(dim=0), 1, self.state_layer_2_dim))).flatten()

    self.state_val = self.state_layer_4(self.state_val.view(inp.size(dim=0),\
                                        1, self.state_layer_3_dim)).flatten()

    self.state_val = self.state_layer_out(self.state_val.view(\
                inp.size(dim=0), 1, self.state_layer_4_dim)).flatten().view(\
                                              inp.size(dim=0), 1)

    self.a_vals = self.decoder_out.flatten().view(inp.size(dim=0),
                                                  self.num_max_actions)

    # self.a_vals_0 = self.decoder_out.flatten().view(inp.size(dim=0), 1,\
    #                           self.decode_output_size * self.num_max_actions)
    
    # self.a_vals_1 = F.relu(self.cnn_layer_1(self.a_vals_0)).flatten()

    # self.a_vals_2 = self.cnn_layer_2(self.a_vals_1.view(\
    #                                  inp.size(dim=0), 1, \
    #                                  self.cnn_layer_1_dimension * \
    #                                  self.num_max_actions)).flatten()

    # self.a_vals_3 = F.relu(self.cnn_layer_3(self.a_vals_2.view(\
    #                                   inp.size(dim=0), 1, \
    #                                   self.cnn_layer_2_dimension * \
    #                                   self.num_max_actions))).flatten()

    # self.a_vals_4 = self.cnn_layer_4(self.a_vals_3.view(\
    #                                  inp.size(dim=0), 1, \
    #                                  self.cnn_layer_3_dimension * \
    #                                  self.num_max_actions)).flatten()

    # self.a_vals = self.cnn_layer_out(self.a_vals_4.view(\
    #                                  inp.size(dim=0), 1, \
    #                                  self.cnn_layer_4_dimension * \
    #                                  self.num_max_actions)).flatten().view(\
    #                                                     inp.size(dim=0),\
    #                                                     self.num_max_actions)

    self.average_a_vals = (self.a_vals.mean(1).view(inp.size(dim=0),\
                                   1).repeat(1, self.num_max_actions).view(\
                                      inp.size(dim=0), self.num_max_actions))

    self.repeated_state_vals = self.state_val.repeat(1, \
                               self.num_max_actions).view(inp.size(dim=0),\
                                                        self.num_max_actions)

    self.q_vals = self.repeated_state_vals + self.a_vals - self.average_a_vals

    self.q_vals = self.q_vals.view(inp.size(dim=0), self.num_max_actions)

    return self.q_vals
