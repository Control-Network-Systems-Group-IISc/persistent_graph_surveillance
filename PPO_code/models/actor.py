'''Module for the implementation of actor network in pytorch'''

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv1d

import networkx as nx

#from torch_geometric.data import Data
#from torch_geometric.data import Batch
#from torch_geometric.utils import add_remaining_self_loops

class TorchSingleAgentActor(nn.Module):
    '''PyTorch implementation of the Actor for PPO'''
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        assert (
                'graph_adjacency' in kwargs
            ), 'Missing adjacency information in custom model config'
        assert (
                'nx_graph' in kwargs
            ), 'Missing Initial NetworkX graph'

        self.graph_adjacency = kwargs['graph_adjacency']
        nx_graph = kwargs['nx_graph']
        self.num_nodes = len(nx_graph.nodes.data())
        num_features = len(nx_graph.nodes[0])

        self.nx_graph = nx_graph

        self.adj_matrix = nx.adjacency_matrix(nx_graph).todense()
        # add self loops
        #for i in range(self.num_nodes):
        #    self.adj_matrix[i,i] = 1

        self.conv1d_layer_1_out_features = 512
        self.conv1d_layer_2_out_features = 1024
        self.conv1d_layer_3_out_features = 128
        self.conv1d_layer_4_out_features = 64

        # self.conv_layer_1_kernel_size = self.num_nodes*(self.num_nodes
        #                             + num_features) + 2*num_features
        self.conv_layer_1_kernel_size = self.num_nodes*(1+ num_features)\
                                        + 1*(num_features + 1)

        self.conv1d_layer_1 = Conv1d(
                in_channels=1, out_channels=self.conv1d_layer_1_out_features,
                kernel_size=self.conv_layer_1_kernel_size,
                stride=self.conv_layer_1_kernel_size,
                )
        self.conv1d_layer_2 = Conv1d(
                in_channels=1, out_channels=self.conv1d_layer_2_out_features,
                kernel_size=self.conv1d_layer_1_out_features,
                stride=self.conv1d_layer_1_out_features,
                )
        self.conv1d_layer_3 = Conv1d(
                in_channels=1, out_channels=self.conv1d_layer_3_out_features,
                kernel_size=self.conv1d_layer_2_out_features,
                stride=self.conv1d_layer_2_out_features,
                )
        self.conv1d_layer_4 = Conv1d(
                in_channels=1, out_channels=self.conv1d_layer_4_out_features,
                kernel_size=self.conv1d_layer_3_out_features,
                stride=self.conv1d_layer_3_out_features,
                )
        self.conv1d_layer_out = Conv1d(
                in_channels=1, out_channels=1,
                kernel_size=self.conv1d_layer_4_out_features,
                stride=self.conv1d_layer_4_out_features,
                )


    def forward(self, conv_input, mask):
        batch_size_orig = conv_input.shape[0]
        conv_input = conv_input.reshape(-1, 1,
                    self.num_nodes*self.conv_layer_1_kernel_size)
        batch_size = conv_input.shape[0]
        x = F.relu(self.conv1d_layer_1(conv_input))
        x = x.reshape(batch_size, 1, -1)
        x = F.sigmoid(self.conv1d_layer_2(x))
        # x = x.reshape(1, -1)
        # add the time_to_go feature here
        #time_to_go_tensor =torch.tensor([obs['time_to_go']]*x.shape[1])
        #time_to_go_tensor = time_to_go_tensor.reshape(1, -1)
        #x = torch.cat([x, time_to_go_tensor])
        x = x.reshape(batch_size, 1,-1)
        x = F.relu(self.conv1d_layer_3(x))
        x = x.reshape(batch_size, 1, -1)
        x = self.conv1d_layer_4(x)
        x = x.reshape(batch_size, 1, -1)
        attn = self.conv1d_layer_out(x)
        attn = attn.reshape(batch_size, 1, -1)

        mask = mask.reshape(batch_size, 1, -1)
        inf_tensor = torch.zeros_like(attn)
        inf_tensor[mask] = -torch.inf
        attn = attn + inf_tensor
        attn = attn.reshape(batch_size_orig, -1, self.num_nodes)
        return attn


################################################################################
################################################################################
################################################################################


class ActorUtils():
    '''Class for the actor related utils'''
    def __init__(self, **kwargs):
        assert (
                'graph_adjacency' in kwargs
            ), 'Missing adjacency information in custom model config'
        assert (
                'nx_graph' in kwargs
            ), 'Missing Initial NetworkX graph'

        self.graph_adjacency = kwargs['graph_adjacency']
        nx_graph = kwargs['nx_graph']
        self.num_nodes = len(nx_graph.nodes.data())
        # num_features = len(nx_graph.nodes[0])

        self.nx_graph = nx_graph
        self.shortest_paths = dict(
                    nx.all_pairs_shortest_path(self.nx_graph)
                )

        self.adj_matrix = nx.adjacency_matrix(nx_graph).todense()
        # add self loops
        #for i in range(self.num_nodes):
        #    self.adj_matrix[i,i] = 1

    def create_conv_input(self, ep_states):
        '''Function to create the input to the cnn containing all actions'''
        normalize = False
        if not isinstance(ep_states[0], list):
            ep_states = [ep_states]
        ep_states_conv_input = []
        ep_states_mask = []
        for ep in ep_states:
            ep_conv_inputs = []
            ep_mask = []
            for state in ep:
                graph, max_node_dem = self._convert_obs_to_graph(state,
                                        normalize)
                agent_loc = self._get_agent_location(state)
                non_neighbors = self._get_non_neighbours_mask(agent_loc)
                conv_input = np.array([], dtype=np.float32)
                for node_idx in range(self.num_nodes):
                    # temp = [*graph,
                    #         *state['node_space'][agent_loc][0],
                    #         *state['node_space'][node_idx][0],
                    #         ]
                    node_features = state['node_space'][node_idx][0]
                    if normalize:
                        node_features[0] /= max_node_dem
                        node_features[0] *= 35
                    temp = [*graph,
                            self._get_travel_time_along_path(
                                self.shortest_paths[agent_loc][node_idx]),
                            *node_features,
                            ]
                    temp = np.array(temp)
                    conv_input = np.append(conv_input, temp)
                ep_conv_inputs.append(conv_input)
                ep_mask.append(non_neighbors)
            ep_states_conv_input.append(ep_conv_inputs)
            ep_states_mask.append(ep_mask)
        conv_input = np.array(ep_states_conv_input, dtype=np.float32)
        non_neighbors_mask = np.array(ep_states_mask)

        return conv_input, non_neighbors_mask

    def _get_non_neighbours_mask(self, agent_loc):
        neighbors = self.adj_matrix[agent_loc,:]
        non_neighbors = np.ones_like(neighbors) - neighbors
        non_neighbors = np.where(non_neighbors == 1)[0]
        mask = np.zeros_like(neighbors)
        mask[non_neighbors] = 1
        mask = np.array(mask, dtype=bool)

        return mask

    def _get_agent_location(self, state):
        node_space = state['node_space']
        #edge_space = state['edge_space']
        agent_locs = []
        for node in range(self.num_nodes):
            loc_feature = node_space[node][0][1]
            if loc_feature == 1:
                agent_locs.append(node)
        if len(agent_locs) > 1:
            raise ValueError('More than one agent detected')
        if len(agent_locs) == 0:
            raise ValueError('No agents were detected')

        return agent_locs[0]

    def _convert_obs_to_graph(self, obs, normalize=False):
        if not isinstance(obs, list):
            obs = [obs]
        #batch_size = len(obs)
        #adj_matrix = nx.adjacency_matrix(self.nx_graph).todense()
        # add self loops
        #for i in range(self.num_nodes):
        #    adj_matrix[i,i] = 1

        list_of_graphs = []
        for i in range(len(obs)):
            node_space = obs[i]['node_space']
            #edge_space = obs[i]['edge_space']

            graph_array = np.array([], dtype=np.float32)
            curr_occ_node = self._get_agent_location(obs[i])
            max_node_dem = -np.inf
            for i in range(self.num_nodes):
                max_node_dem = max(max_node_dem, node_space[i].squeeze()[0])
            for i in range(self.num_nodes):
                # graph_array = np.append(graph_array, adj_matrix[i,:])
                shortest_path_length = np.array(
                            [self._get_travel_time_along_path(
                                self.shortest_paths[i][curr_occ_node])]
                                )
                if normalize:
                    node_space[i][0][0] /= max_node_dem
                    node_space[i][0][0] *= 35
                graph_array = np.append(graph_array, shortest_path_length)
                graph_array = np.append(graph_array, node_space[i])

            #graph_tensor = torch.tensor(graph_array, dtype=torch.float32)
            list_of_graphs.append(graph_array)

        list_of_graphs = np.vstack(list_of_graphs)

        #return torch.tensor(list_of_graphs, dtype=torch.float32), batch_size
        return list_of_graphs[0], max_node_dem

    def _get_travel_time_along_path(self, path):
        '''Function to get the travel time along a given path'''
        total_travel_time = 0.0
        for i in range(len(path)-1):
            s = path[i]
            t = path[i+1]
            total_travel_time += self.nx_graph.edges[(s,t)]['travel_time'][0]
        return total_travel_time

