'''Module for the implementation of critic network in pytorch'''

import numpy as np

#import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear

import networkx as nx

#from torch_geometric.data import Data
#from torch_geometric.data import Batch
#from torch_geometric.utils import add_remaining_self_loops

class TorchSingleAgentCritic(nn.Module):
    '''PyTorch implementation of the Critic for PPO'''
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

        self.adj_matrix = nx.adjacency_matrix(nx_graph).todense()
        # add self loops
        #for i in range(self.num_nodes):
        #    self.adj_matrix[i,i] = 1

        self.dqn_lin_0_out_features = 512
        self.dqn_lin_1_out_features = 1024
        self.dqn_lin_2_out_features = 128
        self.dqn_lin_3_out_features = 1
        self.dqn_lin_4_out_features = 1
        #self.dqn_lin_5_out_features = 16
        #self.dqn_lin_6_out_features = self.num_nodes
        # self.lin_0_input_size = self.num_nodes*(
        #                         self.num_nodes + num_features)
        self.lin_0_input_size = self.num_nodes*(
                                1 + num_features)

        self.dqn_lin_0 = Linear(
                in_features=self.lin_0_input_size,
                out_features=self.dqn_lin_0_out_features,
                )

        self.dqn_lin_1 = Linear(
                in_features=self.dqn_lin_0_out_features,
                out_features=self.dqn_lin_1_out_features
                )
        self.dqn_lin_2 = Linear(
                in_features=self.dqn_lin_1_out_features,
                out_features=self.dqn_lin_2_out_features
                )
        self.dqn_lin_3 = Linear(
                in_features=self.dqn_lin_2_out_features,
                out_features=self.dqn_lin_3_out_features
                )
        self.dqn_lin_4 = Linear(
                in_features=self.dqn_lin_3_out_features,
                out_features=self.dqn_lin_4_out_features
                )
        self.dqn_lin_out = Linear(
                in_features=self.dqn_lin_4_out_features,
                out_features=1
                )
        #self.dqn_lin_5 = Linear(
        #        in_features=self.dqn_lin_4_out_features,
        #        out_features=self.dqn_lin_5_out_features
        #        )
        #self.dqn_lin_6 = Linear(
        #        in_features=self.dqn_lin_5_out_features,
        #        out_features=self.dqn_lin_6_out_features
        #        )




    def forward(self, critic_input):
        batch_size = critic_input.shape[0]
        #pyg, batch_size = self._convert_obs_to_pyg(obs)
        critic_input = critic_input.reshape(-1, self.lin_0_input_size)

        x = F.relu(self.dqn_lin_0(critic_input))
        x = F.sigmoid(self.dqn_lin_1(x))
        x = F.relu(self.dqn_lin_2(x))
        # add the time to go here
        #list_of_time_to_go =[obs[i]['time_to_go'] for i in range(len(obs))]
        #time_to_go_tensor = torch.tensor(list_of_time_to_go).reshape(-1,1)
        #x = torch.cat([x, time_to_go_tensor], dim=1)
        x = self.dqn_lin_3(x)
        x = self.dqn_lin_4(x)
        x = self.dqn_lin_out(x)
        #x = self.dqn_lin_4(x)
        values = x.reshape(batch_size, -1, 1)

        return values


################################################################################
################################################################################
################################################################################


class CriticUtils():
    '''Class for the critic related utils'''
    def __init__(self, **kwargs):
        assert (
                'graph_adjacency' in kwargs
            ), 'Missing adjacency information in custom model config'
        assert (
                'nx_graph' in kwargs
            ), 'Missing Initial NetworkX graph'

        self.graph_adjacency = kwargs['graph_adjacency']
        self.nx_graph = kwargs['nx_graph']
        self.shortest_paths = dict(
                    nx.all_pairs_shortest_path(self.nx_graph)
                )
        self.num_nodes = len(self.nx_graph.nodes.data())
        # num_features = len(nx_graph.nodes[0])

        self.adj_matrix = nx.adjacency_matrix(self.nx_graph).todense()
        # add self loops
        #for i in range(self.num_nodes):
        #    self.adj_matrix[i,i] = 1

    def create_input(self, ep_states):
        '''Function to create the input to the cnn containing all actions'''
        normalize = False
        if not isinstance(ep_states[0], list):
            ep_states = [ep_states]
        ep_states_input = []
        for ep in ep_states:
            ep_inputs = []
            for state in ep:
                graph, _ = self._convert_obs_to_graph(state, normalize)
                graph = np.array(graph, dtype=np.float32)
                ep_inputs.append(graph)
            ep_states_input.append(ep_inputs)
        critic_input = np.array(ep_states_input, dtype=np.float32)

        return critic_input

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
        #adj_matrix = self.adj_matrix
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
                                self.shortest_paths[i][curr_occ_node]
                                )])
                if normalize:
                    node_space[i][0][0] /= max_node_dem
                    node_space[i][0][0] *= 35
                graph_array = np.append(graph_array, shortest_path_length)
                graph_array = np.append(graph_array, node_space[i])

            #graph_tensor = torch.tensor(graph_array, dtype=torch.float32)
            list_of_graphs.append(graph_array)

        list_of_graphs = np.vstack(list_of_graphs)

        #return torch.tensor(list_of_graphs, dtype=torch.float32), batch_size
        return list_of_graphs[0]#, batch_size

    def _get_travel_time_along_path(self, path):
        '''Function to get the travel time along a given path'''
        total_travel_time = 0.0
        for i in range(len(path)-1):
            s = path[i]
            t = path[i+1]
            total_travel_time += self.nx_graph.edges[(s,t)]['travel_time'][0]
        return total_travel_time
