'''Module to implement the heuristics'''
import numpy as np
import torch

import networkx as nx

from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import to_networkx

def greedy(obs, graph_adjacency, num_nodes):
    pyg, batch_size = _convert_obs_to_pyg(obs, graph_adjacency, num_nodes)
    agent_loc_idx = _get_agent_location(pyg, batch_size, num_nodes)

    nx_graph = to_networkx(pyg)
    adj_matrix = nx.adjacency_matrix(nx_graph).todense()

    for i in range(num_nodes):
        adj_matrix[i,i] = 1

    neighbours = adj_matrix[agent_loc_idx[0].detach().numpy(),:][0]
    neighbours = np.where(neighbours==1)[0]
    max_val = [-np.inf, -1]
    for n in neighbours:
        node_data = obs['node_space'][n][0]
        # get product of demand and priority
        prod = node_data[2]*node_data[3]
        if prod > max_val[0]:
            max_val = [prod, n]

    return max_val[1]


def _get_agent_location(pyg, batch_size, num_nodes):
    node_features = pyg.x
    node_features = node_features.reshape(
            batch_size, -1, pyg.num_node_features)
    agent_loc_idx = []
    for i in range(batch_size):
        instance = node_features[i]
        # get the agent location which is the second feature
        loc_feature = instance[:,1]
        agent_idx = torch.tensor(range(num_nodes))[loc_feature == 1]
        agent_loc_idx.append(agent_idx)

    return agent_loc_idx


def _convert_obs_to_pyg(obs, graph_adjacency, num_nodes):
    if not isinstance(obs, list):
        obs = [obs]
    batch_size = len(obs)
    list_of_graphs = []
    for i in range(len(obs)):
        node_space = obs[i]['node_space']
        edge_space = obs[i]['edge_space']

        edge_index = torch.tensor(graph_adjacency, dtype=torch.long)
        node_space_t = torch.tensor(
                np.array(node_space), dtype=torch.float32
                )
        edge_space_t = torch.tensor(
                np.array(edge_space), dtype=torch.float32
                )

        node_space_t = node_space_t.reshape(num_nodes, -1)
        edge_space_t = edge_space_t.reshape(len(graph_adjacency), -1)

        graph = Data(x=node_space_t,
                edge_index=edge_index.t().contiguous(),
                edge_attr=edge_space_t)
        looped_edge_index, looped_edge_attr = add_remaining_self_loops(
                edge_index=graph.edge_index, edge_attr=graph.edge_attr)
        graph = Data(x=node_space_t,
                edge_index=looped_edge_index, edge_attr=looped_edge_attr)
        graph.validate(raise_on_error=True)

        list_of_graphs.append(graph)

    if len(list_of_graphs) == 1:
        return list_of_graphs[0], batch_size
    else:
        return Batch.from_data_list(list_of_graphs), batch_size


