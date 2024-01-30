'''Module for the gym environment for a single agent'''
import copy
import gymnasium as gym
from gymnasium import spaces

import networkx as nx
import numpy as np

import graph_utils
import random


class SingleAgentEnv(gym.Env):
    '''Class for the single agent environment'''

    def __init__(self, config):
        assert (
                'nx_graph' in config
                and isinstance(config.get('nx_graph'), type(nx.Graph()))
                ), 'NetworkX graph is not present'
        assert (
                'graph_adjacency' in config
                ), 'Missing Graph adjacency in config'

        self.nx_graph:nx.classes.graph.Graph = config.get('nx_graph')
        self.orig_nx_graph:nx.classes.graph.Graph = copy.deepcopy(
                                config.get('nx_graph')
                )
        self.graph_adjacency = np.array(config.get('graph_adjacency'))
        self.steps_per_episode = (
                config.get('steps_per_episode')
                    if 'steps_per_episode' in config else 15
                )
        self.station_flag = True
        if 'station_flag' in config:
            self.station_flag = config.get('station_flag')

        # action space
        #self.action_space = spaces.Discrete(self.nx_graph.number_of_nodes())
        num_nodes = self.nx_graph.number_of_nodes()
        num_edges = self.nx_graph.number_of_edges()
        self.action_space = spaces.Discrete(num_nodes)

        # get the features of the first node and edge
        node_features = list(self.nx_graph.nodes.data())[0][1]
        edge_features = list(self.nx_graph.edges.data())[0][2]
        # calculate the number of nodes and edges
        num_node_features = sum((len(node_features[x]) for x in node_features))
        num_edge_features = sum((len(edge_features[x]) for x in edge_features))

        # Observation is a graph
        self.observation_space = spaces.Dict(
                {
                    'node_space': spaces.Tuple([
                        spaces.Box(
                            low=-np.inf, high=np.inf, shape=(num_node_features,)
                        ) for _ in range(num_nodes)
                        ]
                    ),
                    'edge_space': spaces.Tuple([
                        spaces.Box(
                            low=-np.inf, high=np.inf, shape=(num_edge_features,)
                        ) for _ in range(num_edges)
                        ]
                    ),
                    'time_to_go': spaces.Tuple([spaces.Box(
                            low=-np.inf, high=np.inf
                        )])
                }
            )

        # store the simulation time
        self.timer = 0

        # lists to store demand histories for plotting
        self.list_of_ep_demands = []
        self.total_demand_list = []

        #list of station nodes
        self.station_nodes = []


    #def reset(self, *, seed=None, options=None):
    def reset(self, random_demands=True, **kwargs):
        # create a random observation
        #print('RESET!!!!!!!!!!!!!!!!!!!!')
        #obs = self.observation_space.sample()
        if 'nx_graph' in kwargs:
            nx_graph = kwargs.get('nx_graph')
            for node in self.nx_graph.nodes.data():
                node_idx = node[0]
                node_features = node[1]
                if self.station_flag:
                    station_node = node_features['station'][0]
                else:
                    station_node = False
                if station_node:
                    node_features['occupied'] = [False]
                    node_features['demand'] = [0.]
                    node_features['priority'] = [0]
                    node_features['station'] = [True]
                else:
                    node_features['occupied'] = [False]
                    node_features['demand'] = [
                            nx_graph.nodes[node_idx]['demand'][0]
                            ]
                    node_features['priority'] = [
                            nx_graph.nodes[node_idx]['priority'][0]
                            ]
                node_features['occupied'] = [
                            nx_graph.nodes[node_idx]['occupied'][0]
                        ]
                node_features['time_to_go'] = [self.steps_per_episode]
                if self.station_flag:
                    node_features['stn_dem'] = [
                                nx_graph.nodes[node_idx]['stn_dem'][0]
                                ]

                self.nx_graph.add_node(node_idx, **node_features)
        elif 'non_uniform' in kwargs:
            if self.station_flag:
                stn_dem, occ_node = self._reset_stn_dem()
                self.nx_graph = graph_utils.reset_graph_non_uniform(
                        self.nx_graph, self.steps_per_episode,
                        stn_dem, occ_node)
            else:
                occ_node = random.choice(list(self.nx_graph.nodes()))
                self.nx_graph = graph_utils.reset_graph_non_uniform_no_stn(
                        self.nx_graph, self.steps_per_episode, occ_node,
                        random_demands=True, random_priorities=False,
                        )
        else:
            #for node in self.nx_graph.nodes.data():
            #    node_idx = node[0]
            #    node_features = node[1]
            #    station_node = node_features['station'][0]
            #    if station_node:
            #        node_features['occupied'] = [True]
            #        node_features['demand'] = [0.]
            #    else:
            #        node_features['occupied'] = [False]
            #        node_features['demand'] = [
            #            random.randint(0,
            #            int(self.steps_per_episode/2)
            #            ) if random_demands else 0
            #            #0.
            #                ]
            #    node_features['time_to_go'] = [self.steps_per_episode]
            #    self.nx_graph.add_node(node_idx, **node_features)
            self.nx_graph = graph_utils.reset_graph(
                    graph=self.nx_graph,
                    max_rand_dem=int(self.steps_per_episode/2)\
                                    if random_demands else 0,
                    max_rand_priority=4,
                    time_to_go=self.steps_per_episode,
                    ref_graph=None if random_demands else self.orig_nx_graph)

        obs = self._convert_nx_to_graph()
        #print('-----------------------------------------------------------')
        self.timer = 0
        self._update_total_demand()
        self.list_of_ep_demands = []
        return obs, {}

    def step(self, action):
        #print(f'Action: {action}', end='\t')
        #self.timer += 1
        #obs = self.observation_space.sample()
        action_valid, agent_loc = self._valid_action(action)
        travel_time = self.nx_graph.edges[(agent_loc, action)]['travel_time'][0]
        #print(travel_time)
        reward = 0
        for t in range(travel_time):
            self.timer += 1
            stn_dem_0_flag = False
            if action in self.station_nodes:
                stn_dem_0_flag = True
            for node in self.nx_graph.nodes.data():
                node_idx = node[0]
                node_features = node[1]
                # if action is valid update agent location
                # else remain in same place
                occupied = node_features['occupied'][0]
                #if occupied:
                #    print(f'Current location = {node_idx}')
                if self.station_flag:
                    station_node = node_features['station'][0]
                else:
                    station_node = False
                if action_valid:
                    if node_idx == action and t == travel_time-1:
                        node_features['occupied'] = [True]
                        node_features['demand'] = [0] # zero demand
                    else:
                        node_features['occupied'] = [False]
                        node_features['demand'][0] += 1
                else: # remanin in same place
                    if not occupied:
                        node_features['demand'][0] += 1
                    else:
                        node_features['demand'] = [0]
                # station node demands remain zero
                if station_node:
                    node_features['demand'] = [0.]
                # zero priority nodes have zero demand
                if node_features['priority'][0] == 0:
                    node_features['demand'] = [0.]
                node_features['time_to_go'] = \
                                    [self.steps_per_episode-self.timer]
                if self.station_flag:
                    if stn_dem_0_flag:
                        node_features['stn_dem'] = [0]
                    else:
                        node_features['stn_dem'][0] += 1
                self.nx_graph.add_node(node_idx, **node_features)
            reward += self._calc_reward()
            if self.timer == self.steps_per_episode:
                break

        obs = self._convert_nx_to_graph()
        #reward = self._calc_reward()
        #print(reward)
        # one step is one second in single agent env
        terminated = self.timer >= self.steps_per_episode

        return obs, reward, terminated, False, {}

    def _calc_reward(self):
        node_features = self.nx_graph.nodes.data()
        sum_of_demands = 0.0
        for node in node_features:
            sum_of_demands += node[1]['demand'][0] * node[1]['priority'][0]
        if self.station_flag:
            sum_of_demands += self.nx_graph.nodes[0]['stn_dem'][0]
        self.list_of_ep_demands.append(sum_of_demands)
        return -sum_of_demands


    def _valid_action(self, action):
        edge_index = np.ascontiguousarray(self.graph_adjacency.T)
        #print(edge_index)
        for node in self.nx_graph.nodes.data():
            node_idx = node[0]
            node_features = node[1]
            neighbor_idx = edge_index[1][edge_index[0]==node_idx]
            # allow self loop
            neighbor_idx = np.append(neighbor_idx,node_idx)
            if(node_features['occupied'] == [True] and action in neighbor_idx):
                return True, node_idx
        print('Action is invalid', action)
        #raise Exception('Invalid action')
        return False, -1

    def _convert_nx_to_graph(self):
        list_of_node_features = []
        for node in self.nx_graph.nodes.data():
            temp_list = np.array([], dtype=np.float32)
            node_features = node[1]
            for _, feature_vector in node_features.items():
                temp_list = np.append(temp_list,
                        np.array(feature_vector, dtype=np.float32))
            list_of_node_features.append(np.array(temp_list,
                        dtype=np.float32).reshape(1,-1))

        list_of_edge_features = []
        for edge in self.nx_graph.edges.data():
            temp_list = np.array([], dtype=np.float32)
            edge_features = edge[2]
            for _, feature_vector in edge_features.items():
                temp_list = np.append(temp_list,
                        np.array(feature_vector, dtype=np.float32))
            list_of_edge_features.append(np.array(temp_list,
                        dtype=np.float32).reshape(1,-1))

        return {'node_space': list_of_node_features,
                'edge_space': list_of_edge_features,
                'time_to_go': [self.steps_per_episode - self.timer]}

    def _reset_stn_dem(self):
        path_lens = dict(nx.all_pairs_shortest_path_length(self.nx_graph))
        stn_dem_low = np.inf
        occupied_node = random.randint(0, self.nx_graph.number_of_nodes()-1)
        self.station_nodes = []
        for node in self.nx_graph.nodes:
            if self.nx_graph.nodes[node]['station'][0]:
                stn_dem_low = min(stn_dem_low, path_lens[occupied_node][node])
                self.station_nodes.append(node)
        stn_dem = random.randint(int(stn_dem_low), 3*self.steps_per_episode)

        return stn_dem, occupied_node

    def _update_total_demand(self):
        if len(self.list_of_ep_demands) > 0:
            self.total_demand_list.append(
                    sum(self.list_of_ep_demands)
                    )
