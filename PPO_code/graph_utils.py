'''Module for the graph related functions'''
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import copy
import random

#def prepare_graph(graph_adj, n_nodes=5):
#    nx_g = nx.DiGraph()
#
#    nx_g.add_nodes_from(
#            [
#                (i, {
#                    'demand': [0.],
#                    'occupied': [False],
#                    'station': [False],
#                    'priority': [1],
#                    }
#                ) for i in range(n_nodes)
#            ]
#        )
#    nx_g.add_edges_from(
#            [
#                (
#                    *x, {
#                        'travel_time': [1.],
#                        }
#                ) for x in graph_adj
#            ]
#        )
#    nx_g.nodes[0]['station'] = [True]
#    nx_g.nodes[0]['occupied'] = [True]
#    nx_g = initiate_node_demands(nx_g)
#
#    return nx_g

def gen_graph_and_agents(num_nodes, num_stations, num_connects_per_node,
                         prob_of_rewiring):
    graph = nx.connected_watts_strogatz_graph(
            n=num_nodes, k=num_connects_per_node, p=prob_of_rewiring)
    graph = graph.to_directed()

    graph_opt = copy.deepcopy(graph)

    nx.set_node_attributes(graph, [False], 'station')
    nx.set_node_attributes(graph, [False], 'occupied')
    nx.set_node_attributes(graph, [np.double(0)], 'demand')
    nx.set_node_attributes(graph, [1], 'priority')
    nx.set_node_attributes(graph, [0], 'time_to_go')

    # setting values for integer graph
    nx.set_node_attributes(graph_opt, False, 'station')
    nx.set_node_attributes(graph_opt, False, 'occupied')
    nx.set_node_attributes(graph_opt, np.double(0), 'demand')
    nx.set_node_attributes(graph_opt, 1, 'priority')


    #nx.set_node_attributes(graph, [0 for _ in range(num_nodes)], 'one_hot_loc')

    for edge in graph.edges():
        # np.random.choice(range(2, 4))
        graph[edge[0]][edge[1]]['travel_time'] = [1]
        graph_opt[edge[0]][edge[1]]['travel_time'] = 1

    station_nodes_list = np.random.choice(graph.nodes(), size=num_stations,
                                        replace=False)

    for station_node in station_nodes_list:
        graph.nodes[station_node]['station'] = [True]
        graph.nodes[station_node]['occupied'] = [True]

        graph_opt.nodes[station_node]['station'] = True
        graph_opt.nodes[station_node]['occupied'] = True

    #for i in range(num_nodes):
    #    graph.nodes[i]['one_hot_loc'] = list(np.eye(num_nodes)[i,:])

    graph, graph_opt = initiate_node_demands(graph, graph_opt)

    return graph, graph_opt, list(graph.edges)

def initiate_node_demands(graph, graph_opt, max_rand_dem=0,
        random_demands=False):
    for node in graph.nodes():
        if graph.nodes[node]['station'][0]: # if station node
            graph.nodes[node]['priority'] = [0]
            graph_opt.nodes[node]['priority'] = 0
            graph.nodes[node]['demand'] = [0]
            graph_opt.nodes[node]['demand'] = 0
        else:
            if random_demands:
                rand_dem = random.randint(0, max_rand_dem)
                graph.nodes[node]['demand'] = [rand_dem]
                graph_opt.nodes[node]['demand'] = rand_dem
            else:
                graph.nodes[node]['demand'] = [0]
                graph_opt.nodes[node]['demand'] = 0
            rand_priority = random.randint(1, 4)
            graph.nodes[node]['priority'] = [rand_priority]
            graph_opt.nodes[node]['priority'] = rand_priority

    return graph, graph_opt

def reset_graph(graph, max_rand_dem=0, max_rand_priority=1,
                        time_to_go=10, ref_graph=None):
    for node in graph.nodes():
        if graph.nodes[node]['station'][0]: # if station node
            graph.nodes[node]['priority'] = [0]
            graph.nodes[node]['demand'] = [0]
            graph.nodes[node]['occupied'] = [True]
        else:
            rand_dem = random.randint(0, max_rand_dem)
            graph.nodes[node]['demand'] = [rand_dem]

            if ref_graph is None:
                rand_priority = random.randint(1, max_rand_priority)
                graph.nodes[node]['priority'] = [rand_priority]
            else:
                graph.nodes[node]['priority'] = [
                                    ref_graph.nodes[node]['priority'][0]
                                    ]

            graph.nodes[node]['occupied'] = [False]
        graph.nodes[node]['time_to_go'] = [time_to_go]

    return graph

def reset_graph_non_uniform(graph, time_to_go, stn_dem, occ_node):
    data_dict = {10:[2,5,3], 15:[5,7,3], 20:[7,10,3], 25:[10,10,5]}
    list_of_nodes = list(graph.nodes)
    random.shuffle(list_of_nodes)
    num_nodes = graph.number_of_nodes()
    zero_priority = list_of_nodes[0:data_dict[num_nodes][0]]
    low_priority = list_of_nodes[data_dict[num_nodes][0]:\
                                sum(data_dict[num_nodes][0:2])]
    high_priority = list_of_nodes[sum(data_dict[num_nodes][0:2]):]

    for node in zero_priority:
        station_node = graph.nodes[node]['station'][0]
        if station_node:
            graph.nodes[node]['station'] = [True]
            graph.nodes[node]['occupied'] = [False]
        else:
            graph.nodes[node]['station'] = [False]
            graph.nodes[node]['occupied'] = [False]
        if node == occ_node:
            graph.nodes[node]['occupied'] = [True]
            graph.nodes[node]['demand'] = [0]
        graph.nodes[node]['priority'] = [0]
        graph.nodes[node]['demand'] = [0]
        graph.nodes[node]['time_to_go'] = [time_to_go]
        graph.nodes[node]['stn_dem'] = [stn_dem]

    for node in low_priority:
        station_node = graph.nodes[node]['station'][0]
        if station_node:
            graph.nodes[node]['station'] = [True]
            graph.nodes[node]['occupied'] = [False]
            graph.nodes[node]['priority'] = [0]
            graph.nodes[node]['demand'] = [0]
        else:
            graph.nodes[node]['station'] = [False]
            graph.nodes[node]['occupied'] = [False]
            graph.nodes[node]['priority'] = [random.randint(1,2)]
            graph.nodes[node]['demand'] = [random.randint(1,4)]
        if node == occ_node:
            graph.nodes[node]['occupied'] = [True]
            graph.nodes[node]['demand'] = [0]
        graph.nodes[node]['time_to_go'] = [time_to_go]
        graph.nodes[node]['stn_dem'] = [stn_dem]

    for node in high_priority:
        station_node = graph.nodes[node]['station'][0]
        if station_node:
            graph.nodes[node]['station'] = [True]
            graph.nodes[node]['occupied'] = [False]
            graph.nodes[node]['priority'] = [0]
            graph.nodes[node]['demand'] = [0]
        else:
            graph.nodes[node]['station'] = [False]
            graph.nodes[node]['occupied'] = [False]
            graph.nodes[node]['priority'] = [random.randint(5,7)]
            graph.nodes[node]['demand'] = [random.randint(10,19)]
        if node == occ_node:
            graph.nodes[node]['occupied'] = [True]
            graph.nodes[node]['demand'] = [0]
        graph.nodes[node]['time_to_go'] = [time_to_go]
        graph.nodes[node]['stn_dem'] = [stn_dem]

    return graph


def reset_graph_non_uniform_no_stn(graph, time_to_go, occ_node,
                        random_demands=True, random_priorities=True):
    if random_demands and not random_priorities: #for cbls
        for node in graph.nodes:
            graph.nodes[node]['demand'] = [random.randint(1, 100)]
            graph.nodes[node]['priority'] = [1]
            graph.nodes[node]['occupied'] = [False]
            if node == occ_node:
                graph.nodes[node]['demand'] = [0]
                graph.nodes[node]['occupied'] = [True]
            graph.nodes[node]['time_to_go'] = [time_to_go]
        return graph

    data_dict = {10:[0,7,3], 15:[0,12,3], 20:[0,17,3], 25:[0,20,5]}
    list_of_nodes = list(graph.nodes)
    random.shuffle(list_of_nodes)
    num_nodes = graph.number_of_nodes()
    zero_priority = list_of_nodes[0:data_dict[num_nodes][0]]
    low_priority = list_of_nodes[data_dict[num_nodes][0]:\
                                sum(data_dict[num_nodes][0:2])]
    high_priority = list_of_nodes[sum(data_dict[num_nodes][0:2]):]

    for node in zero_priority:
        graph.nodes[node]['occupied'] = [False]
        if node == occ_node:
            graph.nodes[node]['occupied'] = [True]
            graph.nodes[node]['demand'] = [0]
        graph.nodes[node]['priority'] = [0]
        graph.nodes[node]['demand'] = [0]
        graph.nodes[node]['time_to_go'] = [time_to_go]

    for node in low_priority:
        graph.nodes[node]['occupied'] = [False]
        graph.nodes[node]['priority'] = [random.randint(1,2)]
        graph.nodes[node]['demand'] = [random.randint(1,4)]
        if node == occ_node:
            graph.nodes[node]['occupied'] = [True]
            graph.nodes[node]['demand'] = [0]
        graph.nodes[node]['time_to_go'] = [time_to_go]

    for node in high_priority:
        graph.nodes[node]['occupied'] = [False]
        graph.nodes[node]['priority'] = [random.randint(5,7)]
        graph.nodes[node]['demand'] = [random.randint(10,19)]
        if node == occ_node:
            graph.nodes[node]['occupied'] = [True]
            graph.nodes[node]['demand'] = [0]
        graph.nodes[node]['time_to_go'] = [time_to_go]

    return graph


def save_graph(nx_graph, graph_path):
    edge_labels = {
            (i,j): f"{nx_graph[i][j]['travel_time']}"
                    for i, j in nx_graph.edges
            }
    pos = nx.spring_layout(nx_graph)
    nx.draw(nx_graph, pos=pos, with_labels=True)
    nx.draw_networkx_edge_labels(
            nx_graph, pos, rotate=False, font_size=8, edge_labels=edge_labels
            )
    # plt.axis('off')
    plt.savefig(graph_path, dpi=300)

