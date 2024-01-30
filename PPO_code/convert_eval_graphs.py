'''Module to convert the eval graphs to Nishchal to mine'''
import os
import pickle
import networkx as nx
import copy

def get_folder_name(n, g):
    return f'{n}N-1A-15T_G{g}'

def convert_graph(nx_graph, time_to_go, station_flag, dem_inc_flag=True,
                    n_nodes=None):
    conv_graph = nx.Graph()
    conv_graph_opt = nx.Graph()
    if n_nodes is None:
        n_nodes = nx_graph.number_of_nodes()
    if station_flag:
        stn_dem = nx_graph.nodes[list(nx_graph.nodes)[-1]]['stn_dem']
    else:
        stn_dem = 0
    for node in list(nx_graph.nodes)[0:n_nodes]:
        conv_graph.add_node(node)
        conv_graph_opt.add_node(node)

        if station_flag:
            conv_graph.nodes[node]['station'] = [
                                    nx_graph.nodes[node]['station']
                                    ]
            conv_graph_opt.nodes[node]['station'] = \
                                    nx_graph.nodes[node]['station']
        else:
            if dem_inc_flag:
                conv_graph.nodes[node]['demand'] = [
                                        nx_graph.nodes[node]['demand'] + 1
                                        ]
                conv_graph_opt.nodes[node]['demand'] = \
                                        nx_graph.nodes[node]['demand'] + 1
            else:
                conv_graph.nodes[node]['demand'] = [
                                        nx_graph.nodes[node]['demand']
                                        ]
                conv_graph_opt.nodes[node]['demand'] = \
                                        nx_graph.nodes[node]['demand']

        conv_graph.nodes[node]['occupied'] = [bool(
                                nx_graph.nodes[node]['num_occ_agents']
                                )]
        conv_graph_opt.nodes[node]['occupied'] = bool(
                                nx_graph.nodes[node]['num_occ_agents']
                                )
        if dem_inc_flag:
            conv_graph.nodes[node]['demand'] = [
                            nx_graph.nodes[node]['demand'] + 1
                            ]
            conv_graph_opt.nodes[node]['demand'] = \
                                nx_graph.nodes[node]['demand'] + 1
        else:
            conv_graph.nodes[node]['demand'] = [nx_graph.nodes[node]['demand']]
            conv_graph_opt.nodes[node]['demand'] = \
                                        nx_graph.nodes[node]['demand']

        if bool(nx_graph.nodes[node]['num_occ_agents']):
            conv_graph.nodes[node]['demand'] = [0]
            conv_graph_opt.nodes[node]['demand'] = 0

        conv_graph.nodes[node]['priority'] = [nx_graph.nodes[node]['priority']]
        conv_graph_opt.nodes[node]['priority']=nx_graph.nodes[node]['priority']

        if station_flag:
            conv_graph.nodes[node]['stn_dem'] = [stn_dem]
            conv_graph_opt.nodes[node]['stn_dem'] = stn_dem

        conv_graph.nodes[node]['time_to_go'] = [time_to_go]
        conv_graph_opt.nodes[node]['time_to_go'] = time_to_go

    for edge in nx_graph.edges:
        if edge[0] >= n_nodes or edge[1] >= n_nodes:
            continue
        conv_graph.add_edge(*edge)
        conv_graph_opt.add_edge(*edge)

        conv_graph.edges[edge]['travel_time'] = \
                                    [nx_graph.edges[edge]['travel_time']]
        conv_graph_opt.edges[edge]['travel_time'] = \
                                    nx_graph.edges[edge]['travel_time']

    conv_graph = conv_graph.to_directed()
    conv_graph_opt = conv_graph_opt.to_directed()

    return conv_graph, conv_graph_opt

if __name__ == '__main__':
    file_path = os.path.dirname(os.path.realpath(__file__))
    file_name = 'CBLS_compare'
    num_nodes_list = [25, 34]
    num_graphs = 1
    num_inst = 50
    steps_per_episode = 15
    stn_flag = False
    for num_nodes in num_nodes_list:
        path = os.path.join(file_path, 'CBLS_eval_graphs',
                            f'{num_nodes}_nodes_eval_graphs')
        os.makedirs(path, exist_ok=True)
        for graph_num in range(1, num_graphs+1):
            folder_name = get_folder_name(num_nodes, graph_num)
            folder_name = os.path.join(file_name, folder_name, 'test_instances')
            sol_array = []
            for i in range(1, num_inst+1):
                inst_path = os.path.join(folder_name,
                                        f'instance_{i}_multi.gpickle')
                csv_path = os.path.join(folder_name,
                                        f'opt_sol_instance_{i}.csv')
                with open(inst_path, 'rb') as f:
                    graph = pickle.load(f)
                with open(csv_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    opt_obj_val = float(lines[0].split(',')[1].split('\n')[0])

                #init_sum_dem = sum([ #pylint: disable=[consider-using-generator]
                #    graph.nodes[n]['demand']*graph.nodes[n]['priority']
                #            for n in graph.nodes
                #    ])
                #opt_obj_val -= init_sum_dem

                conv_nx_graph, conv_nx_graph_opt = convert_graph(
                                    graph, steps_per_episode, stn_flag)
                graph_adjacency = list(conv_nx_graph.edges)

                sol_array.append([copy.deepcopy(conv_nx_graph),
                        copy.deepcopy(conv_nx_graph_opt),
                        graph_adjacency,
                        opt_obj_val,
                        steps_per_episode])

            graph_path = os.path.join(path, f'graph_{graph_num}')
            to_dump = copy.deepcopy(sol_array)
            with open(graph_path, 'wb') as f:
                pickle.dump(to_dump, f)


