'''Module to generate evaluation graphs and optimal solutions'''
import copy
import os
import graph_utils

from opt_solve import Solve_Main
import pickle


def gen_and_solve_graph(num_nodes, steps_per_episode, num_connects_per_node,
        num_stations, prob_of_rewiring, num_init=15):
    # generate a random graph
    nx_graph, nx_graph_opt, graph_adjacency = graph_utils.gen_graph_and_agents(
            num_nodes=num_nodes, num_stations=num_stations,
            num_connects_per_node=num_connects_per_node,
            prob_of_rewiring=prob_of_rewiring)
    sol_array = []
    for _ in range(num_init):
        nx_graph, nx_graph_opt = graph_utils.initiate_node_demands(
                nx_graph, nx_graph_opt, int(steps_per_episode/2),
                random_demands=True)

        opt_sol = get_optimal_solution(
                graph_opt=nx_graph_opt,
                steps_per_episode=steps_per_episode,
                num_agents=1,
                )
        opt_obj_val = opt_sol.model.getVal(opt_sol.obj)
        sol_array.append([copy.deepcopy(nx_graph),
                        copy.deepcopy(nx_graph_opt),
                        graph_adjacency,
                        opt_obj_val,
                        steps_per_episode])

    return sol_array


def get_optimal_solution(graph_opt, steps_per_episode, num_agents):
    agent_set = list(range(num_agents))
    agent_obj = {'list':agent_set, 'm_k_t':[0 for _ in agent_set]}
    c_list = list(
            filter(
                lambda node: (graph_opt.nodes[node]['station']),
                    graph_opt.nodes()))
    x_hat = [[[0 for _ in range(steps_per_episode+1)] for _ in agent_set]
            for _ in graph_opt.nodes()]

    for c in [c_list[0]]:
        for k in agent_set:
            for t_ in range(steps_per_episode+1):
                x_hat[c][k][t_] = 1
    solved_obj = Solve_Main(
            graph_obj=graph_opt,
            agents_obj=agent_obj,
            prev_plan=x_hat,
            optim_time_interval=5,
            optim_time_horizon=steps_per_episode,
            curr_opt_interval_num=1,
            least_time_increment=1,
            init_battery={0:15},
            max_bat={0:15},
        )

    return solved_obj

def gen_random_graphs(path, num_graphs, num_nodes, num_init, steps_per_episode,
        num_connects_per_node, num_stations=1, prob_of_rewiring=0.5):
    for i in range(num_graphs):
        sol_array = gen_and_solve_graph(
                    num_nodes=num_nodes,
                    steps_per_episode=steps_per_episode,
                    num_connects_per_node=num_connects_per_node,
                    num_stations=num_stations,
                    prob_of_rewiring=prob_of_rewiring,
                    num_init=num_init)

        graph_path = os.path.join(path, f'graph_{i}')

        to_dump = copy.deepcopy(sol_array)

        with open(graph_path, 'wb') as f:
            pickle.dump(to_dump, f)

if __name__ == '__main__':
    file_path = os.path.dirname(os.path.realpath(__file__))
    pth = os.path.join(file_path, '50_nodes_eval_graphs')
    os.makedirs(pth, exist_ok=True)
    gen_random_graphs(pth, num_graphs=5, num_nodes=50, num_init=50,
            steps_per_episode=25, num_connects_per_node=5,
            num_stations=1, prob_of_rewiring=0.5)

