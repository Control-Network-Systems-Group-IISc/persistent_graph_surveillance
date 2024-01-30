'''Module to run the MPC evaluations'''
from convert_eval_graphs import convert_graph
import os
import pickle
import glob
from ppo_agent import PPOAgent
import torch
import numpy as np
#pylint:disable=[redefined-outer-name]

def load_graph(graph_path, time_to_go, num_nodes):
    with open(graph_path, 'rb') as f:
        nx_graph = pickle.load(f)
    nx_graph, _ = convert_graph(nx_graph, time_to_go, False, n_nodes=num_nodes)
    graph_adjacency = list(nx_graph.edges())

    model_config = {
            'graph_adjacency': graph_adjacency,
            'nx_graph': nx_graph.copy(),
            }
    return  nx_graph, model_config

def load_models(results_path, model_config,
                        graph_info):
    graph_num = graph_info['graph_num']
    learner_num = graph_info['learner_num']
    ep_num = graph_info['ep_no']

    eval_models_path = os.path.join(results_path, f'graph_{graph_num}',
                            f'learner_{learner_num}')
    #print(eval_models_path)
    eval_models_path = glob.glob(f'{eval_models_path}/results_*')[0]
    if ep_num != 'best':
        eval_models_path = os.path.join(eval_models_path, 'eval_models')
        actor_path = os.path.join(eval_models_path, 'actor',
                                    f'actor_ep_{ep_num}')
        critic_path = os.path.join(eval_models_path, 'critic',
                                    f'critic_ep_{ep_num}')
    else:
        actor_path = os.path.join(eval_models_path, 'best_model_actor')
        critic_path = os.path.join(eval_models_path, 'best_model_critic')

    agent = PPOAgent(model_config, {})
    dev = torch.device('cpu')

    agent.actor.load_state_dict(torch.load(actor_path, map_location=dev))
    agent.critic.load_state_dict(torch.load(critic_path, map_location=dev))

    return agent


def _valid_action(_, nx_graph):
    # edge_index = np.ascontiguousarray(self.graph_adjacency.T)
    #print(edge_index)
    for node in nx_graph.nodes.data():
        node_idx = node[0]
        node_features = node[1]
        #neighbor_idx = edge_index[1][edge_index[0]==node_idx]
        # allow self loop
        #neighbor_idx = np.append(neighbor_idx,node_idx)
        if node_features['occupied'] == [True]:# and action in neighbor_idx):
            return True, node_idx
    #print('Action is invalid', action)
    #raise Exception('Invalid action')
    return False, -1


def step(nx_graph, action, time_to_go, agent_to_reach):
    #obs = self.observation_space.sample()
    for node in nx_graph.nodes.data():
        node_idx = node[0]
        node_features = node[1]
        # if action is valid update agent location
        # else remain in same place
        if node_idx == action and agent_to_reach:
            node_features['occupied'] = [True]
            node_features['demand'] = [0] # zero demand
        else:
            node_features['occupied'] = [False]
            node_features['demand'][0] += 1
        # zero priority nodes have zero demand
        if node_features['priority'][0] == 0:
            node_features['demand'] = [0.]
        node_features['time_to_go'] = [time_to_go]
        nx_graph.add_node(node_idx, **node_features)

    obs = convert_nx_to_graph(nx_graph, time_to_go)
    reward = calc_reward(nx_graph)

    return obs, reward, nx_graph.copy()

def calc_reward(nx_graph):
    node_features = nx_graph.nodes.data()
    sum_of_demands = 0.0
    for node in node_features:
        sum_of_demands += node[1]['demand'][0] * node[1]['priority'][0]
    return -sum_of_demands

def convert_nx_to_graph(nx_graph, time_to_go):
    list_of_node_features = []
    for node in nx_graph.nodes.data():
        temp_list = np.array([], dtype=np.float32)
        node_features = node[1]
        for _, feature_vector in node_features.items():
            temp_list = np.append(temp_list,
                    np.array(feature_vector, dtype=np.float32))
        list_of_node_features.append(np.array(temp_list,
                    dtype=np.float32).reshape(1,-1))

    list_of_edge_features = []
    for edge in nx_graph.edges.data():
        temp_list = np.array([], dtype=np.float32)
        edge_features = edge[2]
        for _, feature_vector in edge_features.items():
            temp_list = np.append(temp_list,
                    np.array(feature_vector, dtype=np.float32))
        list_of_edge_features.append(np.array(temp_list,
                    dtype=np.float32).reshape(1,-1))

    return {'node_space': list_of_node_features,
            'edge_space': list_of_edge_features,
            'time_to_go': [time_to_go]}

def run_simulation(eval_graph_path, instance_num, mpc_horizon,
                agent, num_nodes):
    agent.actor.eval()
    agent.critic.eval()
    base_graph = agent.actor.nx_graph.copy()
    inst_path = os.path.join(eval_graph_path, f'instance_{instance_num}')
    sum_of_demands = 0.0
    ref_sum_of_demands = 0.0
    next_graph = None
    with open(os.path.join(inst_path, 'time_step_0.gpickle'), 'rb') as f:
        time_0_graph = pickle.load(f)
        time_0_graph, _ = convert_graph(time_0_graph, 15, False,
                        dem_inc_flag=False, n_nodes=num_nodes)
    for node in base_graph.nodes():
        base_graph.nodes[node]['demand'] = \
                time_0_graph.nodes[node]['demand']
        base_graph.nodes[node]['priority'] = \
                time_0_graph.nodes[node]['priority']
        base_graph.nodes[node]['time_to_go'] = \
                time_0_graph.nodes[node]['time_to_go']
        if time_0_graph.nodes[node]['occupied'][0]:
            base_graph.nodes[node]['occupied'] = [True]
        else:
            base_graph.nodes[node]['occupied'] = [False]
    current_graph = base_graph.copy()

    # get first action
    obs = convert_nx_to_graph(current_graph, 15)
    action, _ = agent.get_action(obs, deterministic=True)
    action = int(action)

    timer = 0
    while timer < mpc_horizon:
        if mpc_horizon - timer > 15:
            time_to_go = 15
        else:
            time_to_go = mpc_horizon - timer
        _, agent_loc = _valid_action(action, current_graph)
        travel_time = current_graph.edges[(agent_loc, action)]['travel_time'][0]
        for t in range(travel_time):
            timer += 1
            _, reward, next_graph = step(current_graph, action,
                                time_to_go, t==travel_time-1)
            sum_of_demands += abs(reward)
            with open(os.path.join(inst_path,
                        f'time_step_{timer}.gpickle'), 'rb') as f:
                ref_graph = pickle.load(f)
                ref_graph, _ = convert_graph(ref_graph, time_to_go, False,
                                dem_inc_flag=False, n_nodes=num_nodes)
                ref_sum_of_demands += abs(calc_reward(ref_graph))
            # load priorities of reference graph to next graph
            for node in next_graph.nodes():
                next_graph.nodes[node]['priority'] = \
                        ref_graph.nodes[node]['priority']
            # equate current graph to next_graph
            current_graph = next_graph.copy()
            if timer == mpc_horizon:
                break
        # Take action
        if timer != mpc_horizon:
            obs = convert_nx_to_graph(current_graph, time_to_go)
            action, _ = agent.get_action(obs, deterministic=True)
            action = int(action)

    return sum_of_demands, ref_sum_of_demands

if __name__ == '__main__':
    data = {}
    for g_num_ in [*list(range(1,4)), 'grid', 'irreg']:
    #for g_num_ in ['irreg']:
        list_of_opt_gaps = []
        for mpc_horizon in range(15, 151, 15):
            #g_num = 3
            g_num = -1
            if g_num_ != 'grid' or g_num_ != 'irreg':
                g_num = g_num_
                path = f'/home/aravind/Project/2301-surveillance/code/ICAPS_paper_code/PPO/mpc_files/25N-1A-150T_G{g_num}_MPC/test_instances' #pylint:disable=[line-too-long]
                res_path = '/home/aravind/Project/cluster_results/ICAPS_results/25_nodes_eval_graphs/eval'#pylint:disable=[line-too-long]
            if g_num_ == 'grid':
                path = '/home/aravind/Project/2301-surveillance/code/ICAPS_paper_code/PPO/CBLS_MPC/grid_test' #pylint:disable=[line-too-long]
                res_path = '/home/aravind/Project/cluster_results/ICAPS_results/CBLS_results/25_nodes_eval_graphs/eval/'#pylint:disable=[line-too-long]
                g_num = 1
            if g_num_ == 'irreg':
                path = '/home/aravind/Project/2301-surveillance/code/ICAPS_paper_code/PPO/CBLS_MPC/irreg_test' #pylint:disable=[line-too-long]
                res_path = '/home/aravind/Project/cluster_results/ICAPS_results/CBLS_results/34_nodes_eval_graphs/eval/'#pylint:disable=[line-too-long]
                g_num = 1

            # mpc_horizon = 150
            num_instance = 5
            num_learners = 5
            learners = [*list(range(1, num_learners+1))]
            ep_nos = [100000]#, 'best']
            n_nodes = 25 if g_num_ != 'irreg' else 34
            learner_data = []
            for learner_num in learners:
                for ep_no in ep_nos:
                    print(f'LEARNER {learner_num}\tep {ep_no}\t {g_num_} \t horz {mpc_horizon}')
                    #list_of_opt_gaps = []
                    for inst_num in range(1, num_instance+1):
                        #g_path = os.path.join(path,
                        #    f'instance_{inst_num}/time_step_0.gpickle')
                        g_path = os.path.join(path,
                                'base_instance_multi_step.gpickle')
                        nx_graph, model_config = load_graph(g_path, 15, n_nodes)
                        agent = load_models(res_path, model_config,
                                            {'graph_num':g_num,
                                            'learner_num':learner_num,
                                            'ep_no':ep_no})
                        sum_of_dem, ref_sum_of_dem = run_simulation(path,
                                                inst_num, mpc_horizon,
                                                agent, n_nodes)
                        #print(
                        #f'{sum_of_dem}\t{ref_sum_of_dem}
                        # for instance {inst_num}')
                        opt_gap = (sum_of_dem -\
                                ref_sum_of_dem)/ref_sum_of_dem * 100
                        #print(f'opt_gap for instance {inst_num} = {opt_gap}')
                        # list_of_opt_gaps.append(opt_gap)
                        learner_data.append(opt_gap)
                    #print('Avg:', np.average(list_of_opt_gaps))
                    #print('Std:', np.std(list_of_opt_gaps))
                    # learner_data[learner_num] = list_of_opt_gaps
                    #print()
                    #print()
            list_of_opt_gaps.append(np.average(learner_data))
        data[g_num_] = list_of_opt_gaps
    print(data)
    with open('mpc_files/mpc_multi_time_horizons', 'wb') as f:
        pickle.dump(data, f)



