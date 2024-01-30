'''Module to run RL evaluations again'''
import os
import gymnasium as gym
from env.single_agent_env import SingleAgentEnv #pylint: disable=unused-import
from itertools import count
import pickle
import pandas as pd
import glob
from ppo_agent import PPOAgent
import torch
import numpy as np
import time
#pylint:disable=[redefined-outer-name]

def load_configs(eval_graphs_path):
    with open(eval_graphs_path, 'rb') as f:
        soll_array = pickle.load(f)
    nx_graph = soll_array[0][0]
    #nx_graph_opt = soll_array[0][1]
    graph_adjacency = soll_array[0][2]
    station_flag = 'station' in nx_graph.nodes[0]

    env_config = {
            'nx_graph': nx_graph.copy(),
            'graph_adjacency': graph_adjacency,
            'steps_per_episode': soll_array[0][4],
            'station_flag': station_flag,
            }
    model_config = {
            'graph_adjacency': graph_adjacency,
            'nx_graph': nx_graph.copy(),
            }
    return env_config, model_config, soll_array

def load_models_and_env(results_path, env_config, model_config,
                        graph_info):
    graph_num = graph_info['graph_num']
    learner_num = graph_info['learner_num']
    ep_num = graph_info['ep_no']
    env = gym.make('SingleAgentEnv-v0', config=env_config)

    eval_models_path = os.path.join(results_path, f'graph_{graph_num}',
                            f'learner_{learner_num}')
    eval_models_path = glob.glob(f'{eval_models_path}/results_*')[0]
    eval_models_path = os.path.join(eval_models_path, 'eval_models')
    actor_path = os.path.join(eval_models_path, 'actor', f'actor_ep_{ep_num}')
    critic_path = os.path.join(eval_models_path, 'critic',
                                f'critic_ep_{ep_num}')

    agent = PPOAgent(model_config, {})
    dev = torch.device('cpu')

    agent.actor.load_state_dict(torch.load(actor_path, map_location=dev))
    agent.critic.load_state_dict(torch.load(critic_path, map_location=dev))

    return agent, env

def run_simulation(env, agent, sol_array):
    opt_gap = []
    paths = [[]]
    sol_times = []
    for i in range(len(sol_array)):
        sum_of_demands = 0.0
        state, _ = env.reset(nx_graph=sol_array[i][0])
        agent.reset_rollout()
        sum_of_times = 0.0
        for _ in count():
            start = time.time()
            action, _ = agent.get_action(state, deterministic=True)
            sum_of_times += time.time() - start
            action = int(action)
            paths[-1].append(action)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            sum_of_demands += abs(reward)
            state = next_state

            if done:
                gap = sum_of_demands - sol_array[i][3]
                opt_gap.append((gap/sol_array[i][3])*100)
                sol_times.append(sum_of_times)
                paths.append([])
                break

    return opt_gap, paths[:-1], sol_times

def save_results(eval_path, eval_df, opt_gaps, paths, ep_num):
    os.makedirs(store_path, exist_ok=True)
    os.makedirs(os.path.join(store_path, 'paths'), exist_ok=True)
    eval_df.loc[len(eval_df.index)] = [ep_num, np.average(opt_gaps),
                                        np.std(opt_gaps), opt_gaps]
    eval_df.to_csv(os.path.join(eval_path, 'opt_data.csv'))
    paths_df = pd.DataFrame(columns=['instance_num', 'path'])
    for i in range(len(paths)):
        paths_df.loc[len(paths_df)] = [i, paths[i]]
    paths_df.to_csv(os.path.join(eval_path, 'paths', f'paths_{ep_num}.csv'))

if __name__ == '__main__':
    res_path = '/home/aravind/Project/cluster_results/ICAPS_results'
    num_nodes = [10, 15, 20, 25]
    num_graphs = 3
    num_learners = 2
    dict_of_sol_times = {}
    for n in num_nodes:
        dict_of_sol_times[n] = []
        for g in range(1, num_graphs+1):
            graph_path = os.path.join(res_path, f'{n}_nodes_eval_graphs',
                                    f'graph_{g}')
            env_cfg, model_cfg, sol_array = load_configs(graph_path)
            for learner in range(2, num_learners+1):
                print(f'\n\nRunning graph {g}, learner {learner}\n\n')
                eval_path = os.path.join(res_path, f'{n}_nodes_eval_graphs',
                                    'eval')
                eval_df = pd.DataFrame(columns=['ep_no', 'avg_opt', 'std_dev',
                                        'opt_gaps'])
                store_path = os.path.join(eval_path, f'graph_{g}',
                                f'learner_{learner}', 're_eval')
                for ep_no in range(100000, 100500, 500):
                    agent, env = load_models_and_env(eval_path, env_cfg,
                            model_cfg,{'graph_num':g, 'learner_num':learner,
                                        'ep_no':ep_no})
                    opt_gaps, paths, sol_times = run_simulation(env,
                                                agent, sol_array)
                    dict_of_sol_times[n] += sol_times
                    #save_results(store_path, eval_df, opt_gaps, paths, ep_no)
    for n in num_nodes:
        print(len(dict_of_sol_times[n]))
        avg_sol_time = np.average(dict_of_sol_times[n])
        std_sol_time = np.std(dict_of_sol_times[n])

        print(f'For {n} nodes graph:')
        print(f'Avg = {np.round(avg_sol_time,4)}')
        print(f'Std = {np.round(std_sol_time,4)}')

    with open(os.path.join(res_path, 'sol_times'), 'wb') as f:
        pickle.dump(dict_of_sol_times, f)





