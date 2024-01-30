'''Module for Single agent training'''
import os
import sys
import gymnasium as gym
import matplotlib.pyplot as plt
import pprint
import graph_utils
from itertools import count
import torch
from env.single_agent_env import SingleAgentEnv      #pylint: disable=unused-import
from prettytable import PrettyTable
import datetime as dt
import pytz
from ppo_agent import PPOAgent
import numpy as np
from opt_solve import Solve_Main
import pickle
import argparse

import pandas as pd

import heuristics

def create_configs(num_nodes, steps_per_episode,
        dict_of_hyper_parameters, config_path, graph_path,
        eval_graphs_path, station_flag):
    # create the graph adjacency
    #graph_adjacency = [(0,1), (1,2), (2,3), (3,4), (4,1), (0,2)]

    # make the nx graph
    #nx_graph = graph_utils.prepare_graph(graph_adjacency,num_nodes)

    # generate random graph
    nx_graph = nx_graph_opt = graph_adjacency = None
    sol_array = []
    if eval_graphs_path is None:
        (nx_graph, nx_graph_opt,
                graph_adjacency) = graph_utils.gen_graph_and_agents(
                num_nodes=num_nodes, num_stations=1,
                num_connects_per_node=5, prob_of_rewiring=0.5)
    else:
        with open(eval_graphs_path, 'rb') as f:
            sol_array = pickle.load(f)
        nx_graph = sol_array[0][0]
        nx_graph_opt = sol_array[0][1]
        graph_adjacency = sol_array[0][2]
        dict_of_hyper_parameters['num_nodes'] = nx_graph.number_of_nodes()
        dict_of_hyper_parameters['steps_per_episode'] = sol_array[0][4]

    graph_utils.save_graph(nx_graph, graph_path)

    # create the environment config
    env_config = {
            'nx_graph': nx_graph.copy(),
            'graph_adjacency': graph_adjacency,
            'steps_per_episode': dict_of_hyper_parameters['steps_per_episode'],
            'station_flag': station_flag,
            }

    # create the model config
    model_config = {
            'graph_adjacency': graph_adjacency,
            'nx_graph': nx_graph.copy(),
        }
    optimal_obj_val = -1
    if eval_graphs_path is None:
        opt_sol = get_optimal_solution(
                graph_opt=nx_graph_opt,
                steps_per_episode=steps_per_episode,
                num_agents=1,
            )
        optimal_obj_val = opt_sol.model.getVal(opt_sol.obj)
        print(f'Optimal Solution obtained. Obj val={optimal_obj_val}')

    with open(config_path, 'w', encoding='utf8') as f:
        to_write = pprint.pformat(nx_graph.nodes.data())
        f.write(to_write)
        f.write('\n\n')
        to_write = pprint.pformat(nx_graph.edges.data())
        f.write(to_write)
        f.write('\n\n')
        to_write = pprint.pformat(dict_of_hyper_parameters, indent=4)
        f.write(to_write)
        f.write('\n\n')
        to_write = pprint.pformat(env_config, indent=4)
        f.write(to_write)
        f.write('\n\n')
        to_write = pprint.pformat(model_config, indent=4)
        f.write(to_write)
        f.write('\n\n')
        if eval_graphs_path is None:
            f.write(f'Optimal solution = {optimal_obj_val}')
            f.write('\n\n')
        else:
            f.write(f'Using graph pickle from {eval_graphs_path}')
            f.write('\n\n')
        to_write = f'CUDA Available:{torch.cuda.is_available()}\n'
        f.write(to_write)
        f.write('\n\n')

    return model_config, env_config, optimal_obj_val, sol_array


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
            init_battery={0:1},
            max_bat={0:15},
        )

    return solved_obj

def write_model_summary(model, config_path):
    table = PrettyTable(['Modules', 'Parameters'])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    with open(config_path, 'a', encoding='utf-8') as f:
        f.write(str(model) + '\n\n')
        f.write(table.get_string() + '\n')
        f.write(f'Total Trainable Params: {total_params}\n\n')

    print(table)
    print(f'Total Trainable Params: {total_params}')


def main(config_path, graph_path, demand_graph_path, loss_graph_path,
             eval_path, best_model_path, opt_gap_path, eval_graphs_path,
             saved_model_path):
    rewards = []
    dict_of_loss = {'actor_loss': [], 'critic_loss': []}
    list_of_eval_rewads = []
    list_of_opt_gap = []

    def plot_sum_of_demands():
        plt.figure(2)
        plt.clf()
        demands_t = torch.FloatTensor(rewards)
        plt.title('Training(without attention)...')
        plt.xlabel('Episode')
        plt.ylabel('Demand')
        plt.plot(demands_t.numpy(), label='Demand')
        # Take 100 episode averages and plot them too
        if len(demands_t) >= 10:
            means = demands_t.unfold(0, 10, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(9), means))
            plt.plot(means.numpy(), label='10 episode avg of demand')
        plt.legend()
        plt.tight_layout()
        plt.savefig(demand_graph_path, dpi=300)
        #plt.pause(0.001)  # pause a bit so that plots are updated

    def plot_loss():
        plt.figure(3)
        plt.clf()

        plt.subplot(2,1,1)
        plt.title('Actor loss')
        plt.xlabel('')
        plt.ylabel('Loss')
        plt.plot(dict_of_loss['actor_loss'])

        plt.subplot(2,1,2)
        plt.title('Critic Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.plot(dict_of_loss['critic_loss'])
        plt.tight_layout()

        plt.savefig(loss_graph_path, dpi=300)
        #plt.pause(0.001)  # pause a bit so that plots are updated

    def plot_eval():
        plt.figure(4)
        plt.clf()
        plt.title('Evaluation demands')
        plt.xlabel(f'Episode/{eval_freq}')
        plt.ylabel('Demand')
        plt.plot(list_of_eval_rewads)
        plt.tight_layout()

        plt.savefig(eval_path, dpi=300)

    def plot_opt_gap():
        plt.figure(5)
        plt.clf()
        plt.title('Percentage Optimal Gap')
        plt.xlabel(f'Episode/{eval_freq}')
        plt.ylabel('% optimal gap')
        plt.plot(list_of_opt_gap)
        plt.tight_layout()

        plt.savefig(opt_gap_path, dpi=300)

        temp = os.path.dirname(opt_gap_path)
        temp_txt = os.path.join(temp, 'opt_gap.txt')
        with open(temp_txt, 'w', encoding='utf8') as f:
            to_write = pprint.pformat(list_of_opt_gap, indent=4)
            f.write(to_write)
        temp_bin = os.path.join(temp, 'opt_gap')
        with open(temp_bin, 'wb') as f:
            pickle.dump(list_of_opt_gap, f)



    # parameters
    num_nodes = 25
    steps_per_episode = 15
    num_episodes = 100000
    gamma = 1.0
    actor_lr = 1e-4
    critic_lr = 5e-4
    clip = 0.01
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 10
    eps_greedy = False
    grad_clip_value = 100
    eval_freq = 500 # evaluate every 100 episodes
    num_eval_traj = 1
    updates_per_iteration = 10
    update_every = 50
    heuristic = None
    station_flag = False
    i_ep_start = 80000 if saved_model_path is not None else -1
    dict_of_hyper_parameters = {
            'num_episodes': num_episodes,
            'actor_lr': actor_lr,
            'critic_lr': critic_lr,
            'gamma': gamma,
            'num_nodes': num_nodes,
            'steps_per_episode': steps_per_episode,
            'grad_clip_value': grad_clip_value,
            'eps_decay': eps_decay,
            'eps_end': eps_end,
            'eval_freq': eval_freq,
            'num_eval_traj': num_eval_traj,
            'updates_per_iteration': updates_per_iteration,
            'clip': clip,
            'eps_greedy': eps_greedy,
            'update_every': update_every,
            'heuristic': heuristic,
            'station_flag': station_flag,
            'saved_model_path': saved_model_path,
            'i_ep_start': i_ep_start,
        }
    (model_config,
            env_config,
            optimal_obj_val,
            sol_array) = create_configs(num_nodes, steps_per_episode,
                                dict_of_hyper_parameters,
                                config_path=config_path,
                                graph_path=graph_path,
                                eval_graphs_path=eval_graphs_path,
                                station_flag=station_flag,)
    if eval_graphs_path is not None:
        num_nodes = sol_array[0][0].number_of_nodes()
        steps_per_episode = sol_array[0][4]

    env = gym.make('SingleAgentEnv-v0', config=env_config)

    agent = PPOAgent(model_config, dict_of_hyper_parameters)
    # run a dummy batch to init model params
    #agent.actor([env.reset()[0]])
    #agent.actor([env.reset()[0]])
    #if eval_graphs_path is None:
    #    agent.critic([env.reset()[0]])
    #    agent.actor([env.reset()[0]])
    #else:
    #    agent.critic([env.reset(non_uniform=True)[0]])
    #    agent.actor([env.reset(non_uniform=True)[0]])
    if saved_model_path is not None:
        print(f'Resuming Training from {saved_model_path} episode {i_ep_start}')
        actor_path = os.path.join(saved_model_path, 'eval_models',
                        'actor', f'actor_ep_{i_ep_start}')
        critic_path = os.path.join(saved_model_path, 'eval_models',
                        'critic', f'critic_ep_{i_ep_start}')
        agent.load_saved_model(actor_path, critic_path)

    write_model_summary(agent.actor, config_path)
    write_model_summary(agent.critic, config_path)

    eval_df = pd.DataFrame(
            columns=['ep_no', 'avg_opt', 'std_dev', 'opt_gaps', 'sol_times'])

    eps = eps_start
    print(eps)
    best_demand = np.inf
    for i_ep in range(i_ep_start+1, num_episodes+1):
        sys.stdout.write('\033[K') # Clear to the end of line
        print(f'Episode {i_ep} of {num_episodes}\t\
                Last opt_gaps:{np.round(list_of_opt_gap[-3:],2)}', end='\r')

        ########################### Evaluation ################################
        with torch.no_grad():
            agent.actor.eval()
            agent.critic.eval()
            if i_ep % eval_freq == 0:# and i_ep > 0:
                if eval_graphs_path is None:
                    eval_rewards = []
                    opt_gap = []
                    for i in range(num_eval_traj):
                        sum_of_demands = 0.0
                        state, _ = env.reset(random_demands=False)
                        # agent.reset_rollout()
                        actions_list = []
                        for _ in count():
                            action, _ = agent.get_action(
                                        state, deterministic=True)
                            if heuristic is not None:
                                action = heuristics.greedy(
                                    obs=state,
                                    graph_adjacency=model_config[
                                                    'graph_adjacency'],
                                    num_nodes=num_nodes)
                            action = int(action)
                            actions_list.append(action)
                            (next_state, reward,
                                    terminated, truncated, _) = env.step(action)
                            done = terminated or truncated

                            sum_of_demands += abs(reward)
                            state = next_state

                            if done:
                                eval_rewards.append(sum_of_demands)
                                gap = sum_of_demands - optimal_obj_val
                                opt_gap.append((gap / optimal_obj_val) * 100)
                                print(f'\ni={i}\tactions={actions_list}')
                                break
                    list_of_eval_rewads.append(np.average(eval_rewards))
                    list_of_opt_gap.append(np.average(opt_gap))
                    if np.average(eval_rewards) < best_demand:
                        best_demand = np.average(eval_rewards)
                        torch.save(agent.actor.state_dict(),
                                f'{best_model_path}_actor')
                        torch.save(agent.critic.state_dict(),
                                f'{best_model_path}_critic')
                    plot_eval()
                    plot_opt_gap()
                else:
                    opt_gap = []
                    sol_times = []
                    for i in range(len(sol_array)):
                        #g = sol_array[i][0]
                        #init_sum_dem = sum( #pylint:disable=[consider-using-generator]
                        #    [
                        #    g.nodes[n]['demand'][0]*g.nodes[n]['priority'][0]
                        #                    for n in g.nodes
                        #    ]
                        #)
                        sum_of_demands = 0.0
                        state, _ = env.reset(nx_graph=sol_array[i][0])
                        sum_of_times = 0.0
                        # agent.reset_rollout()
                        for _ in count():
                            start = dt.datetime.now()
                            action, _ = agent.get_action(
                                    state, deterministic=True)
                            sum_of_times += (
                                    dt.datetime.now() - start).total_seconds()
                            if heuristic is not None:
                                action = heuristics.greedy(
                                    obs=state,
                                    graph_adjacency=model_config[
                                        'graph_adjacency'],
                                    num_nodes=num_nodes)
                            action = int(action)
                            (next_state, reward,
                                    terminated, truncated, _) = env.step(action)
                            done = terminated or truncated

                            sum_of_demands += abs(reward)
                            state = next_state

                            if done:
                                #sum_of_demands += abs(init_sum_dem)
                                gap = sum_of_demands - sol_array[i][3]
                                opt_gap.append((gap / sol_array[i][3]) * 100)
                                sol_times.append(sum_of_times)
                                break
                    list_of_opt_gap.append(np.average(opt_gap))
                    if np.average(opt_gap) < best_demand:
                        best_demand = np.average(opt_gap)
                        torch.save(agent.actor.state_dict(),
                                f'{best_model_path}_actor')
                        torch.save(agent.critic.state_dict(),
                                f'{best_model_path}_critic')

                    eval_df.loc[len(eval_df.index)] = [i_ep,
                                                np.average(opt_gap),
                                                np.std(opt_gap), opt_gap,
                                                sol_times]

                    plot_opt_gap()

                eval_path = os.path.join(
                            os.path.dirname(best_model_path), 'eval_models')
                torch.save(agent.actor.state_dict(),
                            os.path.join(
                                os.path.join(eval_path,'actor'),
                                f'actor_ep_{i_ep}'
                                )
                            )
                torch.save(agent.critic.state_dict(),
                            os.path.join(
                                os.path.join(eval_path,'critic'),
                                f'critic_ep_{i_ep}'
                                )
                            )
                eval_df.to_csv(
                        os.path.join(
                            os.path.dirname(best_model_path),
                            'opt_data.csv'
                            )
                        )

        ########################### Evaluation ################################

        if heuristic is not None:
            continue
        # Switch to training mode
        agent.actor.train()
        agent.critic.train()

        if i_ep % update_every == 0 and i_ep > 0:
            agent.learn()
            dict_of_loss['actor_loss'].append(
                    np.average(agent.logger['actor_loss'])
                    )
            dict_of_loss['critic_loss'].append(
                    np.average(agent.logger['critic_loss'])
                    )
            agent.reset_rollout()

        if eval_graphs_path is None:
            state, _ = env.reset()
        else:
            state, _ = env.reset(non_uniform=True)
        agent.new_ep()
        sum_of_demands = 0.0
        for step in count():
            action, log_prob = agent.get_action(state, eps=eps,
                                                epsilon_greedy=eps_greedy)
            action = int(action)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update_rollout(state=state, action=action, log_prob=log_prob,
                                reward=reward, next_state=next_state, done=done)

            sum_of_demands += abs(reward)
            t = steps_per_episode*i_ep + step
            eps = np.exp(-t/eps_decay)
            eps = max(eps, eps_end)
            state = next_state

            if done:
                rewards.append(sum_of_demands)
                if i_ep % 100 == 0:
                    plot_sum_of_demands()
                    plot_loss()
                break
    print(
    f'\nSaving best model to {best_model_path} with best\
             optimality gap {best_demand}'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('eval_graphs_pth', nargs='?', default=None)
    parser.add_argument('graph_num', nargs='?' , default=None)
    parser.add_argument('learner_num', nargs='?', default=None)
    parser.add_argument('saved_model_path', nargs='?', default=None)

    args = parser.parse_args()
    eval_graphs_pth = args.eval_graphs_pth
    graph_num = args.graph_num
    learner_num = args.learner_num
    saved_model_pth = args.saved_model_path

    file_path = os.path.dirname(os.path.realpath(__file__))
    time = dt.datetime.now(pytz.timezone('Asia/Kolkata'))
    current_time = f'{time.year}_{time.month}_{time.day}_{time.hour}' +\
                        f'_{time.minute}_{time.second}_{time.microsecond}'
    results_path = os.path.join(file_path, f'results_{current_time}')

    if eval_graphs_pth is not None:
        #eval_graphs_pth = os.path.join(file_path,
        #            f'{eval_graphs_pth}','graph_{graph_num}')
        eval_graphs_pth = os.path.join(f'{eval_graphs_pth}',
                f'graph_{graph_num}')
    #eval_graphs_pth = None

    if graph_num is not None and learner_num is not None\
                and eval_graphs_pth is not  None:
        results_path = os.path.join(os.path.dirname(eval_graphs_pth),
                        'eval', f'graph_{graph_num}',
                        f'learner_{learner_num}', f'results_{current_time}')

    os.makedirs(results_path, exist_ok=True)
    eval_models_path = os.path.join(results_path, 'eval_models')
    os.makedirs(eval_models_path, exist_ok=True)
    os.makedirs(os.path.join(eval_models_path, 'actor'), exist_ok=True)
    os.makedirs(os.path.join(eval_models_path, 'critic'), exist_ok=True)


    config_pth = os.path.join(results_path, 'config.txt')
    graph_pth = os.path.join(results_path, 'graph.png')
    demand_graph_pth = os.path.join(results_path, 'demand.png')
    loss_graph_pth = os.path.join(results_path, 'loss.png')
    eval_pth = os.path.join(results_path, 'eval.png')
    best_model_pth = os.path.join(results_path, 'best_model')
    opt_gap_pth = os.path.join(results_path, 'opt_gap.png')

    main(config_pth, graph_pth, demand_graph_pth,
            loss_graph_pth, eval_pth, best_model_pth, opt_gap_pth,
            eval_graphs_pth, saved_model_pth)
    #plt.show()
    time_now = dt.datetime.now(pytz.timezone('Asia/Kolkata'))
    diff = time_now - time
    with open(config_pth, 'a', encoding='utf8') as f_:
        f_.write('\n\n')
        f_.write(f'Started at :{time}\n')
        f_.write(f'Completed at : {time_now}\n')
        f_.write(f'Time taken: {diff.total_seconds()} seconds\n\n')


