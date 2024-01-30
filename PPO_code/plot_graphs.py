'''Module to plot graphs from collected data'''
import os
import pandas as pd
from matplotlib import pyplot as plt
import argparse
import numpy as np
import pickle

def convert_str2list(string):
    l = string.split('[')[1].split(']')[0].split(',')
    l = [float(x.strip()) for x in l]
    return l

parser = argparse.ArgumentParser()
parser.add_argument('eval_graphs_pth', nargs='?', default=None)
parser.add_argument('num_learners', nargs='?', default=5)
parser.add_argument('num_graphs', nargs='?', default=1)
parser.add_argument('num_nodes', nargs='?', default=25)

args = parser.parse_args()

path = args.eval_graphs_pth
num_learners = int(args.num_learners)
num_graphs = int(args.num_graphs)
num_nodes = int(args.num_nodes)
file_path = os.path.dirname(os.path.realpath(__file__))
#path = os.path.join(file_path, path, 'eval')

dict_of_df = {}
for graph_num in range(1, num_graphs + 1):
    graph_eval_path = os.path.join(path, f'graph_{graph_num}')
    dict_of_df[graph_num] = {}
    for learner_num in range(1, num_learners+1):
        learner_path = os.path.join(graph_eval_path, f'learner_{learner_num}')
        results_path = os.path.join(learner_path, os.listdir(learner_path)[0])
        #results_path = os.path.join(learner_path, 'eval')
        csv_file_path = os.path.join(results_path, 'opt_data.csv')

        dict_of_df[graph_num][learner_num] = pd.read_csv(csv_file_path)


plot_data_y = []
plot_data_x = []
num_rows = dict_of_df[1][1].shape[0]
max_opt_gap = 0
for i in range(-1,num_rows+3):
    temp = []
    for graph_num in range(1, num_graphs + 1):
        for learner_num in range(1, num_learners+1):
            try:
                opt_gaps_list = convert_str2list(
                        dict_of_df[graph_num][learner_num]['opt_gaps'][i]
                        )
                temp += opt_gaps_list
                #temp.append(dict_of_df[graph_num][learner_num]['avg_opt'][i])
            except Exception as _: #pylint: disable=[broad-exception-caught]
                pass
    if len(temp) > 0:
        max_opt_gap = max(max(temp), max_opt_gap) #pylint:disable=[nested-min-max]
    plot_data_y.append(temp)
    plot_data_x.append(i+1)

plt.figure()
medianprops = {'linestyle':'-', 'linewidth':0, 'color':'k'}
meanprops = {'linestyle':'-', 'linewidth':5, 'color':'k'}
c = 'red'
plt.boxplot(plot_data_y,
            whis=[10, 90],
            positions=plot_data_x,
            meanline=True, showmeans=True,
            medianprops=medianprops, meanprops=meanprops,
            notch=False, patch_artist=False,
            capprops={'color':c}, whiskerprops={'color':c},
            flierprops={'color':c, 'markeredgecolor':c})
            #, boxprops=dict(facecolor=c, color=c)
plt.title(f'Box plot for graph of {num_nodes} nodes')
plt.xlabel('Num episodes/500')
plt.ylabel('% opt gap')
plt.yticks(list(range(0, int(max_opt_gap), 5)), size=6)
plt.grid()
plt.tight_layout()
save_path = os.path.join(path, f'box_plot_{num_nodes}_nodes.png')
print(f'Saving to {save_path}')
plt.savefig(save_path, dpi=800)
#plt.show()
print(f'Avg opt_gap = {np.round(np.average(plot_data_y[-4]),2)}')
print(f'Std opt_gap = {np.round(np.std(plot_data_y[-4]),2)}')

dict_of_data = {'avg':[], 'std':[]}
for ep_data in plot_data_y:
    if len(ep_data) == 0:
        continue
    dict_of_data['avg'].append(np.average(ep_data))
    dict_of_data['std'].append(np.std(ep_data))

with open(os.path.join(path, f'{num_nodes}_nodes_opt_gaps'), 'wb') as f:
    pickle.dump(dict_of_data, f)
