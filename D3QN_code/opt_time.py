
import csv
import numpy as np
from matplotlib import pyplot as plt


import data_file


BEAM_SEARCH = data_file.BEAM_SEARCH
NUM_NODES = data_file.NUM_NODES
NUM_AGENTS = data_file.NUM_AGENTS
SEQ_ACT = data_file.SEQ_ACT
if BEAM_SEARCH:
  BEAM_WIDTH = data_file.BEAM_WIDTH
MAX_SIM_TIME = data_file.MAX_SIM_TIME
HORIZON = data_file.HORIZON
REPLAN_INTERVAL = data_file.REPLAN_INTERVAL 
NUM_TEST_INSTANCES = data_file.NUM_TEST_INSTANCES
EVAL_INTERVAL = data_file.EVAL_INTERVAL
EVAL_FOLDER = data_file.EVAL_FOLDER



opt_path = f"./{NUM_NODES}N-{NUM_AGENTS}A-{MAX_SIM_TIME}T_G4/test_instances/"
opt_sol_time = {}

eval_path = f"./{NUM_NODES}N-{NUM_AGENTS}A-{MAX_SIM_TIME}T_G4/eval_data/"
eval_opt_sol_time = {}

for i_ in range(1, NUM_TEST_INSTANCES+1):
  with open(f"{opt_path}opt_sol_instance_{i_}.csv") as f:
    reader = csv.reader(f)
    for row_ind, row in enumerate(reader):
      if row[0] == "sol_time":
        opt_sol_time[i_] = float(row[1])
        break

  with open(f"{eval_path}opt_sol_instance_pol_3_inst_{i_}.csv") as f:
    reader = csv.reader(f)
    for row_ind, row in enumerate(reader):
      if row[0] == "sol_time":
        eval_opt_sol_time[i_] = float(row[1])
        break


temp_list = []
for i_ in range(1, NUM_TEST_INSTANCES+1):
  temp_list.append(100*(opt_sol_time[i_] - eval_opt_sol_time[i_])/eval_opt_sol_time[i_])


temp_list = np.asarray(temp_list)

print(f"average: {np.average(temp_list)}, stddev: {np.std(temp_list)}")



plt.hist(temp_list)
plt.show()

