
BEAM_SEARCH = False
if BEAM_SEARCH:
  BEAM_WIDTH = 7

SEQ_ACT = True

NUM_NODES = 10

num_node_priority_dict = {10: [2, 3, 5], 15: [5, 3, 7], 20: [7, 3, 10], \
                          25: [10, 5, 10]}

num_station_dict = {10: 0, 15: 0, 20: 0, 25: 0}
# {10: 2, 15: 2, 20: 3, 25: 3} #{10: 1, 15: 1, 20: 2, 25: 3}

avg_connect_dict = {10: 5, 15: 7, 20: 5, 25: 7}                     

num_nodes_zero_priority = num_node_priority_dict[NUM_NODES][0]
num_nodes_high_priority = num_node_priority_dict[NUM_NODES][1]
num_nodes_low_priority = num_node_priority_dict[NUM_NODES][2]


NUM_AVG_CONN = avg_connect_dict[NUM_NODES] #2 #

NUM_AGENTS = 1

NUM_STATIONS = num_station_dict[NUM_NODES]

MAX_SIM_TIME = 15

HORIZON = MAX_SIM_TIME

REPLAN_INTERVAL = MAX_SIM_TIME

NUM_TEST_INSTANCES = 50

EVAL_INTERVAL = 500

EVAL_FOLDER = f"./{NUM_NODES}N-{NUM_AGENTS}A-{MAX_SIM_TIME}T_G"

CHARGE_PENALTY_PER_TIMESTEP = 0

IS_BATTERY_DYNAMICS = False

NUM_TRAIN_EPISODES = 5100

NUM_TRAIN_POL = 5 #

NUM_GRAPHS = 3 #1 #
