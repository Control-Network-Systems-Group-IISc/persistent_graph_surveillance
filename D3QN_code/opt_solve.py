#!/usr/bin/env python3
# Copyright 2010-2022 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Integer programming examples that show how to use the APIs."""
# [START program]
# [START import]
import sys
from pyscipopt import Model
from pyscipopt import quicksum
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
# from pprint import pprint

import data_file

# [END import]

class solve_main():
  """
  Example doc string
  """
  def __init__(self, graph_obj, agents_obj, prev_plan, init_battery, max_bat,
               optim_time_interval, optim_time_horizon, curr_opt_interval_num,
               least_time_increment, init_G, opt_bound=None, par_s=None,
               is_battery_dyn=data_file.IS_BATTERY_DYNAMICS):
    # [START solver]
    # Create the mip solver with the SCIP backend.
    self.model = Model()
    if not self.model:
      return
    # [END solver]

    self.infinity = self.model.infinity()
    self.opt_bound = self.infinity if (opt_bound is None) else opt_bound
    self.is_battery_dyn = is_battery_dyn

    self.graph = graph_obj
    self.V = list(filter(lambda node: (\
        not (self.graph.nodes[node]['agent'])), self.graph.nodes()))

    # print(f"len of V: {len(self.V)}")

    self.V_not_C = list(filter(lambda node: (\
        not (self.graph.nodes[node]['station'])), self.V))
    self.C = list(filter(lambda node: (self.graph.nodes[node]['station']), 
                                       self.V))
    self.dt = least_time_increment
    self.Tc = optim_time_interval
    self.Th = optim_time_horizon
    self.l = curr_opt_interval_num
    self.A = agents_obj['list']
    self.x_hat = prev_plan
    self.init_bat = init_battery
    self.max_bat = max_bat
    self.par_sol = par_s
    self.m_k_0 = agents_obj['m_k_t']
    self.D_j_0 = [self.graph.nodes[i]['demand'] for i in self.V]
    self.G_0 = init_G
    self.time_inst_list = range(0, self.Th+1, self.dt) 
    # range(self.l*self.Tc, self.l*self.Tc + self.Th + self.dt, self.dt)
    # self.M = 2*self.l
    self.M_obj = 2*(max([self.graph.nodes[_]['priority'] for _ in self.V]) *  \
           (max([self.graph.nodes[_]['demand'] for _ in self.V]) + (self.Th+1)))
    self.M_bat = (2*max([self.max_bat[_] for _ in self.max_bat.keys()]))

    self.M_stn_dyn = 2*(max(self.G_0) + (self.Th))

    # print(type(self.M_obj))
    # exit()
    #np.exp(self.Th**2)

    ######### Optimization variable creation #########

    self.x = [[[self.model.addVar(name=f"x({i},{k},{t})", \
      vtype="B") for t in self.time_inst_list] for k in self.A] for i in self.V]

    self.D = [[self.model.addVar(lb=0.0, name=f"D({i},{t})", \
      vtype="I") for t in self.time_inst_list] for i in self.V]

    self.G = [self.model.addVar(lb=0.0, name=f"G({t})", \
      vtype="I") for t in self.time_inst_list]

    self.obj = self.model.addVar() #ub=self.opt_bound)

    self.obj_act = self.model.addVar()

    self.dummy_vars = {f"demand_update": None, f"battery_dyn": None, \
                       f"m_k_t_update": None, f"plan_update": None, \
                       f"node_trvl_time": None, f"obj": None, \
                       f"stn_dem_dyn": None}

    ######### Optimization variable creation #########

    self.model.addCons(self.obj_act == quicksum(\
        self.graph.nodes[j]['priority'] * self.D[j][t+1] for j in \
                        self.V_not_C for t in self.time_inst_list[:-1])) # + \
                    # quicksum(self.G[t_+1] for t_ in self.time_inst_list[:-1]))
    # self.model.addCons(self.obj <= self.opt_bound) 

    self.model.addCons(self.obj == quicksum(\
        self.graph.nodes[j]['priority'] * self.D[j][t+1] for j in \
                        self.V_not_C for t in self.time_inst_list[:-1])) # + \
                    # quicksum(self.G[t_+1] for t_ in self.time_inst_list[:-1]))

    ######### Optimization variable creation #########


    ######### Initial position and battery canstraints #########

    for k_ in self.A:
      if self.is_battery_dyn:
        self.model.addCons(self.battery[k_][0] == self.init_bat[k_])
      # for i_ in self.V:
        # self.model.addCons(self.x[i_][k_][0] == self.x_hat[i_][k_][0])

    ######### Initial position and battery canstraints #########


    ######### Initial station demand canstraints #########

    ##### NOT COMPATIBLE WITH MULTI-AGENT!!!!! #####

    # self.model.addCons(self.G[0] == self.G_0[0])

    ######### Initial station demand canstraints #########



    ######### station demand dynamics canstraints #########

    ##### NOT COMPATIBLE WITH MULTI-AGENT!!!!! #####

    # self.dummy_vars[f"stn_dem_dyn"] = []

    # for t_ in self.time_inst_list[:-1]:

    #   self.dummy_vars[f"stn_dem_dyn"].append(self.model.addVar(name=\
    #       f"dummy_var_stn_dem_dyn_{len(self.dummy_vars[f'stn_dem_dyn'])}",\
    #       vtype="B"))

    #   self.model.addCons(self.G[t_+1] <= self.G[t_] + 1)
    #   self.model.addCons(self.G[t_+1] <= self.M_stn_dyn*(1 - quicksum(\
    #                                       self.x[i][0][t_+1] for i in self.C)))
    #   self.model.addCons(self.G[t_+1] >= (self.G[t_] + 1) - (4*self.M_stn_dyn*\
    #                                 (1 - self.dummy_vars[f"stn_dem_dyn"][-1])))
    #   self.model.addCons(self.G[t_+1] >= self.M_stn_dyn*(1 - quicksum(\
    #                                     self.x[i][0][t_+1] for i in self.C)) \
    #                                                   - ((4*self.M_stn_dyn) * \
    #                                       self.dummy_vars[f"stn_dem_dyn"][-1]))

    ######### station demand dynamics canstraints #########


    
    ######### Node demand dynamics constraints #########
    # done min updating

    self.dummy_vars[f"demand_update"] = {}
    self.dummy_vars[f"demand_update"]["min_1_sum"] = []
    self.dummy_vars[f"demand_update"]["min_d(t)+inc_(1-min)"] = []
    for i_ in self.V_not_C:
      self.model.addCons(self.D[i_][0] == (self.D_j_0[i_]) * \
                       (1 - quicksum(self.x_hat[i_][k__][0] for k__ in self.A)))
      for t_ in self.time_inst_list[:-1]:
        # m = min(1, quicksum(...))

        self.dummy_vars[f"demand_update"]["min_1_sum"].append(\
        self.model.addVar(name=\
        f"dummy_var_dem_{len(self.dummy_vars[f'demand_update']['min_1_sum'])}",\
        vtype="B"))

        self.dummy_vars[f"demand_update"]["min_1_sum"].append(\
        self.model.addVar(name=\
        f"dummy_var_dem_{len(self.dummy_vars[f'demand_update']['min_1_sum'])}",\
        vtype="B"))

        # for k_ in self.A:
        self.model.addCons(self.dummy_vars[f"demand_update"]["min_1_sum"][-1]\
                                                                           <= 1)

        self.model.addCons(self.dummy_vars[f"demand_update"]["min_1_sum"][-1] <=
                           quicksum(self.x[i_][k__][t_+1] for k__ in self.A)) 

        self.model.addCons(self.dummy_vars[f"demand_update"]["min_1_sum"][-1] \
          >= (1 - (2*self.M_obj*(1 - \
            self.dummy_vars[f"demand_update"]["min_1_sum"][-2]))))

        self.model.addCons(self.dummy_vars[f"demand_update"]["min_1_sum"][-1] \
          >= (quicksum(self.x[i_][k__][t_+1] for k__ in self.A) - \
          (2*self.M_obj*(self.dummy_vars[f"demand_update"]["min_1_sum"][-2]))))

        # self.model.addCons(self.D[i_][t_] >= 0)

        # MILP and Big-M method to implement non-linear dynamics 
        # D(t+1) = (D(t) + inc(t)) * (1- m) as 
        # D(t+1) = min( (D(t) + inc(t)) , (1- m) )
        self.dummy_vars[f"demand_update"]["min_d(t)+inc_(1-min)"].append(\
          self.model.addVar(name=\
          f"dummy_var_dem_{len(self.dummy_vars[f'demand_update']['min_d(t)+inc_(1-min)'])}", vtype="B"))
        
        self.model.addCons(self.D[i_][t_+1] <= (self.D[i_][t_] + \
          self.f_i(i_, t_+1)))


        self.model.addCons(self.D[i_][t_+1] <= self.M_obj * (1 - \
          self.dummy_vars[f"demand_update"]["min_1_sum"][-1]))

        self.model.addCons(self.D[i_][t_+1] >= ((self.D[i_][t_] + \
        self.f_i(i_, t_+1))) - ((2*self.M_obj) * \
        (1 - self.dummy_vars[f"demand_update"]["min_d(t)+inc_(1-min)"][-1])))

        self.model.addCons(self.D[i_][t_+1] >= (self.M_obj * \
          (1 - self.dummy_vars[f"demand_update"]["min_1_sum"][-1])) - \
          ((2*self.M_obj) * \
          self.dummy_vars[f"demand_update"]["min_d(t)+inc_(1-min)"][-1]))

          # == (self.D[i_][t_-1] + \
        	# self.f_i(i_, t_)) * (1 - self.dummy_vars[f"demand_update"][-1]))

    for c_ in self.C:
      for t_ in self.time_inst_list:
        self.model.addCons(self.D[c_][t_] == 0)

    ######### Node demand dynamics constraints #########

    ######### Agent occupancy constraints #########

    for k in self.A:
      for t_ in self.time_inst_list[:-1]:
        self.model.addCons(quicksum(self.x[i_][k][t_] for i_ in self.V) == 1)

    for k in self.A:
      for i in self.V:
        self.model.addCons(self.x[i][k][0] <= quicksum(self.x_hat[j_][k][0] \
            for j_ in self.graph.neighbors(i)))
        for t_ in self.time_inst_list[:-1]:    
          # self.model.addCons(self.x[i][k][t_] <= quicksum((\
          #   nx.adjacency_matrix(self.graph).todense()[i, j_]) * \
          #   self.x[j_][k][t_-1] for j_ in self.V))
          self.model.addCons(self.x[i][k][t_+1] <= quicksum(self.x[j_][k][t_] \
            for j_ in self.graph.neighbors(i)))

    ######### Agent occupancy constraints #########

    
    
    # self.model.addCons(self.obj == quicksum(\
    # 	[(self.graph.nodes[j]['priority'] * self.D[j][t]) for j in self.V \
    #    for t in self.time_inst_list]))
    # self.model.addCons(self.obj <= self.opt_bound) 
    self.model.setObjective(self.obj_act, sense="minimize")

    _ = self.model.optimize()

  def f_i(self, node, time):
    return self.dt#*node

def gen_graph_and_agents(num_nodes, num_stations, num_connects_per_node, 
                         prob_of_rewiring, num_agents):
  G = nx.connected_watts_strogatz_graph(n=num_nodes, k=num_connects_per_node,
                                        p=prob_of_rewiring)
  A = range(num_agents)

  nx.set_node_attributes(G, False, "station")

  nx.set_node_attributes(G, 0, "demand")

  for edge in G.edges():
    G[edge[0]][edge[1]]['travel_time'] = 1 #np.random.choice(range(2, 4))

  G.add_edges_from([(node, node) for node in G.nodes()], travel_time=1)

  station_nodes_list = np.random.choice(G.nodes(), size=num_stations, \
  	                              replace=False)

  for station_node in station_nodes_list:
    G.nodes[station_node]['station'] = True

  return G, A

if __name__ == '__main__':

  Tc = 5
  Th = 10
  l = 1
  dt = 1

  G, A = gen_graph_and_agents(4, 1, 3, 0.5, 1)
  A_obj = {'list':A, 'm_k_t':[0 for a in A]}

  V = G.nodes()
  # print(f"**** len(V): {len(V)}****")
  C = list(filter(lambda node: (G.nodes[node]['station']), G.nodes()))
  # print(f"**** len(C): {len(C)}****")
  time_indices = range(0, Th+1, dt)

  x_hat = [[[0 for t in time_indices] for k in A] for i in V]
  # print(f"**** shape of x_hat: {np.shape(np.asarray(x_hat))}****")

  for c in [C[0]]:
    for k in A:
      for t in time_indices:
        x_hat[c][k][t] = 1


  solved_obj = solve_main(G, A_obj, x_hat, Tc, Th, l, dt)

  x_sol = np.asarray([[[\
      int(solved_obj.model.getVal(solved_obj.x[i][k][t]) > 0.5) \
      for t in solved_obj.time_inst_list] for k in solved_obj.A] \
      for i in solved_obj.V])

  print(f"np.shape(x_sol): {np.shape(x_sol)}, station nodes: {C}")

  print(f"==============\n")

  np.set_printoptions(threshold=sys.maxsize)

  for k in A:
    print(f"for all nodes")
    for t in time_indices:
      for i in V:
        if x_sol[i][k][t] > 0:
          print(f"At time {t}, agent {k} is in node {i}") 
          #, {sum(x_sol[j][k][t + G[i][j]['travel_time']] \
          # for j in G[i] if (t+G[i][j]['travel_time'] <= time_indices[-1]))} \
          # >= {x_sol[i][k][t]}") #,, {sum(x_sol[j][k][t + t_hat] for j in G[i]\
          # for t_hat in range(1, time_indices[-1]+1-t) if (t_hat < G[i][j][\
          # 'travel_time']) ) + x_sol[i][k][t]}")

  print(f"\n==============\n")

  print(f"*****num_vars: {int(solved_obj.model.getNVars())}*****")
  print(f"*****num_cons: {int(solved_obj.model.getNConss())}*****")

  for i in V:
    for j in G[i]:
      print(f"{j} is a neighbour of \
      	{i} and travel time is {G[i][j]['travel_time']}")

  label_dict_edges = {}
  label_dict_nodes = {}

  for edge in G.edges():
    label_dict_edges[edge] = f"e:{G.edges[edge]['travel_time']}"

  for node in G.nodes():
    label_dict_nodes[node] = f"{node}:{int(G.nodes[node]['station'])}"

  print(f"*************{solved_obj.model.getVal(solved_obj.obj)}")
  pos = nx.kamada_kawai_layout(G)   
  nx.draw(G, pos, labels=label_dict_nodes, with_labels=True)
  # plt.savefig(f"{save_path}/plt.png", dpi=600)
  # plt.tight_layout()
  plt.show()

