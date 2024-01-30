#!/usr/bin/env python3
# Copyright 2010-2022 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#pylint: disable=[unused-import][invalid-name][f-string-without-interpolation][line-too-long][unused-argument][bad-indentation][inconsistent-quotes][superfluous-parens][trailing-whitespace][consider-using-generator]
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

#import data_file

# [END import]

class Solve_Main():
  """
  Example doc string
  """
  def __init__(self, graph_obj, agents_obj, prev_plan, init_battery, max_bat,
               optim_time_interval, optim_time_horizon, curr_opt_interval_num,
               least_time_increment, opt_bound=None,
               is_battery_dyn=False):
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
    #self.V = list(filter(lambda node: (\
    #    not (self.graph.nodes[node]['agent'])), self.graph.nodes()))
    self.V = self.graph.nodes()
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
    self.m_k_0 = agents_obj['m_k_t']
    self.D_j_0 = [self.graph.nodes[i]['demand'] for i in self.V]
    self.time_inst_list = range(0, self.Th+1, self.dt) 
    # range(self.l*self.Tc, self.l*self.Tc + self.Th + self.dt, self.dt)
    # self.M = 2*self.l
    self.M_obj = (max([self.graph.nodes[_]['priority'] for _ in self.V]) *  \
            (max([self.graph.nodes[_]['demand'] for _ in self.V] + [100]) * (self.Th)))
    self.M_bat = (2*max([self.max_bat[_] for _ in self.max_bat.keys()]))

    # print(type(self.M_obj))
    # exit()
    #np.exp(self.Th**2)

    ######### Optimization variable creation #########

    self.x = [[[self.model.addVar(name=f"x({i},{k},{t})", \
      vtype="B") for t in self.time_inst_list] for k in self.A] for i in self.V]

    
    # self.m = [[self.model.addVar(lb=0.0, name=f"m({k},{t})", \
    #   vtype="I") for t in self.time_inst_list] for k in self.A]

    self.D = [[self.model.addVar(lb=0.0, name=f"D({i},{t})", \
      vtype="C") for t in self.time_inst_list] for i in self.V]
    
    self.battery = [[self.model.addVar(lb=0.0, name=f"bat({k},{t})", \
      vtype="C") for t in self.time_inst_list] for k in self.A]
    
    # self.b = [[self.model.addVar(name=f"b({k},{t})", \
    #  vtype="B") for t in self.time_inst_list] for k in self.A]
    
    self.obj = self.model.addVar()

    self.dummy_vars = {f"demand_update": None, f"battery_dyn": None, \
                       f"m_k_t_update": None, f"plan_update": None, \
                       f"node_trvl_time": None, f"obj": None}


    ######### Optimization variable creation #########

    self.model.addCons(self.obj == quicksum(\
            self.graph.nodes[j]['priority'] * self.D[j][t] for j in self.V for t in self.time_inst_list[1:]))
    # self.model.addCons(self.obj <= self.opt_bound) 


    ######### Initial position and battery canstraints #########

    for k_ in self.A:
      if self.is_battery_dyn:
        self.model.addCons(self.battery[k_][0] == self.init_bat[k_])
      for i_ in self.V:
        self.model.addCons(self.x[i_][k_][0] == self.x_hat[i_][k_][0])


    ######### Initial position and battery canstraints #########

    ######### Battery dynamics canstraints #########

    if self.is_battery_dyn:
    
      self.dummy_vars[f"battery_dyn"] = []
      for k_ in self.A:
        for t_ in self.time_inst_list[1:]:
          
          # bat(t) == min((bat(t-1)-1), M*(not_in_any_station)) + 
          #                                                    (maxC*in_a_station)
          
          
          self.dummy_vars[f"battery_dyn"].append(self.model.addVar(lb=0.0, name=\
          f"dummy_var_batdyn_{len(self.dummy_vars[f'battery_dyn'])}",\
          vtype="C")) # = min((bat(t-1)-1), M*(not_in_any_station))

          self.dummy_vars[f"battery_dyn"].append(self.model.addVar(name=\
          f"dummy_var_batdyn_{len(self.dummy_vars[f'battery_dyn'])}",\
          vtype="B")) # = dummy bool var for min linearization

          self.model.addCons(self.dummy_vars[f"battery_dyn"][-2] <= 
                             (self.battery[k_][t_-1] - 1))

          self.model.addCons(self.dummy_vars[f"battery_dyn"][-2] <= 
                             self.M_bat* (1 - quicksum(self.x[c_][k_][t_-1]
                                                       for c_ in self.C)))

          self.model.addCons(self.dummy_vars[f"battery_dyn"][-2] >= 
                             (self.battery[k_][t_-1] - 1) - 
                             (2 * self.M_bat * 
                              (1 - self.dummy_vars[f"battery_dyn"][-1])))

          self.model.addCons(self.dummy_vars[f"battery_dyn"][-2] >= 
                             self.M_bat * (1 - quicksum(self.x[c_][k_][t_-1] 
                                                        for c_ in self.C)) -
                             (2*self.M_bat * self.dummy_vars[f"battery_dyn"][-1]))

          self.model.addCons(self.battery[k_][t_] == 
                             self.dummy_vars[f"battery_dyn"][-2] + 
                             (self.max_bat[k_]*quicksum(self.x[c_][k_][t_-1]
                                                        for c_ in self.C)))
          
    ######### Battery dynamics canstraints #########


    ######### Node demand dynamics constraints #########
    # done min updating

    self.dummy_vars[f"demand_update"] = {}
    self.dummy_vars[f"demand_update"]["min_1_sum"] = []
    self.dummy_vars[f"demand_update"]["min_d(t-1)+inc_(1-min)"] = []
    for i_ in self.V_not_C:
      self.model.addCons(self.D[i_][0] == self.D_j_0[i_])
      for t_ in self.time_inst_list[1:]:
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
                           quicksum(self.x[i_][k__][t_] for k__ in self.A)) 

        self.model.addCons(self.dummy_vars[f"demand_update"]["min_1_sum"][-1] \
          >= (1 - (2*self.M_obj*(1 - \
            self.dummy_vars[f"demand_update"]["min_1_sum"][-2]))))

        self.model.addCons(self.dummy_vars[f"demand_update"]["min_1_sum"][-1] \
          >= (quicksum(self.x[i_][k__][t_] for k__ in self.A) - \
          (2*self.M_obj*(self.dummy_vars[f"demand_update"]["min_1_sum"][-2]))))

        # self.model.addCons(self.D[i_][t_] >= 0)

        # MILP and Big-M method to implement non-linear dynamics 
        # D(t+1) = (D(t) + inc(t)) * (1- m) as 
        # D(t+1) = min( (D(t) + inc(t)) , (1- m) )
        self.dummy_vars[f"demand_update"]["min_d(t-1)+inc_(1-min)"].append(\
          self.model.addVar(name=\
          f"dummy_var_dem_{len(self.dummy_vars[f'demand_update']['min_d(t-1)+inc_(1-min)'])}", vtype="B"))
        
        self.model.addCons(self.D[i_][t_] <= (self.D[i_][t_-1] + \
          self.f_i(i_, t_)))


        self.model.addCons(self.D[i_][t_] <= self.M_obj * (1 - \
          self.dummy_vars[f"demand_update"]["min_1_sum"][-1]))

        self.model.addCons(self.D[i_][t_] >= ((self.D[i_][t_-1] + \
        self.f_i(i_, t_))) - ((2*self.M_obj) * \
        (1 - self.dummy_vars[f"demand_update"]["min_d(t-1)+inc_(1-min)"][-1])))

        self.model.addCons(self.D[i_][t_] >= (self.M_obj * \
          (1 - self.dummy_vars[f"demand_update"]["min_1_sum"][-1])) - \
          ((2*self.M_obj) * \
          self.dummy_vars[f"demand_update"]["min_d(t-1)+inc_(1-min)"][-1]))

          # == (self.D[i_][t_-1] + \
        	# self.f_i(i_, t_)) * (1 - self.dummy_vars[f"demand_update"][-1]))

    for c_ in self.C:
      for t_ in self.time_inst_list:
        self.model.addCons(self.D[c_][t_] == 0)

    ######### Node demand dynamics constraints #########

    ######### Agent occupancy constraints #########

    for k in self.A:
      for t_ in self.time_inst_list[1:]:
        self.model.addCons(quicksum(self.x[i_][k][t_] for i_ in self.V) == 1)
        for i in self.V:
          # self.model.addCons(self.x[i][k][t_] <= quicksum((\
          #   nx.adjacency_matrix(self.graph).todense()[i, j_]) * \
          #   self.x[j_][k][t_-1] for j_ in self.V))
          self.model.addCons(self.x[i][k][t_] <= quicksum(self.x[j_][k][t_-1] \
            for j_ in self.graph.neighbors(i)))

          # self.model.addCons(self.battery[k][t_] >= self.x[i][k][t_]*\
          #   min([nx.shortest_path_length(self.graph, source=i, target=c_) 
          #        for c_ in self.C]))

    ######### Agent occupancy constraints #########

    ######### Node to node travel time constraints #########
    # for i in self.V:
    #   for k in self.A:
    #     for t in self.time_inst_list[:-1]:
    #       self.model.addCons(sum(self.x[j][k][t +\
    #        self.graph[i][j]['travel_time']] for j in self.graph[i] \
    #         if t+self.graph[i][j]['travel_time'] <= \
    #         self.time_inst_list[-1]) >= self.x[i][k][t])

    #       self.model.addCons(sum(self.x[j][k][t + t_hat] for j in self.graph[i]\
    #        for t_hat in range(1, self.time_inst_list[-1]+1-t) \
    #        if t_hat < self.graph[i][j]['travel_time']) + \
    #        self.x[i][k][t] <= 1)

    #       for j_ in self.graph[i]:
    #         self.model.addCons(sum(self.x[j_hat][k][t + \
    #         	self.graph[i][j_]['travel_time']] for j_hat in self.V \
    #         	if ((t+self.graph[i][j_]['travel_time'] <= \
    #         	 self.time_inst_list[-1]) and (j_hat not in self.graph[i]))) <= \
    #         	  1-self.x[i][k][t])
            
    #         self.model.addCons(sum(self.x[j_hat][k][t + t_hat] \
    #         	for j_hat in self.V if j_hat not in self.graph[i] \
    #         	for t_hat in range(1, self.time_inst_list[-1]+1-t) \
    #         	 if t_hat < self.graph[i][j_]['travel_time']) + \
    #         	 self.x[i][k][t] <= 1)

    ######### Node to node travel time constraints #########

    # ######## Agent m_k_t update constraints #########

    # self.dummy_vars[f"m_k_t_update"] = {}
    # self.dummy_vars[f"m_k_t_update"]["vars"] = []
    # self.dummy_vars[f"m_k_t_update"]["max"] = []

    # for k in self.A:
    #   self.model.addCons(self.m[k][0] == self.m_k_0[k])
    #   self.dummy_vars[f"m_k_t_update"][k] = {}
    #   for t in self.time_inst_list[1:]:
    #     self.dummy_vars[f"m_k_t_update"][k][t] = [self.m[k][t-1]] + [quicksum(\
    #     	[self.m[k_hat][t]*self.x[i][k_hat][t]*self.x[i][k][t] \
    #     	 for i in self.V_not_C]) for k_hat in self.A if k_hat != k] + \
    #       [self.l*self.x[c][k][t] for c in self.C]
    #     self.dummy_vars[f"m_k_t_update"]["vars"].append([self.model.addVar(\
    #     	name=\
    #     	f"dummy_var_mkt_var_{len(self.dummy_vars[f'm_k_t_update']['vars'])}",\
    #     	vtype="B") for dum_var in self.dummy_vars[f"m_k_t_update"][k][t]])
    #     self.dummy_vars[f"m_k_t_update"]["max"].append(self.model.addVar(name=\
    #     	f"dummy_var_mkt_max_{len(self.dummy_vars[f'm_k_t_update']['max'])}",\
    #     	vtype="C"))
    #     self.model.addCons(quicksum(\
    #     	self.dummy_vars[f"m_k_t_update"]["vars"][-1]) == 1)
    #     for dum_var_ind in range(len(self.dummy_vars[f"m_k_t_update"][k][t])):
    #       self.model.addCons(\
    #       	self.dummy_vars[f"m_k_t_update"][k][t][dum_var_ind] <= \
    #       	self.dummy_vars[f"m_k_t_update"]["max"][-1])
    #       self.model.addCons(self.dummy_vars[f"m_k_t_update"]["max"][-1] <= \
    #       	2*self.M*(1 - self.dummy_vars[f"m_k_t_update"][k][t][dum_var_ind]))

    #     self.model.addCons(self.m[k][t] == \
    #     	  self.dummy_vars[f"m_k_t_update"]["max"][-1])

    # ######### Agent m_k_t update constraints #########

    # ######### Agent plan update constraints #########
    # # done min and max updating

    # self.dummy_vars[f"plan_update"] = {}
    # self.dummy_vars[f"plan_update"]['max'] = []
    # self.dummy_vars[f"plan_update"]['min'] = []
    # for k in self.A:
    #   for t in self.time_inst_list:

    #     self.dummy_vars[f"plan_update"]["max"].append(self.model.addVar(\
    #     	name=f"dummy_var_plan_{len(self.dummy_vars[f'plan_update']['max'])}",\
    #     	vtype="B"))
    #     self.model.addCons(self.dummy_vars[f"plan_update"]['max'][-1] == \
    #     	((self.l - self.m[k][t])/2) + (abs(self.l - self.m[k][t])/2))
    #     self.dummy_vars[f"plan_update"]["min"].append(self.model.addVar(\
    #     	name=f"dummy_var_plan_{len(self.dummy_vars[f'plan_update']['min'])}", 
    #     	vtype="B"))
    #     self.model.addCons(self.dummy_vars[f"plan_update"]['min'][-1] == \
    #     	  ((1+self.dummy_vars[f"plan_update"]["min"][-1])/2) - \
    #     	  (abs(1-self.dummy_vars[f"plan_update"]["min"][-1])/2))
    #     self.model.addCons(self.b[k][t] == \
    #     	  self.dummy_vars[f"plan_update"]['min'][-1])

    # for i in self.V:
    #   for k in self.A:
    #     for t in self.time_inst_list:
    #       # print(f"node: {i}\tagent: {k}\ttime:{t}")
    #       # print(f"{self.x[i][k][t]}")
    #       # print(f"{self.x_hat[i][k][t]}")
    #       # print(f"{self.b[k][t]}")

    #       self.model.addCons(\
    #       	  self.x[i][k][t] <= self.b[k][t] * self.x_hat[i][k][t] +\
    #       	  (1 - self.b[k][t]))
    #       self.model.addCons(self.x[i][k][t] >= self.b[k][t] * \
    #       	  self.x_hat[i][k][t])

    ######### Agent plan update constraints #########
    
    # self.model.addCons(self.obj == quicksum(\
    # 	[(self.graph.nodes[j]['priority'] * self.D[j][t]) for j in self.V \
    #    for t in self.time_inst_list]))
    # self.model.addCons(self.obj <= self.opt_bound) 
    self.model.setObjective(self.obj, sense="minimize")

    _ = self.model.optimize()

  def f_i(self, node, time):
    return self.dt#*node

