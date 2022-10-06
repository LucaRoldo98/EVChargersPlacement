# -*- coding: utf-8 -*-
import os
import time
import logging
import json
from networkx.algorithms.centrality import group
import numpy as np
import gurobipy as gp
from gurobipy import GRB

class Tester():
    def __init__(self):
        pass

    def compare_sols_lst(
        self, inst, sampler, sols, labels, n_scenarios
    ):
        ans_dict = {}
        reward = sampler.sample_stoch(
            inst,
            n_scenarios=n_scenarios
        )
        for j in range(len(sols)):
            profit_raw_data = self.solve_second_stages(
                inst, sols[j],
                n_scenarios, reward
            )
            ans_dict[labels[j]] = profit_raw_data

        return ans_dict

    def solve_second_stages(
        self, inst, gamma_prime, gamma_2nd, delta_prime, delta_2nd, n_scenarios, demg
    ):
        ans = []
        
        obj_fs = 0
        for i in range(inst.n_parkingLots):
            obj_fs -= inst.F_prime * delta_prime[i] + inst.F_2nd * delta_2nd[i] + inst.C_prime * gamma_prime[i] + inst.C_2nd * gamma_2nd[i]
        
        parkingLots = range(inst.n_parkingLots)
        groups = range(inst.n_groups)
        for s in range(n_scenarios):
            problem_name = "SecondStagePrb"
            model = gp.Model(problem_name)
            x = model.addVars(
            inst.n_parkingLots, inst.n_groups,
            lb=0,
            ub=GRB.INFINITY,
            vtype=GRB.INTEGER,
            name='x'
            )
            y = model.addVars(
                inst.n_parkingLots, inst.n_groups,
                lb=0,
                ub=GRB.INFINITY,
                vtype=GRB.INTEGER,
                name='y'
            )
            w = model.addVars(
                inst.n_parkingLots, inst.n_groups,
                lb=0,
                ub=GRB.INFINITY,
                vtype=GRB.INTEGER,
                name='w'
            )
            x_prime = model.addVars(
                inst.n_parkingLots, inst.n_groups,
                lb=0,
                ub=GRB.INFINITY,
                vtype=GRB.INTEGER,
                name='x_prime'
            )
            obj_funct = 0
            for p in parkingLots:
                obj_funct = obj_funct + inst.Eprime * inst.theta1 * gp.quicksum((y[p, g]+x[p, g]) for g in inst.G_prime[0][0][p])
                obj_funct = obj_funct + inst.Eprime * inst.theta1 * gp.quicksum((w[p, g] + x_prime[p, g])  for g in inst.G_prime[1][0][p])
                obj_funct = obj_funct + inst.Eprime * inst.theta2 * gp.quicksum(y[p, g] for g in inst.G_prime[0][1][p])
                obj_funct = obj_funct + inst.Eprime * inst.theta2 * gp.quicksum(x_prime[p, g] for g in inst.G_prime[1][1][p])
                obj_funct = obj_funct + inst.E_2nd * inst.theta2 * gp.quicksum(x[p, g] for g in inst.G_prime[0][1][p])
                obj_funct = obj_funct + inst.E_2nd * inst.theta2 * gp.quicksum(w[p, g] for g in inst.G_prime[1][1][p])
                obj_funct = obj_funct + inst.E_2nd * inst.theta3 * gp.quicksum((x[p, g] + y[p, g]) for g in inst.G_prime[0][2][p])
                obj_funct = obj_funct + inst.E_2nd * inst.theta3 * gp.quicksum((w[p, g] + x_prime[p,g])  for g in inst.G_prime[1][2][p])
            model.setObjective(obj_funct, GRB.MAXIMIZE)
            
            # CONSTRAINT (9)
            for g in groups:
                model.addConstr(
                    gp.quicksum(y[p,g] for p in inst.preferiteParkingsPerGroup[0][g]) <= inst.beta1 * demg[s][g],
                    f"max_y"
                )


            # CONSTRAINT (10)

            for g in groups:
                s1= gp.quicksum(x[p,g] for p in inst.preferiteParkingsPerGroup[0][g])
                s2= gp.quicksum(x_prime[p,g] for p in inst.preferiteParkingsPerGroup[1][g])
                model.addConstr(
                    s1+s2 <= inst.beta2 * demg[s][g] ,
                    f"max_x"
                )
            # CONSTRAINT (11)
            for g in groups :

                model.addConstr(
                    gp.quicksum(w[p,g] for p in inst.preferiteParkingsPerGroup[1][g]) <= inst.beta3 * demg[s][g] ,
                    f"max_z"
                )

            # CONSTRAINT (12)
            
            for t in inst.timeSlotList:
                for p in parkingLots:
                    model.addConstr(
                        gp.quicksum(y[p, g] for g in inst.G_2ND[t][p][0])+ gp.quicksum(x[p, g] for g in inst.G_2ND[t][p][2])+gp.quicksum(x_prime[p, g] for g in inst.G_2ND[t][p][4])+ gp.quicksum(w[p, g] for g in inst.G_2ND[t][p][6])<= gamma_prime[p],
                        f"max_gamma_prime"
                    )
            # CONSTRAINT (13)

            for t in inst.timeSlotList:
                for p in parkingLots:
                    model.addConstr(
                        gp.quicksum(y[p, g] for g in inst.G_2ND[t][p][1])+ gp.quicksum(x[p, g] for g in inst.G_2ND[t][p][3])+ gp.quicksum(x_prime[p, g] for g in inst.G_2ND[t][p][5])+ gp.quicksum(w[p, g] for g in inst.G_2ND[t][p][7])<=gamma_2nd[p],
                        f"max_gamma_2nd"
                    )
            model.update()
            model.setParam('OutputFlag', 0)
            model.setParam('LogFile', './logs/gurobi.log')
            model.optimize()
            if model.status == GRB.Status.OPTIMAL:
                ans.append(obj_fs + model.getObjective().getValue())
        return ans

    def in_sample_stability(self, problem, sampler, instance, n_repetitions, n_scenarios_sol):
        ans = [0] * n_repetitions
        print("\nSTART IN SAMPLE STABILITY:")
        for i in range(n_repetitions):
            print("Starting repetition", i+1, "of", n_repetitions, "...")
            demg = sampler.demg_sample_stoch(
                instance,
                n_scenarios=n_scenarios_sol,
                n_groups=instance.n_groups,
            )
            of, gamma_prime, gamma_2nd, delta_prime, delta_2nd, comp_time = problem.solve(
                instance.get_data(),
                demg,
                n_scenarios_sol
            )
            ans[i] = of
        return ans
    
    def out_of_sample_stability(self, problem, sampler, instance, n_repetitions, n_scenarios_sol, n_scenarios_out):
        ans = [0] * n_repetitions
        print("\nSTART OUT OF SAMPLE STABILITY:")
        for i in range(n_repetitions):
            print("Starting repetition", i+1, "of", n_repetitions, "...")
            demg = sampler.demg_sample_stoch(
                instance,
                n_scenarios=n_scenarios_sol,
                n_groups = instance.n_groups
            )
            of, gamma_prime, gamma_2nd, delta_prime, delta_2nd, comp_time = problem.solve(
                instance.get_data(),
                demg,
                n_scenarios_sol
            )
            demg_out = sampler.demg_sample_stoch(
                instance,
                n_scenarios=n_scenarios_out,
                n_groups = instance.n_groups
            )
            
            
            profits = self.solve_second_stages(
                instance, gamma_prime, gamma_2nd, delta_prime, delta_2nd,
                n_scenarios_out, demg_out
            )
            ans[i]=np.mean(profits)

            # Uncomment this section and comment from line 175 to 179 to perform stability testing for the heuristic model -> Heuristic had different format than math problem, so need to reinitialize data each repetition
            '''
            dict_data = instance.get_data()
            
            all_scanarios_fast_users = []
            all_scenarios_slow_users = []
            for s in range(n_scenarios_out):
                all_fast_users, all_slow_users = problem.return_users(dict_data, demg_out, s, dict_data["n_groups"])
                all_scanarios_fast_users.append(all_fast_users)
                all_scenarios_slow_users.append(all_slow_users)

            # Divide them for arrival time
            problem.all_scenario_divided_by_arrival_fast = []
            problem.all_scenario_divided_by_arrival_slow = []
            for s in range(n_scenarios_out):
                all_divided_by_arrival_fast, all_divided_by_arrival_slow = problem.divide_by_arrival\
                                            (dict_data, all_scanarios_fast_users[s], all_scenarios_slow_users[s])
                problem.all_scenario_divided_by_arrival_fast.append(all_divided_by_arrival_fast)
                
                problem.all_scenario_divided_by_arrival_slow.append(all_divided_by_arrival_slow)
            # Solve the heuristic problem

            fast_chargers_list, slow_chargers_list = problem.solve_first_stage(instance.get_data())

            num_users_using_fast, num_users_using_slow = problem.solve_second_stage\
            (instance.get_data(), demg_out, n_scenarios_out, fast_chargers_list, slow_chargers_list)

            # Calculate the outputs needed

            # Is a vector of binary variables that is 1 if tha parking lot p is equipped
            # with at least one fast charger, 0 otherwise
            index = [it for it, item in enumerate(fast_chargers_list) if item != 0]
            delta_prime_vect = [0 for item in fast_chargers_list]
            for it in index:
                delta_prime_vect[it] = 1

            # Is a vector of binary variables that is 1 if tha parking lot p is equipped
            # with at least one slow charger, 0 otherwise
            index = [it for it, item in enumerate(slow_chargers_list) if item != 0]
            delta_2nd_vect = [0 for item in slow_chargers_list]
            for it in index:
                delta_2nd_vect[it] = 1

            # Number of fast chargers installed in parking lot p
            gamma_prime_vect = fast_chargers_list

            # Number of slow chargers installed in parking lot p
            gamma_2nd_vect = slow_chargers_list

            # Objective function value
            # Fixed cost of equipping a parking lot with fast chargers
            
            F_prime = dict_data['F_prime']
            # Fixed cost of equipping a parking lot with slow chargers
            F_2nd = dict_data['F_2nd']
            # The cost of purchasing a fast charger
            C_prime = dict_data['C_prime']
            # The cost of purchasing a slow charger
            C_2nd = dict_data['C_2nd']
            # Evaluate the fixed part of the objective function
            fixed_part = - (sum(F_prime*delta_prime_vect) + sum(F_2nd*delta_2nd_vect)+ \
                sum(C_prime*gamma_prime_vect) + sum(C_2nd*gamma_2nd_vect))
            # Evaluate the variable part of the objective function
            # The income achieved by charging an EV by a fast charger
            E_prime = dict_data['Eprime']
            # The income achieved by charging an EV by a slow charger
            E_2nd = dict_data['E_2nd']
            # Duration of short, medium, and long dwell times respectively
            theta1 = dict_data['theta1']
            theta2 = dict_data['theta2']
            theta3 = dict_data['theta3']
            variable_part_fast = E_prime * (theta1*num_users_using_fast['theta1'] + theta2*num_users_using_fast['theta2'] +\
                theta3*num_users_using_fast['theta3'])
            variable_part_slow = E_2nd * (theta1*num_users_using_slow['theta1'] + theta2*num_users_using_slow['theta2'] +\
                theta3*num_users_using_slow['theta3'])
            variable_part = variable_part_fast + variable_part_slow
            # Suppose that every scenario has the same probability
            variable_part = variable_part/n_scenarios_out
            # Final value of the objective function
            of = fixed_part + variable_part
            ans[i] = of '''
        return ans
