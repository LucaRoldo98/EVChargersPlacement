# -*- coding: utf-8 -*-
import time
import math
import sys
import logging
import numpy as np
import gurobipy as gp
from gurobipy import GRB

class onlyFirstStageHeu():

    # Initialize all the structures needed in the evaluation of the heuristics
    # These are the structures that are built one and for all in the heuristics
    def __init__(self):
        pass

    def solve_first_stage(self, dict_data):
        tot = 0 
        slow_chargers = np.zeros((dict_data['n_parkingLots'])) # list where there is the number of slow charger for each parking
        fast_chargers = np.zeros((dict_data['n_parkingLots'])) # list where there is the number of fast charger for each parking 
        chargPerParking = []
        # total number of available chargers 
        M_prime = dict_data['M_prime'] 
        # max number of parking that can be equipped with chargers
        M = dict_data['M']
        # max number of chargers that can be installed in any parking lots 
        M_2nd = dict_data['M_2nd']

        # The maximum number of chargers that can be placed based on the constraints M, M_prime and M_second 
        maxTotChargers = min(M_prime, M_2nd * M)
        
        # divide the portion of tot charger in fast and slow based on values of beta1, beta2, beta3 
        # divide beta2 into half because half of users uses fast charger, the other half uses slow 
        # and use same with beta1 and beta3
        perc_slow = dict_data["beta2"]/2 + dict_data["beta1"]/3 + (dict_data["beta3"]*2)/3  
        perc_fast = dict_data["beta2"]/2 + (dict_data["beta1"]*2)/3 + (dict_data["beta3"])/3  
        
        # based on the price and the revenue calculate the percentage 
        price_slow = dict_data['Eprime']/(dict_data['C_prime'] + dict_data['F_prime'])
        price_fast = dict_data['E_2nd']/(dict_data['C_2nd'] + dict_data['F_2nd'])
        tot_price = price_slow + price_fast
        price_slow = price_slow/tot_price
        price_fast = price_fast/tot_price
        # divide the tot available chargers in available slow and fast 
        av_fast = math.floor(maxTotChargers * (perc_slow + price_slow)/2)
        av_slow = math.floor(maxTotChargers * (perc_fast + price_fast)/2)

        start = time.time()

        # put to most popular buildings max chargers per parkings, first put all fast chargers
        # then put all slow chargers 
        for i,item in enumerate(dict_data['buildingPerParking']):
            if i>=M: 
                break
            else:
                chargPerParking.append(item)
                chargPerParking[i]['fast_char'] = 0
                chargPerParking[i]['slow_char'] = 0
        
                if av_fast >= M_2nd:  
                    chargPerParking[i]['fast_char'] = M_2nd
                    av_fast = av_fast - M_2nd 
                elif av_fast < M_2nd and av_fast>0 and av_slow>=(M_2nd-av_fast): 
                    chargPerParking[i]['fast_char'] = av_fast 
                    chargPerParking[i]['slow_char'] = M_2nd-av_fast
                    av_slow = av_slow - (M_2nd-av_fast)
                    av_fast = 0 
                elif av_fast==0 and av_slow>=M_2nd:
                    chargPerParking[i]['slow_char'] = M_2nd
                    av_slow = av_slow - M_2nd
                elif av_fast==0 and av_slow<M_2nd and av_slow>0:
                    chargPerParking[i]['slow_char'] = av_slow
                    av_slow = 0
                else: 
                    chargPerParking[i]['slow_char'] = 0
    
        for i,item in enumerate(chargPerParking):
            if item['fast_char']>0:
                index = int(str(item['name']).split('_')[1])
                fast_chargers[index] = item['fast_char']
            if item['slow_char']>0:
                index = int(str(item['name']).split('_')[1])
                slow_chargers[index] = item['slow_char']  

        return fast_chargers, slow_chargers
        
    def solve(self, instance, demg, n_scenarios):
        start = time.time()
        
        dict_data = instance.get_data()

        fast_chargers_list, slow_chargers_list = self.solve_first_stage(dict_data)

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

        finalProfits = self.exact_solve_second_stages(instance, gamma_prime_vect, gamma_2nd_vect, delta_prime_vect, delta_2nd_vect, n_scenarios, demg)

        of = np.mean(finalProfits)

        if of < 0:
            of = 0
            gamma_prime_vect = [0 for item in fast_chargers_list]
            gamma_2nd_vect = [0 for item in slow_chargers_list]
            delta_prime_vect = [0 for item in fast_chargers_list]
            delta_2nd_vect = [0 for item in slow_chargers_list]

        end = time.time()
        comp_time = end - start
        return of, gamma_prime_vect, gamma_2nd_vect, delta_prime_vect, delta_2nd_vect, comp_time


    def exact_solve_second_stages(
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