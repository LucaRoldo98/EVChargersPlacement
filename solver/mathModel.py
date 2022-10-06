# -*- coding: utf-8 -*-
import time
import logging
import gurobipy as gp
from gurobipy import GRB
import random
import numpy as np

class mathModel():
    def __init__(self):
        pass

    def solve(
        self, dict_data, demg, n_scenarios, time_limit=None,
        gap=None, verbose=False
    ):
        parkingLots= range(0,dict_data['n_parkingLots'])
        groups= range(0,dict_data['n_groups'])
        scenarios = range(0,n_scenarios)
        times = dict_data["timeSlotList"]
        G_PRIME = dict_data["G_prime"]
        G_2ND =dict_data["G_2ND"]
        problem_name = "EVChargersOptimalPlacing"
        logging.info("{}".format(problem_name))
        preferiteParkingsPerGroup=dict_data["preferiteParkingsPerGroup"]
        # logging.info(f"{problem_name}")

        model = gp.Model(problem_name)
        x = model.addVars(
            dict_data['n_parkingLots'],dict_data['n_groups'],n_scenarios,
            lb=0,
            ub=GRB.INFINITY,
            vtype=GRB.INTEGER,
            name='x'
        )
        y = model.addVars(
            dict_data['n_parkingLots'],dict_data['n_groups'],n_scenarios,
            lb=0,
            ub=GRB.INFINITY,
            vtype=GRB.INTEGER,
            name='y'
        )
        w = model.addVars(
            dict_data['n_parkingLots'],dict_data['n_groups'],n_scenarios,
            lb=0,
            ub=GRB.INFINITY,
            vtype=GRB.INTEGER,
            name='w'
        )
        x_prime = model.addVars(
            dict_data['n_parkingLots'],dict_data['n_groups'],n_scenarios,
            lb=0,
            ub=GRB.INFINITY,
            vtype=GRB.INTEGER,
            name='x_prime'
        )
        delta = model.addVars(
            dict_data['n_parkingLots'],
            lb=0,
            ub=1,
            vtype=GRB.BINARY,
            name='delta'
        )
        delta_prime = model.addVars(
            dict_data['n_parkingLots'],
            lb=0,
            ub=1,
            vtype=GRB.BINARY,
            name='delta_prime'
        )
        delta_2nd = model.addVars(
            dict_data['n_parkingLots'],
            lb=0,
            ub=1,
            vtype=GRB.BINARY,
            name='delta_2nd'
        )
        gamma_prime = model.addVars(
            dict_data['n_parkingLots'],
            lb=0,
            ub=GRB.INFINITY,
            vtype=GRB.INTEGER,
            name='gamma_prime'
        )
        gamma_2nd = model.addVars(
            dict_data['n_parkingLots'],
            lb=0,
            ub=GRB.INFINITY,
            vtype=GRB.INTEGER,
            name='gamma_2nd'
        )


        obj_funct=0
        obj_funct_firstStage = -(gp.quicksum(
            dict_data['F_prime'] * delta_prime[p] + dict_data['F_2nd'] * delta_2nd[p] + dict_data['C_prime'] *
            gamma_prime[p] + dict_data['C_2nd'] * gamma_2nd[p] for p in parkingLots))
        for s in scenarios:
            for p in parkingLots:
                obj_funct = obj_funct + dict_data['Eprime']* dict_data['theta1'] * gp.quicksum((y[p, g, s]+x[p, g, s]) for g in G_PRIME[0][0][p])
                obj_funct = obj_funct + dict_data['Eprime'] * dict_data['theta1'] * gp.quicksum((w[p, g, s] + x_prime[p, g, s])  for g in G_PRIME[1][0][p])
                obj_funct = obj_funct + dict_data['Eprime']* dict_data['theta2'] * gp.quicksum(y[p, g, s] for g in G_PRIME[0][1][p])
                obj_funct = obj_funct + dict_data['Eprime'] * dict_data['theta2'] * gp.quicksum(x_prime[p, g, s] for g in G_PRIME[1][1][p])
                obj_funct = obj_funct + dict_data['E_2nd'] * dict_data['theta2'] * gp.quicksum(x[p, g, s] for g in G_PRIME[0][1][p])
                obj_funct = obj_funct + dict_data['E_2nd'] * dict_data['theta2'] * gp.quicksum(w[p, g, s] for g in G_PRIME[1][1][p])
                obj_funct = obj_funct + dict_data['E_2nd'] * dict_data['theta3'] * gp.quicksum((x[p, g, s] + y[p, g, s]) for g in G_PRIME[0][2][p])
                obj_funct = obj_funct + dict_data['E_2nd'] * dict_data['theta3'] * gp.quicksum((w[p, g, s] + x_prime[p,g,s])  for g in G_PRIME[1][2][p])

        obj_funct=obj_funct*(1/n_scenarios)
        obj_funct += obj_funct_firstStage
        model.setObjective(obj_funct, GRB.MAXIMIZE)

        #CONSTRAINT (2)
        model.addConstr(
            gp.quicksum(delta[p] for p in parkingLots) <= dict_data['M'],
            f"max_park_equipped"
        )
        # CONSTRAINT (4)
        for p in parkingLots:
            model.addConstr(
                 delta_prime[p]+delta_2nd[p] <= 2*delta[p],
                f"slot_equipement_constraint_{p}"
            )
        # CONSTRAINT (5)
        for p in parkingLots:
            model.addConstr(
                 gamma_prime[p]<= dict_data['M_2nd']*delta_prime[p],
                f"fast_chargers_per_slot_constraint_{p}"
            )
        # CONSTRAINT (6)
        for p in parkingLots:
            model.addConstr(
                 gamma_2nd[p]<= dict_data['M_2nd']*delta_2nd[p],
                f"slow_chargers_per_slot_constraint_{p}"
            )
        # CONSTRAINT (7)
        for p in parkingLots:
            model.addConstr(
                 gamma_prime[p]+gamma_2nd[p] <= dict_data['M_2nd'],
                f"total_chargers_per_slot_constraint_{p}"
            )
        # CONSTRAINT (8)
        model.addConstr(
            gp.quicksum(gamma_prime[p] +gamma_2nd[p] for p in parkingLots) <= dict_data['M_prime'],
            f"max_park_equipped_2"
        )

        # CONSTRAINT (9)
        for s in scenarios:
            for g in groups :
                model.addConstr(
                    gp.quicksum(y[p,g,s] for p in preferiteParkingsPerGroup[0][g]) <= dict_data['beta1']* demg[s][g] ,
                    f"max_y_{s}_{g}"
                )


        # CONSTRAINT (10)
        for s in scenarios:
            for g in groups :
                s1= gp.quicksum(x[p,g,s] for p in preferiteParkingsPerGroup[0][g])
                s2= gp.quicksum(x_prime[p,g,s] for p in preferiteParkingsPerGroup[1][g])
                model.addConstr(
                    s1+s2 <= dict_data['beta2']* demg[s][g] ,
                    f"max_x_{s}_{g}"
                )
        # CONSTRAINT (11)
        for s in scenarios:
            for g in groups :

                model.addConstr(
                    gp.quicksum(w[p,g,s] for p in preferiteParkingsPerGroup[1][g]) <= dict_data['beta3']* demg[s][g] ,
                    f"max_w_{s}_{g}"
                )

        # CONSTRAINT (12)

        for s in scenarios:
            for t in times:
                for p in parkingLots:
                    model.addConstr(
                        gp.quicksum(y[p, g, s] for g in G_2ND[t][p][0])+ gp.quicksum(x[p, g, s] for g in G_2ND[t][p][2])+gp.quicksum(x_prime[p, g, s] for g in G_2ND[t][p][4])+ gp.quicksum(w[p, g, s] for g in G_2ND[t][p][6])<= gamma_prime[p],
                        f"max_gamma_prime_{s}_{t}_{p}"
                    )
        # CONSTRAINT (13)

        for s in scenarios:
            for t in times:
                for p in parkingLots:

                    model.addConstr(
                        gp.quicksum(y[p, g, s] for g in G_2ND[t][p][1])+ gp.quicksum(x[p, g, s] for g in G_2ND[t][p][3])+ gp.quicksum(x_prime[p, g, s] for g in G_2ND[t][p][5])+ gp.quicksum(w[p, g, s] for g in G_2ND[t][p][7])<=gamma_2nd[p],
                        f"max_gamma_2nd_{s}_{t}_{p}"
                    )
        
        model.update()
        if gap:
            model.setParam('MIPgap', gap)
        if time_limit:
            model.setParam(GRB.Param.TimeLimit, time_limit)
        if verbose:
            model.setParam('OutputFlag', 1)
        else:
            model.setParam('OutputFlag', 0)
        model.setParam('LogFile', './logs/gurobi.log')
        model.write("./logs/model.lp")

        start = time.time()
        model.optimize()
        end = time.time()
        comp_time = end - start
        
        sol1 = [0] * len(parkingLots)
        sol2 = [0] * len(parkingLots)
        delta_prime_vect = [0] * len(parkingLots)
        delta_2nd_vect = [0] * len(parkingLots)
        
        of = -1
        
        if model.status == GRB.Status.OPTIMAL:
            for i in parkingLots:
                grb_var1 = model.getVarByName(
                    f'gamma_prime[{i}]'
                )
                sol1[i] = grb_var1.X
                grb_var2 = model.getVarByName(
                    f'gamma_2nd[{i}]'
                )
                sol2[i] = grb_var2.X
                delta_prime_vect[i] = model.getVarByName(f"delta_prime[{i}]").X
                delta_2nd_vect[i] = model.getVarByName(f"delta_2nd[{i}]").X
            of = model.getObjective().getValue()

        # Spostato nel main visto che la soluzione ci serve completa 
        '''gamma_prime_sol=[]
        gamma_prime_vector = np.zeros(dict_data["n_parkingLots"])
        gamma_2ND_sol=[]
        gamma_2ND_vector = np.zeros(dict_data["n_parkingLots"])
        for i,val in enumerate(sol1):
            gamma_prime_vector[i] = val
            if val != 0.0 or val !=-0.0:
                gamma_prime_sol.append(dict(index=i,val=val))
        for i,val in enumerate(sol2):
            gamma_2ND_vector[i] = val
            if val != 0.0 or val !=-0.0:
                gamma_2ND_sol.append(dict(index=i,val=val))'''
        gamma_prime_vect = sol1
        gamma_2nd_vect = sol2
        
        return of, gamma_prime_vect, gamma_2nd_vect, delta_prime_vect, delta_2nd_vect, comp_time
