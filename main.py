#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import logging
import numpy as np
from simulator.instance import Instance
from simulator.tester import Tester
from solver.mathModel import mathModel
from heuristic.simpleHeu import SimpleHeu
from heuristic.ImprovedHeu import ImprovedHeu
from heuristic.onlyFirstStageHeu import onlyFirstStageHeu
from solver.sampler import Sampler
import time
from utility.plot_results import plot_comparison_bar, plot_comparison_hist

np.random.seed(0)

if __name__ == '__main__':
    log_name = "./logs/main.log"
    logging.basicConfig(
        filename=log_name,
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO, datefmt="%H:%M:%S",
        filemode='w'
    )

    fp = open("./etc/sim_setting.json", 'r')
    sim_setting = json.load(fp)
    fp.close()

    sam = Sampler()

    inst = Instance(sim_setting)
    dict_data = inst.get_data()

    # Reward generation
    n_scenarios = sim_setting['n_scenarios']
    n_groups= sim_setting['n_groups']
    demg = sam.demg_sample_stoch(
        inst,
        n_scenarios=n_scenarios,
        n_groups=n_groups
    )
    print("Mean demg:", sam.demg_mean(inst, n_scenarios, n_groups))
    print("Total population mean:", sam.tot_pop_mean(inst, n_scenarios, n_groups))
    
    prb = mathModel()
    of_exact, gamma_prime_vect_exact, gamma_second_vect_exact, delta_prime_vect_exact, delta_2nd_vect_exact, comp_time_exact = prb.solve(
        dict_data,
        demg,
        n_scenarios,
        verbose=True
    )

    sol_exact1=[]
    sol_exact2=[]
    for i,val in enumerate(gamma_prime_vect_exact): 
        if val != 0.0 or val !=-0.0:
            sol_exact1.append(dict(index=i,val=val))
    for i,val in enumerate(gamma_second_vect_exact):
        if val != 0.0 or val !=-0.0:
            sol_exact2.append(dict(index=i,val=val))
    print("\n\t\t\t*** RESULTS MATH MODEL ***\n")
    print("Objective function:", of_exact)
    print("Fast chargers:", sol_exact1)
    print("Slow chargers:", sol_exact2)
    print("Completion time:", comp_time_exact)
    #print(of_exact, sol_exact1, sol_exact2, comp_time_exact)
     # COMPARISON:
    test = Tester()
    
    # Uncomment this section to test in-sample and out-of-sample stabilities for the exact model 
    '''
    in_sample = test.in_sample_stability(prb, sam, inst, 10, 50)
    print("\nIn-sample stability:", in_sample)
    labels = [i+1 for i in range(10)]
    plot_comparison_bar(in_sample, labels, "#run", "obj. function", "In sample stability - Exact model")
    
    
    out_of_sample = test.out_of_sample_stability(prb, sam, inst, 10, 50, 100)
    print("\nOut-of-sample stability:", out_of_sample)
    
    labels = [i+1 for i in range(10)]
    plot_comparison_bar(out_of_sample, labels, "#run", "obj. function", "Out-of-sample stability - Exact model")
    '''
    # Uncomment this section to compare the histograms of two scenario trees 
    '''
    n_scenarios = 1000
    demgTest1 = sam.demg_sample_stoch(
        inst,
        n_scenarios=n_scenarios,
        n_groups=n_groups
    )

    ris1 = test.solve_second_stages(
        inst,
        gamma_prime_vect_exact,
        gamma_second_vect_exact, 
        delta_prime_vect_exact, 
        delta_2nd_vect_exact,
        n_scenarios,
        demgTest1
    )

    demgTest2 = sam.demg_sample_stoch(
        inst,
        n_scenarios=n_scenarios,
        n_groups=n_groups
    )
    ris2 = test.solve_second_stages(
        inst,
        gamma_prime_vect_exact,
        gamma_second_vect_exact, 
        delta_prime_vect_exact, 
        delta_2nd_vect_exact,
        n_scenarios,
        demgTest2
    )
    

    plot_comparison_hist(
        [ris1, ris2],
        ["run1", "run2"],
        ['red', 'blue'],
        "profit", "occurencies"
    )
    '''
    
    # Only first Stage Simple Heuristic
    
    prb = onlyFirstStageHeu()
    
    of_heuFirst, gamma_prime_vect_heuFirst, gamma_second_vect_heuFirst, delta_prime_vect_heuFirst, delta_2nd_vect_heuFirst, comp_time_heuFirst = prb.solve(
        inst,
        demg,
        n_scenarios
    )

    sol_heuFirst1=[]
    sol_heuFirst2=[]
    for i,val in enumerate(gamma_prime_vect_heuFirst): 
        if val != 0.0 or val !=-0.0:
            sol_heuFirst1.append(dict(index=i,val=val))
    for i,val in enumerate(gamma_second_vect_heuFirst):
        if val != 0.0 or val !=-0.0:
            sol_heuFirst2.append(dict(index=i,val=val))
    print("\n\t\t\t*** RESULTS ONLY FIRST STAGE HEURISTIC MODEL ***\n")
    print("Objective function:", of_heuFirst)
    print("Fast chargers:", sol_heuFirst1)
    print("Slow chargers:", sol_heuFirst2)
    print("Completion time:", comp_time_heuFirst) 

    # Simple Heuristic
    
    prb = SimpleHeu()
    
    of_heu, gamma_prime_vect_heu, gamma_second_vect_heu, delta_prime_vect_heu, delta_2nd_vect_heu, comp_time_heu = prb.solve(
        dict_data,
        demg,
        n_scenarios
    )

    sol_heu1=[]
    sol_heu2=[]
    for i,val in enumerate(gamma_prime_vect_heu): 
        if val != 0.0 or val !=-0.0:
            sol_heu1.append(dict(index=i,val=val))
    for i,val in enumerate(gamma_second_vect_heu):
        if val != 0.0 or val !=-0.0:
            sol_heu2.append(dict(index=i,val=val))
    print("\n\t\t\t*** RESULTS SIMPLE HEURISTIC MODEL ***\n")
    print("Objective function:", of_heu)
    print("Fast chargers:", sol_heu1)
    print("Slow chargers:", sol_heu2)
    print("Completion time:", comp_time_heu) 

    # Uncomment this section to test in-sample and out-of-sample stabilities of the simple heuristic
    '''
    in_sample = test.in_sample_stability(prb, sam, inst, 10, 50)
    print("\nIn-sample stability:", in_sample)
    labels = [i+1 for i in range(10)]
    plot_comparison_bar(in_sample, labels, "#run", "obj. function", "In sample stability - Simple heuristic")
    
    out_of_sample = test.out_of_sample_stability(prb, sam, inst, 10, 50, 200)
    print("\nOut-of-sample stability:", out_of_sample)
    
    labels = [i+1 for i in range(10)]
    plot_comparison_bar(out_of_sample, labels, "#run", "obj. function", "Out-of-sample stability - Simple heuristic")
    '''


    # Improved Heuristic
    prb = ImprovedHeu()
    of_iheu, gamma_prime_vect_iheu, gamma_second_vect_iheu, delta_prime_vect_iheu, delta_2nd_vect_iheu, comp_time_iheu = prb.solve(
        dict_data,
        demg,
        n_scenarios
    )

    sol_iheu1=[]
    sol_iheu2=[]
    for i,val in enumerate(gamma_prime_vect_iheu): 
        if val != 0.0 or val !=-0.0:
            sol_iheu1.append(dict(index=i,val=val))
    for i,val in enumerate(gamma_second_vect_iheu):
        if val != 0.0 or val !=-0.0:
            sol_iheu2.append(dict(index=i,val=val))
    print("\n\t\t\t*** RESULTS IMPROVED HEURISTIC MODEL ***\n")
    print("Objective function:", of_iheu)
    print("Fast chargers:", sol_iheu1)
    print("Slow chargers:", sol_iheu2)
    print("Completion time:", comp_time_iheu)

    # printing results of a file
    file_output = open(
        "./results/exp_general_table.csv",
        "w"
    )
    file_output.write("method, of, time, fast_chargers, slow_chargers\n")
    file_output.write("{}, {}, {}, {}, {}\n".format(
        "exact", of_exact,  comp_time_exact, gamma_prime_vect_exact, gamma_second_vect_exact
    ))
    file_output.write("{}, {}, {}, {}, {}\n".format(
        "heuOnlyFirstStage", of_heuFirst, comp_time_heuFirst, gamma_prime_vect_heuFirst, gamma_second_vect_heuFirst
    ))
    file_output.write("{}, {}, {}, {}, {}\n".format(
        "simpleHeu", of_heu, comp_time_heu, gamma_prime_vect_heu, gamma_second_vect_heu
    ))
    file_output.write("{}, {}, {}, {}, {}\n".format(
        "simpleHeu", of_iheu, comp_time_iheu, gamma_prime_vect_iheu, gamma_second_vect_iheu
    ))
    file_output.close()
    
    