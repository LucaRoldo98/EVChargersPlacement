# -*- coding: utf-8 -*-
import time
import math
import sys
import logging
import numpy as np


class ImprovedHeu():

    # Initialize all the structures needed in the evaluation of the heuristics
    # These are the structures that are built one and for all in the heuristics
    def __init__(self):
        pass



    # Return all the fast and slow users divided by the parameters 
    # within myDict
    def return_users(self, dict_data, demg, s, n_groups):
        # All the different fast users in the system
        all_fast_users = []
        # All the different slow users in the system
        all_slow_users = []
        # Divide the users accordingly to the structure myDict
        for it in range(n_groups):
            for n_building in [0, 1]:
                for n_class in dict_data['classes']:
                    # User of class one cannot recharge in the second building
                    if not(n_class == 1 and n_building == 1):
                        # User of class three cannot recharge in the first building
                        if not((n_class == 3 and n_building == 0)):
                        # Note: the users of class 2 recharge either in the first 
                        # or in the second building
                            # Structure that defines the parameters according to which
                            # the users are divided
                            myDict = {
                                "building": None,
                                "arrival_time_slot": None,
                                "charging_mode": None,
                                "dwell_time": None,
                                "number_of_users": 0,
                                "class": None,
                                # Used to bound the first building to the second one
                                "num_group": None
                            }
                            # Set building ID
                            myDict['building'] = dict_data['groups'][it]['building'][n_building]
                            # Set arrival time slot
                            if n_building == 0:
                                myDict['arrival_time_slot'] = dict_data['groups'][it]['arrival_time_slot']
                            else:
                                if dict_data['groups'][it]['dwell_time'][n_building] == 'short':
                                    dwell_time = dict_data['theta1']
                                elif dict_data['groups'][it]['dwell_time'][n_building] == 'medium':
                                    dwell_time = dict_data['theta2']
                                else:
                                    dwell_time = dict_data['theta3']
                                myDict['arrival_time_slot'] = int(dict_data['groups'][it]['arrival_time_slot'] \
                                                            + dwell_time + dict_data['groups'][it]['travel_time'])
                            # Set dwell time
                            myDict['dwell_time'] = dict_data['groups'][it]['dwell_time'][n_building]
                            # Set class
                            myDict['class'] = n_class
                            # Set the number of the group
                            myDict['num_group'] = it
                            # Set number_of_users
                            if n_class == 1:
                                number_of_users = math.ceil(dict_data['beta1'] * demg[s][it])
                            elif n_class == 2:
                                number_of_users = math.ceil(dict_data['beta2'] * demg[s][it])
                            else:
                                number_of_users = math.ceil(dict_data['beta3'] * demg[s][it])
                            myDict['number_of_users'] = number_of_users
                            # Set charging mode of used charger
                            for types in dict_data['types']:
                                if types['class'] == n_class:
                                    if myDict['dwell_time'] in types['dwellTime']:
                                        if n_building == 0:
                                            building = 'first'
                                        else:
                                            building = 'second'
                                        if building == types['building']:
                                            myDict['charging_mode'] = types['chargingMode']
                            # Append to final structure
                            if myDict['charging_mode'] == 'fast':
                                all_fast_users.append(myDict)
                            else:
                                all_slow_users.append(myDict)
        return all_fast_users, all_slow_users



    # Divide all the users considering the time slot in which they arrive
    def divide_by_arrival(self, dict_data, all_scanarios_fast_users, all_scenarios_slow_users):
        all_divided_by_arrival_fast = []
        all_divided_by_arrival_slow = []
        for t in range(dict_data['time_slots']):
            time_slot_arrival = []
            for users in all_scanarios_fast_users:
                if users['arrival_time_slot'] == t:
                    time_slot_arrival.append(users)
            all_divided_by_arrival_fast.append(time_slot_arrival)
        for t in range(dict_data['time_slots']):
            time_slot_arrival = []
            for users in all_scenarios_slow_users:
                if users['arrival_time_slot'] == t:
                    time_slot_arrival.append(users)
            all_divided_by_arrival_slow.append(time_slot_arrival)
        return all_divided_by_arrival_fast, all_divided_by_arrival_slow




    def solve(self, dict_data, demg, n_scenarios):
        self.comp_time = 0
        # Obtain all fast and slow users for each scenario
        all_scanarios_fast_users = []
        all_scenarios_slow_users = []
        for s in range(n_scenarios):
            all_fast_users, all_slow_users = self.return_users(dict_data, demg, s, dict_data["n_groups"])
            all_scanarios_fast_users.append(all_fast_users)
            all_scenarios_slow_users.append(all_slow_users)

        # Divide them for arrival time
        self.all_scenario_divided_by_arrival_fast = []
        self.all_scenario_divided_by_arrival_slow = []
        for s in range(n_scenarios):
            all_divided_by_arrival_fast, all_divided_by_arrival_slow = self.divide_by_arrival\
                                        (dict_data, all_scanarios_fast_users[s], all_scenarios_slow_users[s])
            self.all_scenario_divided_by_arrival_fast.append(all_divided_by_arrival_fast)
            self.all_scenario_divided_by_arrival_slow.append(all_divided_by_arrival_slow)
        # Solve the heuristic problem
        fast_chargers_list, slow_chargers_list = self.solve_first_stage(dict_data)

        num_users_using_fast, num_users_using_slow = self.solve_second_stage\
            (dict_data, demg, n_scenarios, fast_chargers_list, slow_chargers_list)

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
        variable_part = variable_part/n_scenarios
        # Final value of the objective function
        of = fixed_part + variable_part
        if of < 0:
            of = 0
            gamma_prime_vect = [0 for item in fast_chargers_list]
            gamma_2nd_vect = [0 for item in slow_chargers_list]
            delta_prime_vect = [0 for item in fast_chargers_list]
            delta_2nd_vect = [0 for item in slow_chargers_list]

        return of, gamma_prime_vect, gamma_2nd_vect, delta_prime_vect, delta_2nd_vect, self.comp_time




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
        
        
        end = time.time()
        self.comp_time = self.comp_time + end - start

        return fast_chargers, slow_chargers 
        




    def solve_second_stage(self, dict_data, demg, n_scenarios, fast_chargers_list, slow_chargers_list):
        # Create the needed data structure based on the output of the first stage
        self.parks_with_fast, self.number_of_fast = self.return_used_parks(fast_chargers_list)
        self.parks_with_slow, self.number_of_slow = self.return_used_parks(slow_chargers_list)
        # Create a matrix parkings-buildings, where the value is 1 if the users can reach the
        # considered parking from the selected building, else 0.
        Cs = self.return_Cmatrix(self.parks_with_slow, dict_data['buildings'], dict_data['parkingsPerBuilding'])
        Cf = self.return_Cmatrix(self.parks_with_fast, dict_data['buildings'], dict_data['parkingsPerBuilding'])
        #Variables needed to evaluate the objective function
        num_users_using_fast = {
            "theta1": 0,
            "theta2": 0,
            "theta3": 0
        }
        num_users_using_slow = {
            "theta1": 0,
            "theta2": 0,
            "theta3": 0
        }

        start = time.time()

        # Iterate on all the possible scenario
        for s in range(n_scenarios):
            # Iterate on all the possible time slots
            # Initialize S_f and S_s, where S is the set of all the activated fast and slow charging sites respectively
            S_f = []
            S_s = []
            # Initialize R_f and R_s, where R is the set of all the activated fast and slow charging sites respectively,
            # that are not completely full
            R_f = []
            R_s = []
            # Initialize the users that must leave
            self.to_remove_fast = list([] for item in range(int(dict_data['time_slots'])))
            self.to_remove_slow = list([] for item in range(int(dict_data['time_slots'])))
            # O_f is the set of all the occupied fast chargers per parking lots
            O_f = list(0 for item in range(len(self.parks_with_fast)))
            # O_s is the set of all the occupied slow chargers per parking lots
            O_s = list(0 for item in range(len(self.parks_with_slow)))
            # Note: the users of class 2 recharge either in the first or in the second building.
            # Structure that keeps track of the number of people of the second class that have already
            # recharged their car in the first building (no need of charging it in the second one)
            self.already_recharged = []
            # Iterate on all the time slots
            for t in range(int(dict_data['time_slots'])):
                #print(f"SCENARIO CONSIDERED: {s}")
                #print(f"TIMESLOT CONSIDERED: {t}")
                # Initialize all the variable needed for each iteration
                # K_f is the set of users that arrive in this time slot and that need the fast charging
                K_f = self.all_scenario_divided_by_arrival_fast[s][t]
                # K_s is the set of users that arrive in this time slot and that need the slow charging
                K_s = self.all_scenario_divided_by_arrival_slow[s][t]
                # Calculate EVD for each subgroup of users, that is the number of parking
                # lots in which the group of users can park
                EVD_f = self.return_EVD(Cf, K_f)
                EVD_s = self.return_EVD(Cs, K_s)
                # P_f is the set of all the available fast charging sites (parking lots with at least one fast charger)
                # that are not analyzed yet
                P_f = self.parks_with_fast.copy()
                # P_s is the set of all the available slow charging sites (parking lots with at least one slow charger)
                # that are not analyzed yet
                P_s = self.parks_with_slow.copy()

                # Delete all the users that should leave in this time slot
                # Remove all the fast users
                O_f, S_f, R_f = self.remove_users(self.to_remove_fast[t], O_f, S_f, R_f)
                # Remove all the slow users
                O_s, S_s, R_s = self.remove_users(self.to_remove_slow[t], O_s, S_s, R_s)

                # Remove people of class 2 that has been already served
                K_f = self.remove_already_served(K_f)
                K_s = self.remove_already_served(K_s)

                # Add the users that arrive in this time slot that need fast chargers (high priority)
                #print("FIRST SUB-PROBLEM STARTING ...")
                K_f, S_f, R_f, O_f = self.solve_second_stage_subproblem(0, K_f, P_f.copy(), S_f, R_f, O_f,\
                     EVD_f.copy(), Cf, dict_data, t, num_users_using_fast)
	            # Add the users that arrive in this time slot that need slow chargers (lower priority)
                #print("SECOND SUB-PROBLEM STARTING ...")
                K_s, S_s, R_s, O_s = self.solve_second_stage_subproblem(1, K_s, P_s.copy(), S_s, R_s, O_s,\
                     EVD_s.copy(), Cs, dict_data, t, num_users_using_slow)
                # Add the users that arrive in this time slot that need slow chargers, but that find them
                # all occupied, to the fast chargers (if some of them is free)
                #print("THIRD SUB-PROBLEM STARTING ...")
                EVF_s_that_parks_in_f = self.return_EVD(Cf, K_s)
                K_s, S_f, R_f, O_f = self.solve_second_stage_subproblem(0, K_s, P_f.copy(), S_f, R_f, O_f,\
                     EVF_s_that_parks_in_f.copy(), Cf, dict_data, t, num_users_using_fast)
                     
        end = time.time()
        self.comp_time = self.comp_time + end - start

        return num_users_using_fast, num_users_using_slow




    # Remove already served users of class 2
    def remove_already_served(self, K):
        for item in K:
            if item['class'] == 2:
                for it, served in enumerate(self.already_recharged):
                    if item['num_group'] == self.already_recharged[it][0]:
                        item['number_of_users'] = item['number_of_users'] - self.already_recharged[it][1]
        return K




    # Returns all the parkings with at least a charger and the number of chargers per parking lots
    def return_used_parks(self, first_stage_solution):
        parkings = []
        slots = []
        for it, n_slots in enumerate(first_stage_solution):
            if n_slots != 0:
                parkings.append(it)
                slots.append(n_slots)
        return parkings, slots



    # Return a matrix parkings-buildings, where the value is 1 if the users can reach the
    # considered parking from the selected building, else 0. 
    def return_Cmatrix(self, parking_lots, buildings, parkingsPerBuilding):
        Cmatrix = np.zeros((len(parking_lots), len(buildings)))
        for it_p, parking in enumerate(parking_lots):
            for it_b, building in enumerate(buildings):
                for item in parkingsPerBuilding:
                    if item['name'] == building:
                        if parking in item['parks']:
                            Cmatrix[it_p][it_b] = 1
        return Cmatrix



    # Return the list of EVD, that is the number of parking lots in which the group of users can park
    def return_EVD(self, Cmatrix, K_set):
        EVD = []
        for group in K_set:
            EVD_i = 0
            buildings = int(group['building'].split('_')[1])
            for parkings in range(Cmatrix.shape[0]):
                EVD_i = EVD_i + Cmatrix[parkings][buildings]
            EVD.append(int(EVD_i))
        return EVD



    # Return the list of CSD, where CSD is the number of users that can reach the selected parking lot
    def return_CSD(self, Cmatrix, K_set):
        CSD = []
        for parkings in range(Cmatrix.shape[0]):
            CSD_i = 0
            for group in K_set:
                buildings = int(group['building'].split('_')[1])
                CSD_i = CSD_i + Cmatrix[parkings][buildings]*group['number_of_users']
            CSD.append(int(CSD_i))
        return CSD



    # Update the to remove field to remove the users when they need to go away.
    # Suppose that if the time is short, the user stay for theta1 time slot.
    # If the time is medium the user stay for theta2 time slots,
    # and if the time is long the user stay for theta3 time slots.
    def save_users_to_remove(self, flag, users, dict_data, current_time, parking_lot, park, quantity):
        myDict = {
            "quantity": quantity,
            "parking_lots_index": parking_lot,
            "parking_lot": park
        }
        tot_time = int(dict_data['time_slots'])
        if flag == 0:
            if users['dwell_time'] == 'short':
                # Check if the timeslot is still one of intestest
                if current_time+dict_data['theta1'] < tot_time: 
                    self.to_remove_fast[current_time+dict_data['theta1']].append(myDict)
            elif users['dwell_time'] == 'medium':
                if current_time+dict_data['theta2'] < tot_time: 
                    self.to_remove_fast[current_time+dict_data['theta2']].append(myDict)
            elif users['dwell_time'] == 'long':
                if current_time+dict_data['theta3'] < tot_time: 
                    self.to_remove_fast[current_time+dict_data['theta3']].append(myDict)
        else:
            if users['dwell_time'] == 'short':
                if current_time+dict_data['theta1'] < tot_time: 
                    self.to_remove_slow[current_time+dict_data['theta1']].append(myDict)
            elif users['dwell_time'] == 'medium':
                if current_time+dict_data['theta2'] < tot_time: 
                    self.to_remove_slow[current_time+dict_data['theta2']].append(myDict)
            elif users['dwell_time'] == 'long':
                if current_time+dict_data['theta3'] < tot_time: 
                    self.to_remove_slow[current_time+dict_data['theta3']].append(myDict)



    # Delete all the users that should leave in this time slot
    def remove_users(self, users, O, S, R):
        if users != []:
            for element in users:
                park = element['parking_lot']
                park_index = element['parking_lots_index']
                # Update the number of users in the parking lot selected
                O[park_index] = O[park_index] - element["quantity"]
                # Remove the selected park from the active parks if its number of user is 0
                if park in S and O[park_index] == 0:
                    S.remove(park)
                # Remove the selected park from the active and not full parks if its number of user is 0
                if park in R and O[park_index] == 0:
                    R.remove(park)
                # Add the selected park to the active and not full parks if it was previously full 
                # and now it is no more full
                if park not in R and O[park_index] != 0:
                    R.append(park)
        return O, S, R

    

    # Add the users that arrive in this time slot that need fast/slow chargers
    # flag -> 0 => fast chargers
    # flag -> 1 => slow chargers
    def solve_second_stage_subproblem(self, flag, K, P, S, R, O, EVD,\
         Cmatrix, dict_data, current_time, num_of_recharged_users):
        EVD_i = EVD.copy()
        # While the users are not ended or the parking lots are not all checked
        while K != [] and P != []:
            # Calculate CSD for each parking lots with at least one fast/slow charger, where CSD is the number
            # of users that can reach the selected parking lot
            CSD = self.return_CSD(Cmatrix, K)
            # Remove the parking lots where no one can park
            for it, value in enumerate(CSD):
                if value == 0:
                    if flag == 0:
                        if self.parks_with_fast[it] in P:
                            P.remove(self.parks_with_fast[it])
                    else:
                        if self.parks_with_slow[it] in P:
                            P.remove(self.parks_with_slow[it])
            # Take the parking lot in which more users can park
            flag_park = True
            flag_stop = True
            while flag_park and flag_stop:
                i = np.argmax(CSD)
                if flag == 0:
                    park = self.parks_with_fast[i]
                    slots = self.number_of_fast[i]
                else:
                    park = self.parks_with_slow[i]
                    slots = self.number_of_slow[i]
                if park in P:
                    flag_park = False
                else:
                    # So that it cannot be the max anymore
                    CSD[i] = 0
                if all([item == 0 for item in CSD]):
                    flag_stop = False
            if flag_stop:
                # Remove the selected i parking lot from the parkings that are not analyzed yet
                P.remove(park)
                # If missing, add i to the active parking lots
                if park not in S:
                    S.append(park)
                number_of_assigned_users = 0
                # If all the users that needs the recharge (CSD) are more than the available 
                # chargers (total number of fast parking lots – used number of fast parking lots)
                if slots - O[i] == 0:
                    pass
                elif slots - O[i] < CSD[i]:
                    # Assign to the parking lots i (slots - O) users
                    while number_of_assigned_users < slots - O[i]:
                        # Take the users that can charge their car in the smallest number of parks
                        # that can park in the selected parkings
                        # Take the group that can park in less slots
                        min_value = sys.maxsize
                        for it, val in enumerate(EVD_i):
                            group = K[it]
                            building = int(group['building'].split('_')[1])
                            if val > 0 and Cmatrix[i][building] == 1:
                                if val < min_value:
                                    min_value = val
                                    index = it
                        j = index
                        # Select the number of users of the selected group
                        CSD_i = K[j]['number_of_users']
                        # Add these users to the ones served by the selected parking lots
                        if CSD_i > slots - O[i]:
                            K[j]['number_of_users'] = K[j]['number_of_users'] - (slots - O[i])
                            number_of_assigned_users = number_of_assigned_users + (slots - O[i])
                            quantity = slots - O[i]
                            O[i] = slots
                            self.save_users_to_remove(flag, K[j], dict_data, current_time, i, park, quantity)
                            if K[j]['class'] == 2:
                                # Add to the already recharged veichles of class 2 the number of the group and the quantity
                                # of users that has been already recharged
                                self.already_recharged.append([K[j]['num_group'], quantity])
                            # Add the users to the one served
                            if K[j]['dwell_time'] == 'short':
                                num_of_recharged_users['theta1'] = num_of_recharged_users['theta1'] + quantity
                            elif K[j]['dwell_time'] == 'medium':
                                num_of_recharged_users['theta2'] = num_of_recharged_users['theta2'] + quantity
                            else:
                                num_of_recharged_users['theta3'] = num_of_recharged_users['theta3'] + quantity
                        else:
                            O[i] = O[i] + CSD_i
                            quantity = CSD_i
                            self.save_users_to_remove(flag, K[j], dict_data, current_time, i, park, quantity)
                            if K[j]['class'] == 2:
                                # Add to the already recharged veichles of class 2 the number of the group and the quantity
                                # of users that has been already recharged
                                self.already_recharged.append([K[j]['num_group'], quantity])
                            # Add the users to the one served
                            if K[j]['dwell_time'] == 'short':
                                num_of_recharged_users['theta1'] = num_of_recharged_users['theta1'] + quantity
                            elif K[j]['dwell_time'] == 'medium':
                                num_of_recharged_users['theta2'] = num_of_recharged_users['theta2'] + quantity
                            else:
                                num_of_recharged_users['theta3'] = num_of_recharged_users['theta3'] + quantity
                            del K[j]
                            del EVD_i[j]
                            number_of_assigned_users = number_of_assigned_users + CSD_i
                else:
                    # All the CSD users can be assigned to the parking lot i
                    # Update the number of used chargers
                    O[i] = O[i] + CSD[i]
                    # Add i in the active and not full parkings if it is not already present
                    if park not in R:
                        R.append(park)
                    # Add the users to the ones to be removed
                    to_remove = []
                    for group in K:
                        buildings = int(group['building'].split('_')[1])
                        if Cmatrix[i][buildings] == 1:
                            quantity = group['number_of_users']
                            self.save_users_to_remove(flag, group, dict_data, current_time, i, park, quantity)
                            if group['class'] == 2:
                                # Add to the already recharged veichles of class 2 the number of the group and the quantity
                                # of users that has been already recharged
                                self.already_recharged.append([group['num_group'], quantity])
                            # Add the users to the one served
                            if group['dwell_time'] == 'short':
                                num_of_recharged_users['theta1'] = num_of_recharged_users['theta1'] + quantity
                            elif group['dwell_time'] == 'medium':
                                num_of_recharged_users['theta2'] = num_of_recharged_users['theta2'] + quantity
                            else:
                                num_of_recharged_users['theta3'] = num_of_recharged_users['theta3'] + quantity
                            to_remove.append(group)
                    for item in to_remove:
                        it = K.index(item)
                        del EVD_i[it]
                        del K[it]
        return K, S, R, O