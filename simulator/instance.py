# -*- coding: utf-8 -*-
import logging
from networkx.generators.small import bull_graph
import numpy as np
import os
import networkx as nx
import json
import random
import matplotlib.pyplot as plt
import datetime


graph_path="graph.json"
graph_path1="graph1.json"
addInfo_path="addInfo.json"
addInfo_path1="addInfo1.json"
class Instance():

    def __init__(self, sim_setting):

        # /* -------------------------------------------------------------------------- */
        # /*                                    SETTINGS                                */
        # /* -------------------------------------------------------------------------- */
        random.seed(12)

        logging.info("starting simulation...")

        # Get the data from the simulation settings file
        # Fraction of drivers belonging to classes 1, 2, 3, respectively.
        self.beta1 = sim_setting['beta1']
        self.beta2 = sim_setting['beta2']
        self.beta3 = sim_setting['beta3']
        #Indicating the duration of short, medium, and long dwell times,
        # respectively (expressed in the number of time slots)
        self.theta1 = sim_setting['theta1']
        self.theta2 = sim_setting['theta2']
        self.theta3 = sim_setting['theta3']

        #The income achieved by charging an EV by a fast  charger (expressed in monetary unit per time slot)
        self.Eprime = sim_setting['E_prime']
        # ... for slow charger
        self.E_2nd = sim_setting['E_2nd']

        #Fixed cost of equipping a parking lot with fast chargers
        self.F_prime = sim_setting['F_prime']
        # ... for slow charger
        self.F_2nd = sim_setting['F_2nd']

        #The cost of purchasing a fast charger
        self.C_prime = sim_setting['C_prime']
        # ... for slow charger
        self.C_2nd = sim_setting['C_2nd']

        #Maximum number of parking lots that can be equipped with chargers
        self.M = sim_setting['M']
        #Total number of available chargers
        self.M_prime = sim_setting['M_prime']

        #The maximum number of chargers that can be installed in the parking lot p
        self.M_2nd = sim_setting['M_2nd']
        # Amount of time slots
        self.time_slots = sim_setting['time_slots']
        # Amount of buildings
        self.n_buildings = sim_setting['n_buildings']
        # Amount of parking lots
        self.n_parkingLots = sim_setting['n_parkingLots']
        # Width of city map (expressed in 10 meters unit)
        self.max_distance_city=sim_setting['max_distance_city']
        # Import the radius of the area in which you can't have more than 1 building
        RADIUS = sim_setting['RADIUS']


        # /* -------------------------------------------------------------------------- */
        # /*                        MAP & GRAPH                                         */
        # /* -------------------------------------------------------------------------- */
        #Initialization of list of timeslot
        self.timeSlotList = []
        for i in range(self.time_slots):
            self.timeSlotList.append(i)

        # Creation of building item
        self.buildings = ["b_"+str(i) for i in range(self.n_buildings)]
        # Values of charging time (expressed in the number of time slots)
        self.chargingTimes = sim_setting["charging_times"]
        # Characterization of classes
        self.classes = sim_setting["classes"]
        # Number of group considered in simulation
        self.n_groups=sim_setting["n_groups"]
        # Characterization of type, i.e. charging mode,dwell time associated to a specific class
        self.types = sim_setting["types"]

        # Create random popularity indexes, in order to characterize the building visited more
        self.populiarities = []
        for i in range(self.n_buildings):
            self.populiarities.append(random.uniform(0, 1))

        # Create a list of objects including the features of the buildings (name, popularity)
        self.building_feature=[dict(name=item, index=self.populiarities[i]) for i,item in enumerate(self.buildings)]
        # ... sort it using popularity index
        self.building_feature=sorted(self.building_feature, key=lambda i: i['index'], reverse=True)

        # Creation of parking lots
        self.parkingLots = ["p_"+str(i) for i in range(self.n_parkingLots)]
        # Import maximum distance an user can afford to reach parking
        self.maxWalkingDistance = sim_setting["maxWalkingDistance"]


        # Initialize the graph with building and parking as nodes
        G = nx.Graph()
        G.add_nodes_from(self.buildings + self.parkingLots)

        # /* -------------------------------------------------------------------------- */
        # /*                                    BUILDINGS                               */
        # /* -------------------------------------------------------------------------- */
        # Generate the matrix that represents the physical 2D map of the urban environment
        phy_position_matrix = np.empty((self.max_distance_city, self.max_distance_city), dtype=object)
        first_step = 1
        for building in self.buildings:
            # Get a random position in the map
            i = random.randint(0, self.max_distance_city - 1)
            j = random.randint(0, self.max_distance_city - 1)

            if first_step == 0:
                cond = False
                # Iterate until you generate a valid "physical" position for the building
                while not cond:
                    if phy_position_matrix[i, j] == None: # Check if the position in the matrix is already occupied by another building
                        occupied = np.argwhere(phy_position_matrix) # If not, check if the other buildings are at least RADIUS distance away from the found position
                        cond = True
                        # Check if the slot that you are referring is already occupied
                        for item in occupied:
                            distance = np.floor(np.sqrt((i - item[0]) ** 2 + (j - item[1]) ** 2))
                            if distance < RADIUS:
                                i = random.randint(0, self.max_distance_city - 1)
                                j = random.randint(0, self.max_distance_city - 1)
                                cond = False
                                break


                    else: #If the cell is already occupied, take another random position
                        i = random.randint(0, self.max_distance_city - 1)
                        j = random.randint(0, self.max_distance_city - 1)
            else:
                first_step = 0
            # If you pass all the control step you can assign the building to the slot
            phy_position_matrix[i, j] = building


        # Generate a data format to easily plot the map of the urban environment
        occupied = np.argwhere(phy_position_matrix)
        building_coordinates_final = occupied

        # Extract the coordinate of the buildings
        coord_building = np.zeros((self.n_buildings, 3))
        i = 0
        for building in self.building_feature:
            index = np.argwhere(phy_position_matrix == building['name'])
            building['coor'] = index
            coord_building[i, 0] = index[0][0]
            coord_building[i, 1] = index[0][1]
            coord_building[i, 2] = building['index'] * 350
            i = i + 1



        # /* -------------------------------------------------------------------------- */
        # /*                                    PARKINGS                                */
        # /* -------------------------------------------------------------------------- */

        # Repeat the same operation to place the parking lots
        for park in self.parkingLots:
            #Get a random position
            i = random.randint(0, self.max_distance_city - 1)
            j = random.randint(0, self.max_distance_city - 1)
            if first_step == 0:
                cond = False
                while not cond:
                    if phy_position_matrix[i, j] == None: #Check if the random position is occupied
                        occupied = np.argwhere(phy_position_matrix) # If not, check if the other parkings are at least RADIUS distance away from the found position
                        cond = True
                        for item in occupied:
                            if item not in building_coordinates_final: # If the item is a parking lot (we don't care if a parking is very close to a building; actually, it is more realistic)
                                distance = np.floor(np.sqrt((i - item[0]) ** 2 + (j - item[1]) ** 2))
                                if distance < RADIUS:
                                    i = random.randint(0, self.max_distance_city - 1)
                                    j = random.randint(0, self.max_distance_city - 1)
                                    cond = False
                                    break
                    else: #If the cell is already occupied, take another random position
                        i = random.randint(0, self.max_distance_city - 1)
                        j = random.randint(0, self.max_distance_city - 1)
            else:
                first_step = 0

            phy_position_matrix[i, j] = park

        occupied = np.argwhere(phy_position_matrix)
        total_occupied = occupied
        total_occupied = [tuple(item) for item in total_occupied]
        building_coordinates_final = [tuple(item) for item in building_coordinates_final]
        coord_parking = np.zeros((self.n_parkingLots, 2))
        i = 0
        # Data format for the map generation
        parking_coordinate_final = []
        for item in total_occupied:
            if item not in building_coordinates_final:
                parking_coordinate_final.append(item)
                coord_parking[i, 0] = item[0]
                coord_parking[i, 1] = item[1]
                i = i + 1

        # Geneate the map of the urban environment, differentiating between parkings and buildings and representing the popularity index of each building
        plt.rc("axes", axisbelow = True)
        plt.scatter(coord_building[:, 0], coord_building[:, 1], s=coord_building[:, 2], label='buildings')
        for building in coord_building:
            plt.annotate(phy_position_matrix[int(building[0]), int(building[1])], (building[0], building[1]))
        plt.scatter(coord_parking[:, 0], coord_parking[:, 1], label='parkings')
        for parking in coord_parking:
            plt.annotate(phy_position_matrix[int(parking[0]), int(parking[1])], (parking[0], parking[1]))
        plt.legend()
        plt.grid()
        plt.savefig(f"./results/CityMap.png")
        plt.show()

        # /* -------------------------------------------------------------------------- */
        # /*                                    GROUPS                                  */
        # /* -------------------------------------------------------------------------- */

        # Create random groups
        self.groups = []
        # There is a maximum number of unique groups based on the input settings
        if self.n_groups > (self.n_buildings * len(self.timeSlotList) * 3):
            print("WARNING BAD SETTINGS---> More group than possible combinations")
            exit(10)

        dwellT = list(self.chargingTimes.keys())
        for i in range(self.n_groups):
            # Pick two random buildings
            b = random.sample(range(0, self.n_buildings - 1), 2)
            start_p = building_coordinates_final[b[0]]
            end_p = building_coordinates_final[b[1]]
            # Calculate the travel time based on the actual distance between the two buildings and the scale of the map (factor 0.002 was set using trial and error)
            travel_time = np.ceil((np.floor(np.sqrt((start_p[0] - end_p[0]) ** 2 + (start_p[1] - end_p[1]) ** 2)))*0.002)
            d = []
            d.append(random.randint(0, len(dwellT)-1))
            d.append(random.randint(0, len(dwellT)-1))
            # Calculate the maximum time slot of arrival at the first building. The full charge in the first building, the travel time and the full charge in the seocond building have to happen within the considered time slots
            maxslot = np.floor(len(self.timeSlotList) - 1 - self.chargingTimes[dwellT[d[0]]] - self.chargingTimes[dwellT[d[1]]] - travel_time)
            # Generate the group instance
            group = {"building": [self.buildings[b[0]], self.buildings[b[1]]],
                     "arrival_time_slot": self.timeSlotList[random.randint(0, maxslot)],
                     "dwell_time": [dwellT[d[0]], dwellT[d[1]]],
                     "travel_time":travel_time}
            # Check if the generated group does not already exist
            while group in self.groups:
                # If it exists, generate a new group until it is a unique one
                b = random.sample(range(0, self.n_buildings - 1), 2)
                start_p = building_coordinates_final[b[0]]
                end_p = building_coordinates_final[b[1]]
                travel_time = np.ceil((np.floor(np.sqrt((start_p[0] - end_p[0]) ** 2 + (start_p[1] - end_p[1]) ** 2))) * 0.002)
                d=[]
                d.append(random.randint(0, len(dwellT)-1))
                d.append(random.randint(0, len(dwellT)-1))
                maxslot = np.floor(len(self.timeSlotList) - 1 - self.chargingTimes[dwellT[d[0]]] - self.chargingTimes[dwellT[d[1]]] - travel_time)
                group = {"building": [self.buildings[b[0]], self.buildings[b[1]]],
                         "arrival_time_slot": self.timeSlotList[random.randint(0, maxslot)],
                         "dwell_time": [dwellT[d[0]], dwellT[d[1]]],
                         "travel_time": travel_time}
            self.groups.append(group)


        # Assign a unique ID to each group
        for i in range(self.n_groups):
            self.groups[i]["groupID"] = i

        # /* -------------------------------------------------------------------------- */
        # /*                                  DISTANCES                                 */
        # /* -------------------------------------------------------------------------- */

        # Calculate the distance between each parking and each building
        distances = []

        for slot in building_coordinates_final:
            tmp = []
            for item in parking_coordinate_final:
                distance = np.floor(np.sqrt((slot[0] - item[0]) ** 2 + (slot[1] - item[1]) ** 2))
                tmp.append(dict(building=slot, park=item, dist=distance))
            tmp = sorted(tmp, key=lambda i: i['dist'], reverse=False) # For each building, store the distance with the parking lots in ascending order
            distances.append(tmp)

        # Generate a matrix that will contain the distance between each building and parking. Rows = buildings, columns = parkings. cell [i,j] contains the distance between building i and parking j
        distance_matrix = 0 * np.random.rand(self.n_buildings, self.n_parkingLots) # Why random??? Maybe empty is better
        # Store for each building which parking are accessible by foot (distance lower than maxWalkingDistance)
        self.parkingsPerBuilding = [dict(name=x, parks=[]) for x in self.buildings]
        # Store for each parking which building are accessible by foot (distance lower than maxWalkingDistance)
        self.buildingPerParking = [dict(name=x, build=[], pop=0) for x in self.parkingLots]

        # Populate the distance matrix
        for item in distances:
            building_coor = item[0]['building']
            name_building = phy_position_matrix[building_coor[0], building_coor[1]]
            i_index = int(name_building.split('_')[1])
            for subitem in item:
                park_coor = subitem['park']
                name_park = phy_position_matrix[park_coor[0], park_coor[1]]
                j_index = int(name_park.split('_')[1])
                distance_matrix[i_index, j_index] = subitem['dist']

                # Save for each building which groups do access it. It will be used later to divide the population among buildings, following the popularity index
        self.groupsPerBuilding = {}

        for building in self.buildings:
            self.groupsPerBuilding[building] = []
            filtered1 = list(filter(lambda group: group['building'][0] == building, self.groups))
            for group in filtered1:
                self.groupsPerBuilding[building].append(group['groupID'])
            filtered2 = list(filter(lambda group: group['building'][1] == building, self.groups))
            for group in filtered2:
                self.groupsPerBuilding[building].append(group['groupID'])

            self.groupsPerBuilding[building]=list(set(self.groupsPerBuilding[building]))


        # Generate a graph containing all parkings and buildings as nodes. An edge between a parking and a building exists if the distance among them is smaller than maxWalkingDistances
        for ni, i in enumerate(self.buildings):
            for nj, j in enumerate(self.parkingLots):
                if distance_matrix[ni, nj] < self.maxWalkingDistance:
                    self.parkingsPerBuilding[ni]['parks'].append(nj)
                    self.buildingPerParking[nj]['build'].append(ni)
                    G.add_edge(i, j, weight=distance_matrix[ni, nj])


        # Find which building you can reach from a certain parking
        for nj, j in enumerate(self.buildingPerParking):
            for ni in self.buildingPerParking[nj]['build']:
                name = 'b_' + str(ni)
                for item in self.building_feature:
                    if item['name']==name:
                        self.buildingPerParking[nj]['pop'] = item['index'] + self.buildingPerParking[nj]['pop']
        self.buildingPerParking = sorted(self.buildingPerParking, key=lambda i: i['pop'], reverse=True)

        # Select what are the best parking for every group for 1st and 2nd building using the building
        # in which group is associated
        self.preferiteParkingsPerGroup=[[[] for i in range(0,self.n_groups)] for j in range(0,2)]
        for building in self.parkingsPerBuilding:
            filtered1 = list(filter(lambda group: group['building'][0] == building['name'], self.groups))
            for group in filtered1:
                self.preferiteParkingsPerGroup[0][group['groupID']]=building['parks']
            filtered2 = list(filter(lambda group: group['building'][1] == building['name'], self.groups))
            for group in filtered2:
                self.preferiteParkingsPerGroup[1][group['groupID']]=building['parks']






        # Generate the G_prime matrix. G_prime is a 3D matrix where element (i,j,p) is a list containing all those groups that,
        # when in their i-th building (i = 1,2), have a dwell time of theta-j, and the parking lot p is amongst their preferred parking
        # lots when they want to visit that building
        self.G_prime = [[[[] for j in range(0, self.n_parkingLots)] for i in range(0, 3)] for z in range(0,2)]
        for building in self.parkingsPerBuilding:
            filtered1 = list(filter(lambda group: group['building'][0] == building['name'], self.groups))
            for item in filtered1:
                dwell_time = item['dwell_time'][0]
                tmp = item['groupID']
                for p in building['parks']:
                    if dwell_time == 'short':
                        self.G_prime[0][0][p].append(tmp)
                    elif dwell_time == 'medium':
                        self.G_prime[0][1][p].append(tmp)
                    elif dwell_time == 'long':
                        self.G_prime[0][2][p].append(tmp)
            filtered2 = list(filter(lambda group: group['building'][1] == building['name'], self.groups))
            for item in filtered2:
                dwell_time = item['dwell_time'][1]
                tmp = item['groupID']
                for p in building['parks']:
                    if dwell_time == 'short':
                        self.G_prime[1][0][p].append(tmp)
                    elif dwell_time == 'medium':
                        self.G_prime[1][1][p].append(tmp)
                    elif dwell_time == 'long':
                        self.G_prime[1][2][p].append(tmp)




        # Generate G_2ND, its a 3D matrix where cell (t,p,r) is a list containing all the groups that may use the charging mode
        # defined by type r (check the types in the sim_settings) and therefore may occupy parking lot p in time slot t
        self.G_2ND = [[[[] for i in range(len(self.types))] for p in range(self.n_parkingLots)] for t in
                      range(self.time_slots)]
        for building in self.parkingsPerBuilding:
            filtered1 = list(filter(lambda group: group["building"][0] == building["name"], self.groups))
            filtered2 = list(filter(lambda group: group["building"][1] == building["name"], self.groups))
            for item in filtered1:
                dwell_time = item["dwell_time"][0]
                dwell_time_duration = self.chargingTimes[dwell_time]
                arrival_time_slot = item["arrival_time_slot"]
                groupID = item["groupID"]
                for time in self.timeSlotList:
                    if (arrival_time_slot >= time - dwell_time_duration + 1) and arrival_time_slot <= time:
                        for p in building["parks"]:
                            if dwell_time == "short":
                                self.G_2ND[time][p][0].append(groupID)
                                self.G_2ND[time][p][2].append(groupID)
                            elif dwell_time == "medium":
                                self.G_2ND[time][p][0].append(groupID)
                                self.G_2ND[time][p][3].append(groupID)
                            elif dwell_time == "long":
                                self.G_2ND[time][p][1].append(groupID)
                                self.G_2ND[time][p][3].append(groupID)
            for item in filtered2:
                dwell_time1 = item["dwell_time"][0]
                dwell_time2 = item["dwell_time"][1]
                dwell_time_duration_1 = self.chargingTimes[dwell_time1]
                dwell_time_duration_2 = self.chargingTimes[dwell_time2]
                arrival_time_slot = item["arrival_time_slot"]
                groupID = item["groupID"]
                travel_time = item["travel_time"]
                for time in self.timeSlotList:
                    cond1 = (arrival_time_slot >= time - dwell_time_duration_1 - dwell_time_duration_2 - travel_time + 1)
                    cond2 = (arrival_time_slot <= time -dwell_time_duration_1 - travel_time)
                    cond= cond1 and cond2
                    if cond:
                        for p in building["parks"]:
                            if dwell_time2 == "short":
                                self.G_2ND[time][p][4].append(groupID)
                                self.G_2ND[time][p][6].append(groupID)
                            elif dwell_time2 == "medium":
                                self.G_2ND[time][p][4].append(groupID)
                                self.G_2ND[time][p][7].append(groupID)
                            elif dwell_time2 == "long":
                                self.G_2ND[time][p][5].append(groupID)
                                self.G_2ND[time][p][7].append(groupID)




        # Draw the graph
        self.graph = G
        color_map = []
        for node in G:
            if node[0] == 'b':
                color_map.append('blue')
            else:
                color_map.append('green')
        nx.draw_kamada_kawai(G, node_color=color_map, with_labels=True, node_size=1000)
        plt.savefig(f"./results/CityGraph1.png")
        plt.show()
        nx.draw(G, node_color=color_map, with_labels=True, node_size=1000)
        plt.savefig(f"./results/CityGraph2.png")
        plt.show()
        x = []
        for u, v in G.edges():
            x.append((u, v, G.edges[u, v]['weight']))

        self.meanBuildingPopulation = sim_setting["meanBuildingPopulation"]
        self.varBuildingPopulation = sim_setting["varBuildingPopulation"]


    # /* -------------------------------------------------------------------------- */
    # /*                                  IMPORT DATA                               */
    # /* -------------------------------------------------------------------------- */
    # Method used to retrieve the various voices of the instance
    def get_data(self):
        logging.info("Getting data from instance...")
        return {
            "beta1":self.beta1,
            "beta2":self.beta2,
            "beta3":self.beta3,
            "theta1":self.theta1,
            "theta2":self.theta2,
            "theta3":self.theta3,
            "Eprime":self.Eprime,
            "E_2nd":self.E_2nd,
            "F_prime":self.F_prime,
            "F_2nd":self.F_2nd,
            "C_prime":self.C_prime,
            "C_2nd":self.C_2nd,
            "M":self.M,
            "M_prime":self.M_prime,
            "M_2nd":self.M_2nd,
            "time_slots":self.time_slots,
            "n_buildings":self.n_buildings,
            "n_parkingLots":self.n_parkingLots,
            "buildings":self.buildings,
            "chargingTimes":self.chargingTimes,
            "classes":self.classes,
            "n_groups":self.n_groups,
            "types":self.types,
            "groups":self.groups,
            "parkingsPerBuilding":self.parkingsPerBuilding,
            "buildingPerParking": self.buildingPerParking,
            "groupsPerBuilding":self.groupsPerBuilding,
            "popularities":self.populiarities,
            "building_feature":self.building_feature,
            "parkingLots":self.parkingLots,
            "maxWalingDistance":self.maxWalkingDistance,
            "meanBuildingPopulation":self.meanBuildingPopulation,
            "varBuildingPopulation":self.varBuildingPopulation,
            "G_prime":self.G_prime,
            "G_2ND": self.G_2ND,
            "timeSlotList":self.timeSlotList,
            "preferiteParkingsPerGroup": self.preferiteParkingsPerGroup
        }
