# -*- coding: utf-8 -*-
from networkx.algorithms.centrality.closeness import incremental_closeness_centrality
import numpy as np
import random

class Sampler:
    def __init__(self):
        pass

    def demg_mean(self, instance, n_scenarios, n_groups):
        demg = self.demg_sample_stoch(instance, n_scenarios, n_groups)
        denominator = 0
        numerator = 0
        for item in demg:
            denominator += len(item)
            numerator += sum(item)
        
        return (numerator/denominator)

    def tot_pop_mean(self, instance, n_scenarios, n_groups):
        demg = self.demg_sample_stoch(instance, n_scenarios, n_groups)
        numerator = 0
        denominator = len(demg)
        for item in demg:
            numerator += sum(item)
        return (numerator/denominator)

    def demg_sample_stoch(self, instance, n_scenarios, n_groups):
        n_groups=n_groups
        toReturn = [] # Forse meglio una matrice (n_scenarios*instance.n_groups), dove riga i è lo scenario i e colonna j è il demg per il gruppo j nello scenario i
        # to Return restituisce una lista: un elemento per ogni scenario. Ogni elemento è una list contentente in posizione x il demg per il gruppo x
        for scenario in range(n_scenarios):
            populToBuildings = {}
            for i in range(instance.n_buildings):
                populToBuildings[instance.building_feature[i]["name"]]=np.abs(np.around(random.normalvariate(instance.meanBuildingPopulation*instance.building_feature[i]["index"], instance.varBuildingPopulation)))

            demg = [0 for i in range(0, n_groups)]
            for building in instance.building_feature:
                numGroups = len(instance.groupsPerBuilding[building["name"]])

                totPop = populToBuildings[building["name"]]
                weight_list = np.zeros(numGroups)
                for i in range(numGroups):
                    weight_list[i] = random.random()
                weight_list=weight_list/np.sum(weight_list)
                for i,item in enumerate(instance.groupsPerBuilding[building["name"]]):
                    demg[item] += np.around(weight_list[i] * totPop)
            
            toReturn.append(demg)
            
        return toReturn
