# Placement-of-EV-charging-stations

The aim of this project is that of finding an optimal placement, in terms of location and quantity, of fast and slow charging stations within the parking lots of a urban environment. This is because, EVs suffer from driving range limitations compared to the traditional fossil-fuel-based vehicles. Users may need to charge their vehicle away from home and find a sufficient and convenient availability of public charging stations.

## Requirements

* Python 3.6+
* Gurobi

# Run

In order to run the simulation you have to launch the script _main.py_ in the main directory.

The obtained results of the simulation are stored under the directory _/results_. Here you can find numerical results and several plots. In particular

* _exp_general_table.csv_ shows the purely numerical results of the last simulation;
* _CityGraph1_ and _CityGraph2_ show the graphs that describe the connection between parkings and buildings using two different type of visualization;
* _hist_profit_ shows the occurrences of the different values of profit, i.e. the output of the objective function, considering different scenarios and two different run;  
* _In sample stability - Exact model_ shows out of sample results using the exact mathematical model referring to different run;
* _In sample stability - Simple heuristic_ shows in sample results using the heuristic referring to different run;
* _Out-of-sample stability - Exact model_ shows out of sample results using the exact mathematical model referring to different run;
* _Out-of-sample stability - Simple heuristic_ shows out of sample results using the heuristic referring to different run;

 # Output
 On the terminal output in every type of simulation you can find:
 * _Objective function_ that is the value of the objective function itself;
 * _Fast chargers_ and _Slow chargers_ that are list of dictionaries in which are shown the location ('index') and the amount of chargers('val') of the parking.
 * Completion time that is the time spent to solve the steps. 
