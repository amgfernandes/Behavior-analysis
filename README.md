Behavior analysis of zebrafish decision making

# This repository shows the approaches used to try to understand how zebrafish respond to competing stimuli.

I tried Clustering (see folder Clustering) but that approach did not lead to meaningful conclusions.

Based on these results I decided to use a modeling approach that is based on two famous models for decision-making: Averaging and Winner-take-all. 
See folder WTA_vs_Averaging_modeling

This approach was successful and allowed making predictions for the behavioral outcome when animals are confronted with two competing stimuli (in this case two aversive stimuli, looming)


Be carefull when you decide to turn right or left! :)

![Fish escaping](Clustering/fishDecides_v5_small.gif)


Gif credit to Julia Kuhl (https://twitter.com/mulesome)

The modeling approach under the folder WTA_vs_Averaging_modeling was performed with the great help of Joe Donovan (https://github.com/joe311). 

Some of the helper functions for preprocessing the data (see Preprocessing behavioral data folder) were written with the help of Johannes Larsch (https://github.com/jlarsch).

Related to https://www.biorxiv.org/content/10.1101/598383v1

