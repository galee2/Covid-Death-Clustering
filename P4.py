##########################################################################################################
# Author: Gene Lee
# CS 540 (Summer 2021)
########################################################################################################## 

import csv
import numpy as np
import math

############################################ HELPER FUNCTIONS ############################################
def scale_data(data):
    data_min, data_max = min(data), max(data)
    return [round((d - data_min) / data_max, 2) for d in data]

def distance(c1, c2):
    sqr_diff = np.square(np.subtract(c1, c2))
    return math.sqrt(sum(sqr_diff))

def single_link(c1, c2, state_param):
    if c2 == None: print("huh")

    params1 = [state_param[state] for state in c1]
    params2 = [state_param[state] for state in c2]
    
    d_min = np.Infinity

    for p1 in params1: # check all possible pairs in the two clusters for the minimum distance
        for p2 in params2:
            d = distance(p1, p2)
            if d < d_min: d_min = d

    return d_min # return min distance

def complete_link(c1, c2, state_param):
    params1 = [state_param[state] for state in c1]
    params2 = [state_param[state] for state in c2]
    
    d_max = 0

    for p1 in params1: # check all possible pairs in the two clusters for the maximum distance
        for p2 in params2:
            d = distance(p1, p2)
            if d > d_max: d_max = d
        
    return d_max # return max distance

def hierarchial_cluster(k, clusters, state_param, d_func):
    while len(clusters) != k:
        d_min = np.Infinity
        min_pair = [None, None]

        for i in range(len(clusters)):
            for j in range(len(clusters)):
                if i == j: continue # do not compute distance between a cluster and itself
                d = d_func(clusters[i], clusters[j], state_param)
                if d < d_min:
                    d_min = d
                    min_pair = [i, j]
        a, b = min_pair[0], min_pair[1]
        clusters[a] = clusters[a].union(clusters[b])
        clusters.pop(b)
    
    return clusters

def k_means(k, state_param):
    curr_centers = [tuple(np.random.random(6)) for i in range(k)]
    prev_centers = [(1,1,1,1,1,1)]*k

    while prev_centers != curr_centers:
        clusters = [set() for i in range(k)]
        distortion = 0

        for state in state_param: # assign each state to a cluster based on the current centers
            d_min = np.Infinity
            i = None
            for c in range(k):
                d = distance(state_param[state], curr_centers[c])
                if d < d_min:
                    d_min, i = d, c
            distortion = distortion + math.pow(d_min, 2)
            clusters[i].add(state)

        new_centers = [None]*k # calculate new cluster centers
        for c in range(k):
            sum = (0,0,0,0,0,0)
            tot = 0
            for state in clusters[c]:
                sum = tuple(np.add(sum, state_param[state]))
                tot = tot + 1
            if tot == 0: new_centers[c] = curr_centers[c]
            else: new_centers[c] = tuple(np.divide(sum, tot).round(4))

        prev_centers = curr_centers # reassign for iteration
        curr_centers = new_centers

    return clusters, curr_centers, distortion


def q1_2(wi, tx, num):
    name = 'P4_Q' + str(num) + ".txt"
    with open(name, 'w') as filehandle: # write cumulative time series to a text file
        q1_2 = ''
        for num in wi: # store WI cumulative time series to a string
            q1_2 = q1_2 + str(round(num,2)) + ', '
        q1_2 = q1_2[:-2] # remove last ', ' from string
        filehandle.writelines("%s\n" % q1_2) # write WI data to file

        q1_2 = ''
        for num in tx: # store TX cumulative time series to a string
            q1_2 = q1_2 + str(round(num,2)) + ', '
        q1_2 = q1_2[:-2] # remove last ', ' from string
        filehandle.writelines("%s\n" % q1_2) # write TX data to file
def q4(mean, var, stdev, median, per75, maxi, state_names):
    with open('P4_Q4.txt', 'w') as filehandle: # write cumulative time series to a text file
        q4 = ''
        for i in range(len(state_names)):
            q4 = str(round(mean[i],2)) + ", " + str(round(var[i],2)) + ", " + str(round(stdev[i],2)) \
                 + ", " + str(round(median[i],2)) + ", " + str(round(per75[i],2)) \
                 + ", " + str(round(maxi[i],2))
            filehandle.writelines("%s\n" % q4)
def q5_6_7(clusters, state_names, num):
    name = 'P4_Q' + str(num) + ".txt"

    labels = [None]*len(state_names)
    for c in range(len(clusters)):
        for state in clusters[c]:
            i = state_names.index(state)
            labels[i] = c
    
    q_5_6_7 = ""
    for l in labels: # store cluster labels to a string
        q_5_6_7 = q_5_6_7 + str(l) + ', '
    q_5_6_7 = q_5_6_7[:-2] # remove last ', ' from string
    with open(name, 'w') as filehandle: # write clusters to a text file
        filehandle.writelines("%s\n" % q_5_6_7)
def q8(centers):
    with open("P4_Q8.txt", 'w') as filehandle: # write cluster centers to a text file
        for c in centers: # store cluster centers to a string
            q_8 = ""
            for p in c:
                q_8 = q_8 + str(round(p,4)) + ', '
            q_8 = q_8[:-2] # remove last ', ' from string
            filehandle.writelines("%s\n" % q_8)

############################################### DATA SETUP ###############################################
data = []
labels = []

# parse csv file of training data
with open('time_series_covid19_deaths_US.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for row in csv_reader: # iterate over all rows in the csv file
        if row[0] == "UID": continue # ignore first row names
        if row[11] == "0": continue # ignore rows with population 0
        
        data.append(row)
data = np.array(data)

################################################# PART 1 #################################################
state_tot = dict() # create dictionary to keep track of cumulative data (col 12:end) for each state
state_diff = dict() # create dictionary to keep track of time-differenced data for each state
state_pop = dict() # create dictionary to keep track of population (col 11) for each state
state_per = dict() # create dictionary to keep track of time-differenced data divided by population
state_names = [] # create dictionary to keep track of parameters for each state
n = 0 # number of states

for row in data: # add all state names to dictionaries and save data
    state = row[6]
    state_data = np.array(list(map(int, row[12:])))

    if state not in state_tot: # add new state and data
        n = n + 1
        state_tot[state] = state_data
        state_diff[state] = []
        state_pop[state] = int(row[11])
        state_names.append(state)
    else: # add state's data to existing key
        state_tot[state] = np.add(state_tot[state], state_data)
        state_pop[state] = state_pop[state] + int(row[11])

for state in state_tot: # find time differenced data divided by population for each state
    state_data = state_tot[state]
    state_data_diff = []
    for i in range(len(state_data)-1):
        state_data_diff.append(state_data[i+1] - state_data[i])
    state_diff[state] = np.array(state_data_diff)
    state_per[state] = np.divide(state_diff[state], state_pop[state])

# Question 1
q1_2(state_tot['Wisconsin'], state_tot['Texas'], 1)

# Question 2
q1_2(state_diff['Wisconsin'], state_diff['Texas'], 2)

# 6 parameters = mean, variance, standard deviation, median, 75% percentile, maximum
mean, var, stdev, median, per75, maxi = [], [], [], [], [], []
for state in state_names:
    state_data_diff = state_per[state]
    #state_data_diff = state_diff[state]
    mean.append(np.mean(state_data_diff))
    var.append(np.var(state_data_diff))
    stdev.append(np.std(state_data_diff))
    median.append(np.median(state_data_diff))
    per75.append(np.percentile(state_data_diff, 75))
    maxi.append(np.max(state_data_diff))

# rescale parameters to range from 0 to 1
mean_scale = scale_data(mean)
var_scale = scale_data(var)
stdev_scale = scale_data(stdev)
median_scale = scale_data(median)
per75_scale = scale_data(per75)
maxi_scale = scale_data(maxi)

state_param = dict() # store parameters of each state to a dictionary
for i in range(len(state_names)):
    state_param[state_names[i]] = (mean_scale[i], var_scale[i], stdev_scale[i], 
                                   median_scale[i], per75_scale[i], maxi_scale[i])

# Question 4
q4(mean_scale, var_scale, stdev_scale, median_scale, per75_scale, maxi_scale, state_names)

################################################# PART 2 #################################################
k = 6 # number of clusters

# hierarchial clustering
single_clusters = [{s} for s in state_names] # start with each state as its own cluster
complete_clusters = [{s} for s in state_names]

single_clusters = hierarchial_cluster(k, single_clusters, state_param, single_link)
complete_clusters = hierarchial_cluster(k, complete_clusters, state_param, complete_link)

# Question 5
q5_6_7(single_clusters, state_names, 5)

# Question 6
q5_6_7(complete_clusters, state_names, 6)

# k-means clustering
kmeans_clusters, centers, distortion = k_means(k, state_param)

# Question 7
q5_6_7(kmeans_clusters, state_names, 7)

# Question 8
q8(centers)

# Question 9
print(f"Total distortion = {round(distortion, 4)}")