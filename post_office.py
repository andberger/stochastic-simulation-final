from math import log, floor, factorial, sqrt
import statistics as st
import random as random
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from collections import Counter


def single_queue_multiple_servers_simulation(n_customers, n_servers):
    mean_time_between_customers = 1
    mean_service_time = 8
    waiting_times = []
    arrival_dist = stats.expon.rvs(size=n_customers, scale=mean_time_between_customers)
    service_dist = stats.expon.rvs(size=n_customers, scale=mean_service_time)
    
    servers = [0 for _ in range(n_servers)]
    arrival_time = 0
    
    for i in range(n_customers):
        waiting_time = 0
        arrival_time = arrival_time + arrival_dist[i]
        
        server = min(servers)
        server_index = servers.index(server)
        
        if server > arrival_time:
            waiting_time = server - arrival_time
            server = server + service_dist[i]
        else:
            server = arrival_time + service_dist[i]
        
        servers[server_index] = server
        waiting_times.append(waiting_time)
        
    return waiting_times
    
def calculate_confidence_intervals(mean, standard_deviation, n_simulations):
    z_s = stats.t.ppf(0.95, n_simulations)
    lower = mean - z_s * (standard_deviation/sqrt(n_simulations))
    upper = mean + z_s * (standard_deviation/sqrt(n_simulations))
    return (lower, upper)
        
def erlang_B_formula(n, arrival_intensity, mean_service_time):
    A = arrival_intensity * mean_service_time
    B = ((A**n) / (factorial(n))) / (sum([((A**i)/factorial(i)) for i in range(n)]))
    return B  
    

def main():    
    # Single queue multiple servers    
    waiting_times = single_queue_multiple_servers_simulation(10000, 10)
    waiting_times_stip_zeros = [i for i in waiting_times if i > 0]
    
    plt.hist(waiting_times_stip_zeros,  alpha=0.5)
    plt.title("Distribution of waiting times")
    plt.legend(loc='upper right')
    plt.xlabel("Waiting times")
    plt.ylabel("Frequency")
    plt.show()
    
if __name__ == "__main__":
    main()
