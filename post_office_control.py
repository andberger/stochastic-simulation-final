from math import log, floor, factorial, sqrt
import statistics as st
import random as random
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from collections import Counter
import math

def single_queue_multiple_servers_simulation(n_customers, n_servers):
    mean_time_between_customers = 1
    mean_service_time = 8
    waiting_times = []
    blocked = [0 for _ in range(n_customers)]
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
            blocked[i] = 1
        else:
            server = arrival_time + service_dist[i]
        
        servers[server_index] = server
        waiting_times.append(waiting_time)
        
    return (waiting_times, blocked)
    
def calculate_confidence_intervals(mean, standard_deviation, n_simulations):
    z_s = stats.t.ppf(0.95, n_simulations)
    lower = mean - z_s * (standard_deviation/sqrt(n_simulations))
    upper = mean + z_s * (standard_deviation/sqrt(n_simulations))
    return (lower, upper)

def calculate_erlang(m,Ts,A):
    #E = mean_time_between_customers*mean_service_time
    #m = n_service_units, #Ts = mean_service_time
    #Calculate part above the line
    E = A*Ts
    above = (E**m/math.factorial(m))*(m/(m-E))
    #Below the line
    part3 = []
    for k in range(m):
        part3 = np.append(part3, (E**k/math.factorial(k)))
    #Probability of waiting
    Pw = above/(np.sum(part3)+above)
    #Average waiting time
    Tw = Pw*Ts/(m*(1-E/m))
    #Aftur orðið ekki 100% rett
    return Pw, Tw

def control_variate(X,Y,n):
    #X = wait_mean, #Y = block_mean
    meanY = np.mean(Y)
    VarY = np.sum(Y**2)/n - (np.sum(Y)/n)**2
    CovY = np.sum(X*Y.T)/n-(np.sum(X)/n)*np.sum(Y.T)/n
    c = -CovY/VarY
    Z = X + c*(Y-meanY)
    Z_bar = np.sum(Z)/n
    VarZ = np.sum(Z**2)/n - (np.sum(Z)/n)**2
    return Z, Z_bar, VarZ
    
    

def main():    
    # Single queue multiple servers
    waiting_times, _ = single_queue_multiple_servers_simulation(10000, 10)
    waiting_times_strip_zeros = [i for i in waiting_times if i > 0]
    
    plt.hist(waiting_times_strip_zeros, alpha=0.5, ec="black")
    plt.title("Distribution of waiting times")
    plt.legend(loc='upper right')
    plt.xlabel("Waiting times")
    plt.ylabel("Frequency")
    plt.show()
    
    # Run the simulations n times to gain statistical insights
    wait_time_means = []
    blocked_means = []
    n = 50
    for i in range(n):
        waiting_times, n_blocked = single_queue_multiple_servers_simulation(10000, 10)
        wait_time_means.append(np.mean(waiting_times))
        blocked_means.append(np.mean(n_blocked))
        #Control Variates
        
    wait_time_lower, wait_time_upper = calculate_confidence_intervals(np.mean(wait_time_means), np.std(wait_time_means), n)
    blocked_lower, blocked_upper = calculate_confidence_intervals(np.mean(blocked_means), np.std(blocked_means), n)
    
    Pw, Tw = calculate_erlang(10,8,1)
    
    print("\n")
    print("Mean waiting time: {}".format(np.mean(wait_time_means)))
    print("Theoretical mean: {}".format(Tw))
    print("Lower limit: {}".format(wait_time_lower))
    print("Upper limit: {}".format(wait_time_upper))

    print("\n")
    print("Probability of waiting: {}".format(np.mean(blocked_means)))
    print("Theoretical probability: {}".format(Pw))
    print("Lower limit: {}".format(blocked_lower))
    print("Upper limit: {}".format(blocked_upper))
    
    #Control Variate Part
    #Kannski a n að vera annað?
    Z, Z_bar, VarZ = control_variate(np.array(wait_time_means),np.array(blocked_means),50)
    WaitVar = np.sum(np.array(wait_time_means)**2)/50 - (np.sum(np.array(wait_time_means))/50)**2
    
    BlockVar = np.sum(np.array(blocked_means)**2)/50 - (np.sum(np.array(blocked_means))/50)**2
    CVlower, CVupper = calculate_confidence_intervals(Z_bar,np.std(list(Z)),50)
    
    print("\n")
    print("-:-:-:Control Variates:-:-:-")
    print("Mean waiting time: {}".format(Z_bar))
    print("Lower limit: {}".format(CVlower))
    print("Upper limit: {}".format(CVupper))
    print("------Variance------")
    print("Variance: {}".format(VarZ))
    print("Variance without control variate {}".format(WaitVar))
    
    
if __name__ == "__main__":
    main()
