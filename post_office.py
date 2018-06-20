from math import sqrt
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
        
        # Find server with the least amount of process time left
        server = min(servers)
        server_index = servers.index(server)
        
        # Check if server is ready to process customer
        if server > arrival_time:
            waiting_time = server - arrival_time
            # Update service time
            server = server + service_dist[i]
            blocked[i] = 1
        else:
            # Update service time
            server = arrival_time + service_dist[i]
        
        servers[server_index] = server
        waiting_times.append(waiting_time)
        
    return (waiting_times, blocked)

def multiple_queues_multiple_servers_simulation(n_customers, n_servers, assignment_strategy="smallest"):
    mean_time_between_customers = 1
    mean_service_time = 8
    waiting_times = []
    blocked = [0 for _ in range(n_customers)]
    arrival_dist = stats.expon.rvs(size=n_customers, scale=mean_time_between_customers)
    service_dist = stats.expon.rvs(size=n_customers, scale=mean_service_time)
    
    servers = [(0,[]) for _ in range(n_servers)]
    arrival_time = 0
    
    for i in range(n_customers):
        waiting_time = 0
        arrival_time = arrival_time + arrival_dist[i]
        
        # Update curent state of queues before adding a new customer
        for s in servers:
            for c in s[1]:
                if c[1] == 0 and c[0] + c[2] <= arrival_time:
                    s[1].remove((c[0], c[1], c[2]))
                elif c[0] + c[1] <= arrival_time:
                    s[1].remove((c[0], c[1], c[2]))
        
        # Find queue to add customer to
        if assignment_strategy == "smallest":
            # Find the smallest queue
            server = min(servers, key=lambda s: len(s[1]))
        elif assignment_strategy == "random":
            # Find a random queue
            server = servers[np.random.randint(n_servers)]
        server_index = servers.index(server)
        
        # Check if server is ready to process customer
        if server[0] > arrival_time:
            waiting_time = server[0] - arrival_time
            # Add customer to queue. Customer is represented as a triple: (arrival_time, waiting_time, serving time)
            server[1].append((arrival_time, waiting_time, service_dist[i]))
            server = (server[0] + service_dist[i], server[1])
            blocked[i] = 1
        else:
            # Add customer to queue. Customer is represented as a triple: (arrival_time, waiting_time, serving time)
            server[1].append((arrival_time, waiting_time, service_dist[i]))
            server = (arrival_time + service_dist[i], server[1])
        
        servers[server_index] = server
        waiting_times.append(waiting_time)
        
    return (waiting_times, blocked)
    
def calculate_confidence_intervals(mean, standard_deviation, n_simulations):
    z_s = stats.t.ppf(0.95, n_simulations)
    lower = mean - z_s * (standard_deviation/sqrt(n_simulations))
    upper = mean + z_s * (standard_deviation/sqrt(n_simulations))
    return (lower, upper)

def calculate_erlang(n_service_units,mean_service_time,mean_time_between_customers):
    E = mean_time_between_customers*mean_service_time
    m = n_service_units
    Ts = mean_service_time
    #Calculate part above the line
    above = (E**m/math.factorial(m))*(m/(m-E))
    #Below the line
    part3 = []
    for k in range(m-1):
        part3 = np.append(part3, (E**k/math.factorial(k)))
    #Probability of waiting
    Pw = above/(np.sum(part3)+above)
    #Average waiting time
    Tw = Pw*Ts/(m*(1-E/m))
    
    return Pw, Tw


def run_simulation(simulation_to_run):
    waiting_times, _ = simulation_to_run(10000, 10)
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
        waiting_times, n_blocked = simulation_to_run(10000, 10)
        wait_time_means.append(np.mean(waiting_times))
        blocked_means.append(np.mean(n_blocked))

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
    
def run_single_queue_multiple_servers_simulation():
    # Single queue multiple servers    
    run_simulation(single_queue_multiple_servers_simulation)
    
def run_multiple_queues_multiple_servers_simulation():
    # Multiple queues multiple servers
    run_simulation(multiple_queues_multiple_servers_simulation)

def main():
    #run_single_queue_multiple_servers_simulation()   
    run_multiple_queues_multiple_servers_simulation()
    
if __name__ == "__main__":
    main()
