#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 13:10:06 2018

@author: thorsteinngj
"""
import math

#Calculating the Erlang c values
n_service_units = 10
mean_service_time = 8
mean_time_between_customers = 1

E = mean_time_between_customers*mean_service_time
m = n_service_units
Ts = mean_service_time

#Calculate part above the line
above = (E**m/math.factorial(m))*m/(m-E)


#Below the line
part3 = []
for k in range(m-1):
    part3 = np.append(part3, (E**k/math.factorial(k)))
#Probability of waiting
Pw = above/(np.sum(part3)+below)
print('-----Probability of waiting-----')
print('Theoretical mean waiting chances are {0}'.format(Pw))

#Average waiting time
Tw = Pw*Ts/(m*(1-E/m))
print('-----Average waiting time-----')
print('Theoretical average waiting time is {0}'.format(Tw))