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

above = (E**m)/(math.factorial(m))*(m/(m-E))

for i in range(m-1):
    below1 = E**i/math.factorial(i)

below = below1 + E**m/math.factorial(m)*(m/(m-E))

Pw = above/below