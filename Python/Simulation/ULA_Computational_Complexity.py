# -*- coding: utf-8 -*-
"""
Created on Mon May 20 08:58:55 2024

@author: sedla
"""

import doaLib as dsp
import time

num_signals = 1
freq = [20e3]
theta_deg = [20]

num_elem = 5
delta = 0.5

sample_rate = 1e6
num_samples = 100000

snr_dB = 20

ref_data = dsp.generate_data_ULA(freq, theta_deg, num_elem, delta, sample_rate, num_samples)
data = dsp.add_noise(ref_data, snr_dB)

start = time.time()
R_xx1 = dsp.covariance_matrix(data)
results, theta_scan = dsp.doa_bartlett_ULA(R_xx1, delta, 10000)
stop = time.time()
exec_time = (stop - start) * 1000

print("Bartlett: ", exec_time)
    
start = time.time()
R_xx2 = dsp.covariance_matrix(data)
results, theta_scan = dsp.doa_MVDR_ULA(R_xx2, delta, 10000)
stop = time.time()
exec_time = (stop - start) * 1000

print("MVDR: ", exec_time)

start = time.time()
R_xx3 = dsp.covariance_matrix(data)
results, theta_scan = dsp.doa_linpred_ULA(R_xx3, delta, 0, 10000)
stop = time.time()
exec_time = (stop - start) * 1000

print("Lin. pred.: ", exec_time)

start = time.time()
R_xx4 = dsp.covariance_matrix(data)
results, theta_scan = dsp.doa_MUSIC_ULA(R_xx4, delta, num_signals, 10000)
stop = time.time()
exec_time = (stop - start) * 1000

print("MUSIC: ", exec_time)

start = time.time()
R_xx5 = dsp.covariance_matrix(data)
results, theta_scan = dsp.doa_minnorm_ULA(R_xx5, delta, num_signals, 10000)
stop = time.time()
exec_time = (stop - start) * 1000

print("Min-Norm: ", exec_time)

start = time.time()
R_xx6 = dsp.covariance_matrix(data)
theta_result = dsp.doa_rootMUSIC_ULA(R_xx6, delta, num_signals)
stop = time.time()
exec_time = (stop - start) * 1000

print("Root-MUSIC: ", exec_time)