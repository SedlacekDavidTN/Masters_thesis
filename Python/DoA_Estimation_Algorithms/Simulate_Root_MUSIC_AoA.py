# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 12:30:27 2024

@author: sedla
"""

import numpy as np
import matplotlib.pyplot as plt

## Set simulation parameters
sample_rate = 1e6
n = 10000 # Samples
freq = 200e6 # Frequency

delta = 0.5 # Array element spacing
n_elem = 5 # Number of array elements
n_signals = 1 # Number of estimated signals
theta_deg = 22 # Angle of incidence in degrees

## Create the signal
t = np.arange(n) / sample_rate # Time vector
signal = np.exp(-2j * np.pi * freq * t) # Signal vector
signal = signal.reshape(1,-1)

## Simulate the array
theta = theta_deg * np.pi / 180
a = np.exp(-2j * np.pi * delta * np.arange(n_elem) * np.sin(theta)) # Calculate the array factor 
a = a.reshape(-1,1)
noise = np.random.randn(n_elem, n) + 1j*np.random.randn(n_elem, n) # Generate noise

## Get signals at array elements
r = a @ signal + 0.1*noise

## Estimate the spatial covariance matrix
R_xx = r @ r.conj().T / n

M = R_xx.shape[0]

## Obtain eigenvectors and their eigenvalues
w,v = np.linalg.eig(R_xx)

# Sort eigenvectors to obtain the noise subsopace
eig_val_order = np.argsort(np.abs(w))
v_sorted = v[:, eig_val_order]

Vn = np.zeros((n_elem,n_elem-n_signals), dtype=np.complex64)
for i in range(n_elem-n_signals):
    Vn[:,i] = v_sorted[:,i]
    
ct = Vn @ Vn.conj().T

## Define the polynomial using the noise subspace
p_coeff = np.zeros(2 * M - 1, dtype=np.complex64)

for i in range(-M + 1, M):
    p_coeff[i + (M - 1)] = np.trace(ct, i)
    
all_roots = np.roots(p_coeff)

candidate_roots_abs = np.abs(all_roots)
sorted_idx = candidate_roots_abs.argsort()[(M - 1 - n_signals) : (M - 1)]

valid_roots = all_roots[sorted_idx]
args = np.angle(valid_roots)

doas = np.arcsin(args / (delta * 2.0 * np.pi))

# doas = np.arcsin(args / (np.float32(3*10**8/freq) * 2.0 * np.pi))
# doas = to_zero_to_pi(doas)

doas_deg = np.rad2deg(doas)