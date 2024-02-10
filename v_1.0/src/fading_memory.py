import numpy as np
from statistics import *
import sys
import pathlib
import os
import pandas as pd
from model_fading_mem import *
from res import *
from bloqade_reservoir_reg import *


num_runs=5

#for t in range(16):
print(f"--------tau = 100---------")
C_vec=[]
for n in range(num_runs):
    print(f"--------run number {n}---------")
    data=np.random.rand(3000,1)
    seed = 24
    system_type = ['Classical_RC', 'RC_ESN', 'Bloqade']
    mode = ['generative', 'prediction']
    d = 9.0
    lattice_type = 'chain'
    window = 10
    alpha = 0.5
    reg = 0.001
    t_start = 0.0
    t_end = 3.0
    step = (t_end-t_start)/window
    omega = 2*np.pi
    n_sites = 10
    MIN = np.min(data)
    MAX = np.max(data)
    num_timesteps = np.floor((t_end-t_start)/step).astype(int)
    reservoir_size = (n_sites + (n_sites*(n_sites-1))//2)*num_timesteps
    train_step=1000
    test_step=1000
    tau=100

    quantum_reservoir = BloqadeReservoir(t_start, t_end, step, omega, n_sites, d, lattice_type)
    model = ReservoirArchitecture(seed, quantum_reservoir, reservoir_size, train_step, test_step, window, alpha, reg, system_type[2], MIN, MAX, tau)
    Wout = model.fit(data)
    C,Y_pred = model.pred(data, Wout, mode[1])
    C_vec.append(C)
    np.save(f"Results\Metrics\C_vec_5runs_100",C_vec)
    np.save(f"Results\Metrics\C_vec_mean_100",np.array(C_vec).mean())


