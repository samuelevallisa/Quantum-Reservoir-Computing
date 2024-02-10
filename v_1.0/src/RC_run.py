import numpy as np
from bloqade_reservoir_reg import *
from statistics import *
import sys
import pathlib
import os
import pandas as pd
from model_reg import *
from res import *
sys.path.append( str(pathlib.Path(__file__).parent.resolve().joinpath("..")) )

#current_dir = os.path.dirname(__file__)
data = np.load('reservoir-computing\\v_1.0\\Data\\mgts_len=5000000tau=18n=10.0bet=0.25gam=0.1h=0.9T=1.npy')
data = np.expand_dims(data,axis=1)

# reservoir_size = 300 for prediction, 1000 for generation
# TODO add params as input

seed = 42
system_type = ['Classical_RC', 'RC_ESN', 'Bloqade']
mode = ['generative', 'prediction']
train_step = 1000
test_step = 300
d = 10.0
lattice_type = 'chain'
window = 10
alpha = 0.2
reg = 5.5
t_start = 0.0
t_end = 3.0
step = (t_end-t_start)/window
omega = 2*np.pi
n_sites = 10
wave = data[:train_step + window]
MIN = np.min(wave)
MAX = np.max(wave)
num_timesteps = np.floor((t_end-t_start)/step).astype(int)
reservoir_size = (n_sites + (n_sites*(n_sites-1))//2)*num_timesteps
n_run = 1

quantum_reservoir = BloqadeReservoir(t_start, t_end, step, omega, n_sites, d, lattice_type)
#r2_mean, mape_mean, mse_mean, Wout_tot, pred_mean = multiple_run(n_run, data, seed, quantum_reservoir, reservoir_size, train_step, test_step, window, alpha, reg, system_type[2], mode[0], MIN, MAX, plot=True)


# num_fold = 10
# slide = 300

# Wout_tot, r2_tot, mape_tot, mse_tot, nmse_tot, pred_tot = [], [], [], [], [], []    

# print('CV start: ')
# for n_fold in range(1,num_fold):   

#     print('n fold: ', n_fold)

#     t_steps = slide*n_fold
#     data_slice = data[:t_steps + test_step + window + 1]
#     MIN_slice = np.min(data_slice[:t_steps + window])
#     MAX_slice = np.max(data_slice[:t_steps + window])    

#     model = ReservoirArchitecture(seed, quantum_reservoir, reservoir_size, t_steps, test_step, window, alpha, reg, system_type[2], MIN_slice, MAX_slice)
#     Wout = model.fit(data_slice)
#     r2, mape, mse, nmse, pred = model.pred(data_slice, Wout, mode[0])
#     r2_tot.append(r2)
#     mape_tot.append(mape)
#     mse_tot.append(mse)
#     nmse_tot.append(nmse)
#     pred_tot.append(pred)

#     plt.figure(1).clear()
#     plt.plot(data_slice[t_steps+window:t_steps+window+len(pred)], "g")
#     plt.plot(np.arange(len(pred)), pred, "b")
#     plt.xlabel('Steps')
#     plt.ylabel('M-G Eq.')
#     plt.title(f"Target vs generated signals: {n_run} run, alpha={alpha}, reg={reg}")

#     plt.legend(["Target signal", "Free-running predicted signal"])
#     plt.savefig(f'v_1.0/Results/Plots/Expanding_CV_{n_fold}_fold_{system_type[2]}_{n_run}_{alpha}_{reg}_run_gen_mode.png')

# data_metrics = {"R2": r2_tot,"MAPE": mape_tot,"MSE": mse_tot,"NMSE": nmse_tot}
# df = pd.DataFrame(data_metrics)
# df.to_csv(f'v_1.0/Results/Metrics/Expanding_CV_metrics_{system_type[2]}_{n_run}_{alpha}_{reg}_run_gen_mode.csv', index=False, header=True)    






model = ReservoirArchitecture(seed, quantum_reservoir, reservoir_size, train_step, test_step, window, alpha, reg, system_type, MIN, MAX)

num_fold = 50
slide = 300

Wout_tot, r2_tot, mape_tot, mse_tot, nmse_tot, pred_tot = [], [], [], [], [], []    

print('CV start: ')
for n_fold in range(num_fold):   

    print('n fold: ', n_fold)

    data_slice = data[n_fold*slide:n_fold*slide + train_step + test_step + window + 1]
    MIN_slice = np.min(data_slice[:train_step + window])
    MAX_slice = np.max(data_slice[:train_step + window])

    model = ReservoirArchitecture(seed, quantum_reservoir, reservoir_size, train_step, test_step, window, alpha, reg, system_type[2], MIN_slice, MAX_slice)
    Wout = model.fit(data_slice)
    r2, mape, mse, nmse, pred = model.pred(data_slice, Wout, mode[0])
    r2_tot.append(r2)
    mape_tot.append(mape)
    mse_tot.append(mse)
    nmse_tot.append(nmse)
    pred_tot.append(pred)

    plt.figure(1).clear()
    plt.plot(data_slice[train_step+window:train_step+window+len(pred)], "g")
    plt.plot(np.arange(len(pred)), pred, "b")
    plt.xlabel('Steps')
    plt.ylabel('M-G Eq.')
    plt.title(f"Target vs generated signals: {n_run} run, alpha={alpha}, reg={reg}")

    plt.legend(["Target signal", "Free-running predicted signal"])
    plt.savefig(f'reservoir-computing/v_1.0/Results/Plots/CV_{n_fold}_fold_{system_type[2]}_{n_run}_{alpha}_{reg}_run_gen_mode.png')

data_metrics = {"R2": r2_tot,"MAPE": mape_tot,"MSE": mse_tot,"NMSE": nmse_tot}
df = pd.DataFrame(data_metrics)
df.to_csv(f'reservoir-computing/v_1.0/Results/Metrics/CV_metrics_{system_type[2]}_{n_run}_{alpha}_{reg}_run_gen_mode.csv', index=False, header=True)    














#----------------- Fine tuning -----------------#

#alpha = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]  #7
#alpha = [0.2]
#reg = [0.1,0.5,1.0,2.0,4.0,6.0,7.0,8.0,9.0,10.0,11.0,15.0]  #12
#reg = [5.5,6.0]#[5.0,5.3,5.5,5.8,6.0,6.2,6.4,6.6]

# r2_tot, mape_tot, mse_tot, pred_tot = [], [], [], []

# c = 0

# for a in alpha:
#     print('Step ', c+1)
#     for r in reg:
#         r2_mean, mape_mean, mse_mean, Wout_tot, pred_mean = multiple_run(n_run, data, seed, quantum_reservoir, reservoir_size, train_step, test_step, window, a, r, system_type[2], mode[0], MIN, MAX, plot=True)
#         r2_tot.append(r2_mean)
#         mape_tot.append(mape_mean)
#         mse_tot.append(mse_mean)
#         pred_tot.append(pred_mean)
#     c += 1
    
# data_metrics = {"R2": r2_tot,"MAPE": mape_tot,"MSE": mse_tot}
# df = pd.DataFrame(data_metrics)
# df.to_csv(f'v_1.0/Results/Metrics/Metrics_tot_{system_type[2]}_{n_run}_run_{t_end}_step_gen_mode.csv', index=False, header=True)    
# np.save(f'v_1.0/Results/Metrics/Predictions_tot_{n_run}_run_{system_type[2]}_{t_end}_step_gen_mode', pred_tot) 