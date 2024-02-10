import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mse_esn = pd.read_csv('reservoir-computing/v_1.0/Results/Metrics/MSE_mean_RC_ESN_100_run_gen_mode.csv')
mse_classical_rc = pd.read_csv('reservoir-computing/v_1.0/Results/Metrics/MSE_mean_Classical_RC_100_run_gen_mode.csv')

n_run = 100

fig, ax = plt.subplots()

rc = ['RC ESN', 'RC no vstack']
counts = [np.array(mse_esn)[0][0], np.array(mse_classical_rc)[0][0]]
#bar_labels = ['red', 'blue']
bar_colors = ['tab:red', 'tab:blue']

ax.bar(rc, counts, color=bar_colors)

ax.set_ylabel('MSE')
ax.set_title(f'Mean-Squared Error {n_run} run')
plt.savefig(f'reservoir-computing/v_1.0/Results/Plots/MSE_comparison_{n_run}_run.png')