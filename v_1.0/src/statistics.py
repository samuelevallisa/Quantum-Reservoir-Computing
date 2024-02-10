import numpy as np
import pandas as pd
from res import *
from model_reg import *


def multiple_run(n_run: int, data: np.array, seed: int, reservoir, reservoir_size: int, train_step: int, test_step:int, window: int, alpha: float, reg: float, system_type: str, mode: str, MIN: float, MAX: float, plot: bool) -> None:

    Wout_tot, r2_tot, mape_tot, mse_tot, nmse_tot, pred_tot = [], [], [], [], [], []    
    
    with tqdm(range(n_run), bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}') as tindex:

        for _ in tindex:
            model = ReservoirArchitecture(seed, reservoir, reservoir_size, train_step, test_step, window, alpha, reg, system_type, MIN, MAX)
            Wout = model.fit(data)
            Wout_tot.append(Wout)
            r2, mape, mse, nmse, pred = model.pred(data, Wout, mode)
            r2_tot.append(r2)
            mape_tot.append(mape)
            mse_tot.append(mse)
            nmse_tot.append(nmse)
            pred_tot.append(pred)

    arrays = [np.array(x) for x in pred_tot]
    pred_mean = [np.mean(k) for k in zip(*arrays)]
    r2_mean = np.mean(r2_tot)
    mape_mean = np.mean(mape_tot)
    mse_mean = np.mean(mse_tot)
    nmse_mean = np.mean(nmse_tot)

    data_metrics = {"R2": [r2_mean],"MAPE": [mape_mean],"MSE": [mse_mean],"NMSE": [nmse_mean]}
    df = pd.DataFrame(data_metrics)
    df.to_csv(f'v_1.0/Results/Metrics/Metrics_mean_{system_type}_{n_run}_{alpha}_{reg}_run_gen_mode.csv', index=False, header=True)    
    #np.save(f'v_1.0/Results/Metrics/Predictions_mean_{system_type[2]}_{n_run}_{alpha}_{reg}_run_gen_mode', pred_mean) 

    if plot:
        
        plt.figure(1).clear()
        plt.plot(data[train_step+window:train_step+window+len(pred)], "g")
        plt.plot(np.arange(len(pred)), pred, "b")
        plt.xlabel('Steps')
        plt.ylabel('M-G Eq.')
        plt.title(f"Target vs generated signals: {n_run} run, alpha={alpha}, reg={reg}")

        if mode == "generative":
            plt.legend(["Target signal", "Free-running predicted signal"])
            plt.savefig(f'v_1.0/Results/Plots/{system_type}_{n_run}_{alpha}_{reg}_run_gen_mode.png')

        elif mode == "prediction":
            plt.legend(["Target signal", "Predicted signal"])
            plt.savefig(f'v_1.0/Results/Plots/{system_type}_{n_run}_{alpha}_{reg}_run_pred_mode.png')
        else:
            raise Exception("ERROR: 'mode' was not set correctly.")             

    return r2_mean, mape_mean, mse_mean, nmse_mean, Wout_tot, pred_mean