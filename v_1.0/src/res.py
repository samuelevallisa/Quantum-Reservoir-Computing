import numpy as np
import random
from scipy import linalg
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt



def lin_reservoir(x, r, reservoir_size, a, W, A):     
 
    r = (1 - a) * r + a * np.tanh(np.dot(W, np.vstack((1, x))) + np.dot(A, r))
    #r = (1 - a) * r + a * np.tanh(np.dot(W, x) + np.dot(A, r))

    return r

def reservoir_model(data, win_size, seed, initLen, resSize, trainstep, teststep, mode, plot):

    # NB: reservoir performances should be averaged accross at least 30 random instances (with the same set of parameters)

    """
    mode = "prediction"  receives the real input at each time steps and tries to predict the next input
    mode = 'generative' the read-out output is feedback to the input INSTEAD of the real input. Thus the reservoir 
                        is in 'free run' mode, it does not receive real input anymore after training phase.

    load the data and select which parts are used for 'warming', 'training' and 'testing' the reservoir
    30 seems to be enough for initLen with leak_rate=0.3 and reservoir size (resSize) = 300
    initLen: number of time steps during which internal activations are washed-out during training
    trainstep: number of time steps during which we train the network
    teststep: number of time steps during which we test/run the network
    """

    # generate the ESN reservoir (Echo State Network)
    inSize = outSize = 1  # input/output dimension
    a = 0.3  # leaking rate  #TODO: try to change the value
    spectral_radius = 1.25  # TODO: try to change the value
    input_scaling = 1.0  # TODO: try to change the value
    reg = 1e-6  # None # regularization coefficient, if None, pseudo-inverse is use instead of ridge regression  


    random.seed(seed) #set_seed(seed) 
    W = (np.random.rand(resSize, 1 + inSize) - 0.5) * input_scaling  # shape = (reservoir_size,data_input_size)  
    A = np.random.rand(resSize, resSize) - 0.5   # shape = (reservoir_size,reservoir_size) 

    # Option 1 - direct scaling (quick&dirty, reservoir-specific):
    # W *= 0.135
    
    # Option 2 - normalizing and setting spectral radius (correct, slow):
    rhoW = max(abs(linalg.eig(A)[0]))
    A *= spectral_radius / rhoW

    # allocated memory for the design (collected states) matrix
    O_total = np.zeros((1 + inSize + resSize, trainstep - initLen))   # shape = (reservoir_size+input_size,train step-init len)
    # set the corresponding target matrix directly
    Yt = data[initLen + 1 : trainstep + 1]   # shape = (train step-init len,1)

    print('Yt: ', Yt.shape)

    # run the reservoir with the data and collect X
    r = np.zeros((resSize, 1))
    print('---- Evaluating Wout ----')
    for t in tqdm(range(trainstep)):
        x = data[t]  # select each sample per timestep

        # ESN update equation = we compute x(t+1) based on x(t) and input u(t)
        r = (1 - a) * r + a * np.tanh(np.dot(W, np.vstack((1, x))) + np.dot(A, r))
        if t >= initLen:
            O_total[:, t - initLen] = np.vstack((1, x, r))[:, 0]

    # train the output
    O_total_T = O_total.T
    if reg is not None:
        # use ridge regression (linear regression with regularization)
        Wout = np.dot(np.dot(Yt.T, O_total_T), linalg.inv(np.dot(O_total, O_total_T) + reg * np.eye(1 + inSize + resSize)))

    else:
        # use pseudo inverse
        Wout = np.dot(Yt, linalg.pinv(O_total))
    
    # run the trained ESN in a generative mode. no need to initialize here,
    # because x is initialized with training data and we continue from there.
    Y = np.zeros((outSize, teststep))
    x = data[trainstep]

    print('---- Evaluating Y ----')
    for t in tqdm(range(teststep-win_size)):
        r = (1 - a) * r + a * np.tanh(np.dot(W, np.vstack((1, x))) + np.dot(A, r))
        y = np.dot(Wout, np.vstack((1, x, r)))
        Y[:, t] = y

        if mode == "generative":
            # generative mode:
            x = y
        elif mode == "prediction":
            # predictive mode:
            x = data[trainstep + t + win_size]
        else:
            raise Exception("ERROR: 'mode' was not set correctly.")        

    # compute MSE for the first errorLen time steps
    errorLen = teststep  # 2000 #500
    mse = (sum(np.square(data[trainstep + 1 : trainstep + errorLen + 1] - Y[0, 0:errorLen]))/errorLen)


    if plot:
        
        plt.figure(1).clear()
        plt.plot(data, "g")
        plt.plot(Y.T, "b")
        plt.title("Target and generated signals")

        if mode == "generative":
            plt.legend(["Target signal", "Free-running predicted signal"])
            plt.savefig('reservoir-computing/v_1.0/res_gen_function')

        elif mode == "prediction":
            plt.legend(["Target signal", "Predicted signal"])
            plt.savefig('reservoir-computing/v_1.0/res_pred_function')
        else:
            raise Exception("ERROR: 'mode' was not set correctly.")

    return mse, Y.T
