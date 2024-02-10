import torch
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from scipy import linalg
import pandas as pd
from sklearn.metrics import r2_score

class ReservoirArchitecture:

    def __init__(self, seed, reservoir, reservoir_size, train_step, test_step, window, alpha, reg, system_type, min, max) -> None:

        """
        initialize reservoir architecture. 
        takes a reservoir (instance of a class or function).
        """

        random.seed(seed)

        self.output_size = 1
        self.input_size = 1 # number of input features
        #self.init_lenght = 100
        self.reservoir_size = reservoir_size # output dimention of the reservoir
        self.reservoir = reservoir
        self.train_step = train_step
        self.test_step = test_step
        self.alpha = alpha # leaking rate 
        self.reg = reg # regularization parameter       
        self.input_scaling = 1
        #self.spectral_radius = 1.25
        self.window = window
        self.system_type = system_type
        self.MIN = min
        self.MAX = max

        self.W = (np.random.rand(self.reservoir_size, 1 + self.input_size) - 0.5) * self.input_scaling  # shape = (reservoir_size,data_input_size)  
        self.A = np.random.rand(self.reservoir_size, self.reservoir_size) - 0.5   # shape = (reservoir_size,reservoir_size)    
        #print(self.A) 
        #rhoW = max(abs(linalg.eig(self.A)[0]))
        #self.A *= self.spectral_radius / rhoW  


    def __call__(self, u: np.array, *args) -> torch.Tensor: #eventually add x_prec if want to keep memory of past reservoir's outputs

      
        if len(args) != 0:
            
            self.r = args[0]
            self.reservoir_size = args[1]
            self.a = args[2]
            self.W = args[3]
            self.A = args[4]
               
            x = self.reservoir(u, self.r, self.reservoir_size, self.reg, self.W, self.A)
            
            return x
        
        else:  # bloqade case
            #print(u)
            x = self.reservoir(u, self.MIN, self.MAX)
            x = np.array(x, dtype=np.float32)  #dim (1,resDim)

            return x 

    
    def fit(self, data: np.array):

        train_data = data[:self.train_step + self.window]
        
        Yt = data[self.window + 1: self.train_step + self.window + 1]

        if self.system_type == 'RC_ESN':
            O_total = np.zeros((1 + self.input_size + self.reservoir_size, self.train_step - self.init_lenght))
        
        else:    
            #O_total = np.zeros((self.reservoir_size, self.train_step - self.init_lenght))  
            O_total = np.zeros((self.reservoir_size, self.train_step))       

        self.r = np.zeros((self.reservoir_size, 1))

        #print('---- Training ----')
        #for t in tqdm(range(self.train_step), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        for t in range(self.train_step):            
            
            if self.system_type == 'Classical_RC' or self.system_type == 'RC_ESN':

                u = data[t:t+self.window]
                self.r = self(u, self.r, self.reservoir_size, self.a, self.W, self.A)

            elif self.system_type == 'Bloqade':

                u = train_data[t:t+self.window]                
                #self.r = self(u)
                self.r = (1-self.alpha)*self.r + (self.alpha)* (np.expand_dims(self(u), axis=1)) 
               
                
            else:
                raise Exception("ERROR: 'system_type' was not set correctly.")

            #if t >= self.init_lenght:                
                
            if self.system_type == 'RC_ESN':   
                #O_total[:, t - self.init_lenght] = np.vstack((1, u, self.r))[:, 0]  
                O_total[:, t] = np.vstack((1, u, self.r))[:, 0]                       
            
            elif self.system_type == 'Bloqade':                               
                #print('r[:, 0]: ', self.r[:, 0])                  
                O_total[:, t] = np.vstack((self.r))[:, 0]  

            else:
                #O_total[:, t - self.init_lenght] = self.r[:,0] 
                O_total[:, t] = self.r[:,0]             

        O_total_T = O_total.T

        if self.system_type == 'RC_ESN':
            Wout = np.dot(np.dot(Yt.T, O_total_T), linalg.inv(np.dot(O_total, O_total_T) + self.reg * np.eye(1 + self.reservoir_size + self.input_size)))

        else:
            Wout = np.dot(np.dot(Yt.T, O_total_T), linalg.inv(np.dot(O_total, O_total_T) + self.reg * np.eye(self.reservoir_size)))

        return Wout

    def pred(self, data: np.array, Wout: np.array, mode: str):

        test_wave = data[self.train_step + 1:self.train_step + self.test_step + self.window + 1]
        Y = test_wave#[None,:]
        Y_pred = np.zeros((self.output_size, self.test_step))
        x = data[self.train_step]
        u = test_wave[:self.window]
       
        #print('---- Predictions ----')
        #for t in tqdm(range(self.test_step-1), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        for t in range(self.test_step):

            if self.system_type == 'RC_ESN':
                self.r = (1 - self.alpha) * self.r + self.alpha * np.tanh(np.dot(self.W, np.vstack((1, u))) + np.dot(self.A, self.r))
                y = np.dot(Wout, np.vstack((1, x, self.r)))

            elif self.system_type == 'Bloqade':

                #self.r = (1 - self.alpha) * self.r + self.alpha *np.squeeze(self(x), axis=0)
                #self.r = self(x)
                #y = np.dot(Wout, np.squeeze(self.r, axis=0))
                u = u[-self.window:]                
                self.r = (1 - self.alpha) * self.r + (self.alpha)*(np.expand_dims(self(u), axis=1))                
                y = np.dot(Wout, np.vstack((self.r)))
            
            else:
                self.r = (1 - self.alpha) * self.r + self.alpha * np.tanh(np.dot(self.W, np.vstack((1, u))) + np.dot(self.A, self.r))
                y = np.dot(Wout, self.r)

            #Y_pred[:, t] = y

            if mode == "generative":
                # generative mode:
                #x = y
                Y_pred[:, t] = y
                u = np.append(u, y, axis=0)

            elif mode == "prediction":
                # predictive mode:
                u = data[self.train_step + t]
            else:
                raise Exception("ERROR: 'mode' was not set correctly.")
            
        # compute MSE for the first errorLen time steps
        mse = np.square(np.subtract(Y[self.window:], Y_pred.T)).mean()
        mape = (np.abs(np.subtract(Y[self.window:].T[0,:], Y_pred[0,:]))/Y[self.window:].T[0,:]).mean()      
        r2 = r2_score(Y[self.window:].T[0,:], Y_pred[0,:])

        mse0 = np.square(Y[self.window:]).mean()
        nmse=mse/mse0

        # print('R2 score: ', r2)
        # print('MSE: ', mse)
        # print('MAPE: ', mape)
        
        #errorLen = self.test_step  # 2000 #500        
        #mse = (sum(np.square(data[self.train_step + 1 : self.train_step + errorLen + 1] - Y_pred[0, 0:errorLen]))/errorLen)

        return r2, mape, mse, nmse, Y_pred.T
        
        









        

