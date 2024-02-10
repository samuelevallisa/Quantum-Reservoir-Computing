from .data_utils import DataSet
import torch
import numpy as np

#TODO: manage inputs with more features

class ReservoirArchitecture:

    """
    >>> reservoir architectures. 
    >>> to init, specify a reservoir class instance or function.
    >>> if reservoir is a class instance, ensure it is provided with a __call__ function.
    >>> 
    """

    

    def __init__(self, reservoir, trainable_mlp: torch.nn.Module) -> None:

        """
        >>> initialize reservoir architecture. 
        >>> takes a reservoir (instance of a class or function).
        >>> if reservoir is a class, ensure it is provided with a __call__ method 
        >>> trainable mlp is a general multilayer perceptron implemented in torch

        >>> ensure mlp output layer has a number of neurons matching number of features in the sequence
        >>> up to now, train with one feature only is supported.
        """

        
        self.reservoir = reservoir
        self.trainable_mlp = trainable_mlp
        self.compiled = False
    
    def __call__(self, x: np.array) -> torch.Tensor:

        """
        >>> compute the output of ther reservoir architecure.

        arguments:
        >>> x: np.array with shape (batch_size, window)
        """

        x = self.reservoir(x)
        x = np.array(x, dtype=np.float32)
        x = self.trainable_mlp( torch.tensor(x, dtype=torch.float32) )
        return x
    
    def compile(self, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.compiled = True

    
    def fit(self, data: np.array, steps: int, batch_size: int, window: int) -> None:

        """
        >>> fit reservoir architecture on a dataset.

        arguments:
        >>> data: np.array with shape (n_data, n_features = 1) TODO: manage case n_features > 1.
        >>> steps: number of train steps.
        >>> batch_size: dimension of batch for each train step.
        >>> window: number of backward steps to consider per each sequence.

        >>> when training, batches of shape (batch_size, window+1) are fed to the model
        >>> with window+1 being the length of the sequences.
        >>> model is trained to predict next steps in the sequence

        """

        if not self.compiled:
            raise Exception("please compile the model before calling fit method")

        dataset = DataSet(data)

        for step in range(steps):
            x,y = next(dataset(batch_size, window))
            y_pred = self(x) 
            loss = self.loss_fn(y_pred, torch.tensor(y, dtype=torch.float32))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


            if step+1==1 or (step+1)%10==0:

                print("fitting at step: ",step+1)
                print( "loss: ", loss.item() )

    def simulate(self, x_0: np.array, steps: int, window:int ):

        """
        >>> simulate the dynamic of the system starting from an initial state, 
            TODO: solve problem of non matching lens (regarding window, initial_state, related to bloqade_call.jl)
        >>> x_0 : initial state for the simulation, 
        >>> steps : how many extra steps to simulate 
        >>> window: how many steps back in time feeding to the model for the computation of each output, 
            other than the last predicted value (len of input sequence for each feature is window+1)
        """
        
        simulated = np.array(x_0).reshape(1,-1) #shape (1, len_seq) where 1 is the "batch_size" in simulation

        for step in range(steps):
            
            if step+1==1 or (step+1)%10==0:
                print("simulating step: ", step+1)

            y = self(simulated[:, -(window+1):]) #y returned as array of shape (1, 1) 
            simulated = np.append( simulated, y.detach().numpy(), axis=1) # here simulated is (1, len_sq(step-1) + 1)


        return simulated.reshape(-1,1)