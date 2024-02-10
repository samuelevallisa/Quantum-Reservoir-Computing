from juliacall import Main as jl
import numpy as np

jl.include(f"{__package__}/bloqade_call.jl")

class BloqadeReservoir:

    """
    >>> python wrapper for bloade reservoir in julia.
    """
    
    def __init__(self, clock_delta: float = 1.0, omega_max: float = 2*np.pi*4.3, omega_schedule: str = "linear" ) -> None:
        """
        arguments: 
        >>> clock_delta: time difference between each clock in bloqade.
        >>> omega_max: max value for Rabi frequence in bloqade; defines linear schedule of omegas from 0 to omega_max
        """

        self.clock_delta = clock_delta
        self.omega_max = omega_max
        self.omega_schedule = omega_schedule
    
    
    def _single_array_call(self, x: np.array) -> np.array:

        """
        >>> calls the jl.bloqade_call function in bloqade_call.jl, simulating neutral atoms reservoir.
        >>> jl.bloade_call takes as argument a batched np array of shape (batch_size, len_sequence) 
            and other bloqade params fixed in self.init
        arguments:
        >>> x: np.array of shape (batch_size, len_seq)

        returns: a batched np array of shape (batch_size, n_atoms), 
            where values for each batch element as gathered as densities of bloade atoms.
        """
        return np.array(jl.bloqade_call(x, self.clock_delta, self.omega_max, self.omega_schedule))
    
    def __call__(self, x: np.array) -> np.array:

        """
        >>> repeat call to jl.bloqade_call for each element in batch x.

        arguments:
        >>> x: np.array of shape (batch_size, len_seq)

        returns: a batched array of shape (batch_size, n_atoms)
        """
        return np.stack( [ self._single_array_call(x[i]) for i in range( x.shape[0] ) ], axis=0)
