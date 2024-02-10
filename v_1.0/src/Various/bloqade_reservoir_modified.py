from juliacall import Main as jl
import numpy as np
import os
current_dir=os.path.dirname(__file__)
jl.include(f"{current_dir}\\bloqade_call_modified.jl")
#jl.include(")
#jl.include(f"{__file__")#f"{current_dir}/
#C:\\Users\\vallisasa\\Desktop\\Scripts\\Reservoir_computing\\Reservoir_Matti\\qml_code\\reservoir-computing\\from_github\\reservoir
class BloqadeReservoir:

    """
    >>> python wrapper for bloade reservoir in julia.
    """
    def __init__(self,t_start: float,t_end: float,step: float,omega_max: float = 2*np.pi,nsites: int =10,d: float=6.0,lattice_type: str="square")->None:
        
        """
        arguments:
        >>> t_start: starting time of the evolution
        >>> t_end: ending time of the evolution
        >>> step: duration of each time step
        >>> omega_max: Rabi frequence
        >>> nsites: num atoms
        >>> d: atom distance
        >>> lattice_type: lattice structure

        """

        self.t_start=t_start
        self.t_end= t_end
        self.step= step
        self.omega_max= omega_max
        self.nsites = nsites
        self.distance = d
        self.lattice_type = lattice_type

    def _single_array_call(self,x: np.array)->np.array:

        """
        >>> calls the jl.bloqade_call function in bloqade_call.jl, simulating neutral atoms reservoir.
        >>> jl.bloade_call takes as argument a batched np array of shape (batch_size, len_sequence) 
            and other bloqade params fixed in self.init
        arguments:
        >>> x: np.array of shape (batch_size, len_seq)

        returns: a batched np array of shape (batch_size, n_atoms), 
            where values for each batch element as gathered as densities of bloade atoms.
        """
        return np.array(jl.bloqade_call_mod(x,self.t_start,self.t_end,self.step, self.omega_max, self.nsites, self.distance, self.lattice_type))

    def __call__(self, x: np.array) -> np.array:

        """
        >>> repeat call to jl.bloqade_call for each element in batch x.

        arguments:
        >>> x: np.array of shape (batch_size, len_seq)

        returns: a batched array of shape (batch_size, n_atoms)
        """
        return np.stack( [ self._single_array_call(x[i]) for i in range( x.shape[0] ) ], axis=0)
