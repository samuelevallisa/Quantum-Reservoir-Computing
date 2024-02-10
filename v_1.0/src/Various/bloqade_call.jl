# import Pkg
# Pkg.add("Bloqade")
#module bloqade_caller 

using Bloqade
using PythonCall
# using KrylovKit
using SparseArrays

#TODO: implement extra schedules for omega
#TODO: explain clocks schedules bug; need a lower for clocks[-1] ?
#TODO: interface for various lattices

function bloqade_call(delta_values::PyArray, clock_delta::Float64, omega_max ::Float64, omega_schedule::String = "linear")


    delta_values = convert(Vector, delta_values)

    if omega_schedule=="linear"
        omega_values = [omega*omega_max/(length(delta_values)-1) for omega in 0:(length(delta_values)-1)] # linear omega schedule; see TODO
    elseif omega_schedule=="constant"
        omega_values = [omega_max for _ in 0:(length(delta_values)-1)] #constant schedule
    else
        throw(ArgumentError("invalid argument provided for omega_schedule"))
    end


    clocks = [ t*clock_delta for t in 0:(length(delta_values)-1) ] # linear clocks schedule by default
    clocks = convert( Vector, clocks )
    
    BloqadeLattices.DEFAULT_BACKGROUND_COLOR[] = "#FFFFFF" # set white background

    # Square lattice
    nx, ny = 3, 3
    #nsites = nx * ny
    # atoms = generate_sites(SquareLattice(), nx, ny, scale = 6.7) # generate lattice, close atoms = Rydberg blockade
    atoms = generate_sites(SquareLattice(), nx, ny, scale = 6.0)

    # Define the pulses
    total_time = clocks[end] #2.9
    
    omega = piecewise_linear(clocks = clocks, values = omega_values); # Rabi frequencies values=[0, omega_max, omega_max, 0]
    # omega = smooth(omega; kernel_radius = 0.05); # function to smooth the pulse
    # Rabi phase Φ is set to zero

    delta = piecewise_linear(clocks = clocks, values = delta_values ); # detunings    
    # delta = smooth(delta; kernel_radius = 0.05);

    # Define the dynamics
    h = rydberg_h(atoms; Δ=delta, Ω=omega) # define the hamiltonian
    reg = zero_state(size(atoms)[1]); # define a register with the same number of qubits as the lattice
    prob = SchrodingerProblem(reg, total_time, h); # define the dynamics as a Schrodinger equation (standard ODE)
    integrator = init(prob, Vern8()); # define solver: Vern8()

    # Solve the dynamics iteratively and append the results
    densities = [];
    for _ in TimeChoiceIterator(integrator, 0.0:1e-3:total_time)
        push!(densities, rydberg_density(reg))
    end
    D = hcat(densities...)

    # Get spatial positions and densities of atoms
    xs = []
    ys = []
    Ds = []
    for i in 1:size(atoms)[1]
        append!(xs, atoms[i][1])
        append!(ys, atoms[i][2])
        # append!(ys, 0.0) # if ChainLattice
        append!(Ds, D[i, size(D)[2]])
    end

    return Ds

end




