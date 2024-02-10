using Bloqade
using PythonCall
# using KrylovKit
using SparseArrays

function bloqade_call_mod(x::PyArray, t_start::Float64, t_end::Float64, step::Float64, omega::Float64, nsites::Int, d::Float64, MIN::Float64, MAX::Float64, lattice_type::String)
    if lattice_type=="square"
        nx=floor(Int,sqrt(nsites))
        ny=floor(Int,sqrt(nsites))
        atoms=generate_sites(SquareLattice(),nx,ny; scale=d)
    elseif lattice_type=="chain"
        atoms=generate_sites(ChainLattice(),nsites;scale=d)
    elseif lattice_type=="honeycomb"
        atoms=generate_sites(HoneycombLattice(),nsites,nsites;scale=d)
    elseif lattice_type=="triangular"
        atoms=generate_sites(TriangularLattice(),nsites,nsites;scale=d)
    else
        throw(ArgumentError("invalid argument provided for Lattice structure"))
    end
    
    readouts = AbstractBlock[put(nsites, i => Z) for i in 1:nsites] #prepare the readouts matrix (Z_i and Z_i*Z_j)
    for i in 1:nsites
        for j in i+1:nsites
            push!(readouts, chain(put(nsites, i => Z), put(nsites, j => Z)))
        end
    end
    
    # if omega_schedule=="linear"
    #     omega_values = [omega*omega_max/(length(delta_values)-1) for omega in 0:(length(delta_values)-1)] # linear omega schedule; see TODO
    # elseif omega_schedule=="constant"
    #     omega_values = [omega_max for _ in 0:(length(delta_values)-1)] #constant schedule
    # else
    #     throw(ArgumentError("invalid argument provided for omega_schedule"))
    # end 

    # h = rydberg_h(atoms; Δ = Δ, Ω = Ω) #hamiltonian
    # r = zeros(Float64, 2^nsites)
    # r[1] = sqrt(x)
    # r[2^(nsites-1)+1] = sqrt(1-x)    
    # reg = arrayreg(complex(r))

    #spectral = max(abs(maximum(delta_values)), abs(minimum(delta_values)))
    #delta_values = delta_values/spectral*max_delta    

    
    delta_values = convert(Vector, x) #convert delta values from python to julia language
    max_delta = 6.0
    delta_values = (delta_values.-MIN)/(MAX-MIN)*max_delta*2 .-max_delta
    #clocks = collect(range(t_start, t_end, step=step))
    Δ = delta_values#piecewise_constant(clocks=clocks, values= 2π*delta_values)
    Ω = omega#piecewise_constant(clocks=clocks, values = 2π * omega)
    
    h = rydberg_h(atoms; Δ = Δ, Ω = Ω) #hamiltonian
    reg = zero_state(nsites)
    set_zero_state!(reg) #at the start of the simulation all atoms are in the ground state         
    steps = floor(Int, (t_end - t_start) / step)
    out = zeros(steps * length(readouts))    
    prob =  KrylovEvolution(reg, t_start:step:t_end, h)
    
    i = 1
    for (step, reg, _) in prob # step through the state at each time step 
        
        step == 1 && continue # ignore first time step, this is just the initial state
        
        for op in readouts
            out[i] = real(expect(op, reg)) # store the expectation of each operator for the given state in the output vector 
            i+=1
        end
    end
    
    return out
end



