using Bloqade
using PythonCall
plt = pyimport("matplotlib.pyplot")

"""
Plottare le heatmap di H seguendo:

H valutata in istanti temporali diversi:

1. phi = 0, omega fisso = 30.0, delta varia = 2π*[-6.0, -6.0, 6.0, 6.0]
2. phi = 0, delta fisso = 10.1, omega varia = 2π*[0.0, 6.0, 6.0, 0.0]

H valutata con variabili a valori fissati:

1. phi = 0, omega = 0, delta = 0
2. phi = 0, omega = 0, delta fisso (10.1)
3. phi = 0, omega fisso (10.0), delta fisso (10.1)

d diverse (5.72):

1. phi = 0, omega = 4, delta fisso (10.1), d = 5.7, 6.2, 6.7, 7.2, 7.7
2. phi = 0, omega = 10, delta fisso (10.1), d = 4.9, 5.4, 5.9, 6.4

Testare le varie forme d'onda per omega, delta e phi: piecewise_linear, 
piecewise_constant, linear_ramp, sinusoidal
"""

function h_test(omega::Array{Float64}, delta::Array{Float64}, phi::Array{Float64}, clocks::Array{Float64}, d::Float64=5.72, params_constant::Bool=true)
    
    atoms = generate_sites(ChainLattice(), 5, scale=d)

    if params_constant == true

        h = rydberg_h(atoms; Δ=delta[1], Ω=omega[1], ϕ=phi[1])
        ht = h |> attime(clocks[1])

        plt.matshow(real(mat(ht)), cmap="cool")
        plt.colorbar()
        plt.title("distance=$(d) Ω=$(round(omega[1], digits=3))  Δ=$(round(delta[1], digits=3))")
        plt.savefig("v_1.0\\Results\\Hamiltonian_test\\H_heatmaps\\Fixed_params\\H_heatmap_$(d)_$(round(omega[1], digits=3))_$(round(delta[1], digits=3)).png")
          
    else
       
        Ω = piecewise_linear(clocks=clocks , values=omega)
        Δ = piecewise_linear(clocks=clocks, values=delta)
        ϕ = piecewise_constant(clocks=clocks, values=phi)

        fig_1=Bloqade.plot(Ω)
        fig_2=Bloqade.plot(Δ)
      
        fig_1.savefig("v_1.0\\Results\\Hamiltonian_test\\Params\\Varying\\omega.png")
        fig_2.savefig("v_1.0\\Results\\Hamiltonian_test\\Params\\Varying\\delta.png")
        plt.close(fig_1)
        plt.close(fig_2)
        
        time = range(0.0, clocks[end], step=0.15)
        for i in 1:length(time)
            h = rydberg_h(atoms; Δ=Δ, Ω=Ω, ϕ=ϕ)
            ht= h |> attime(time[i])
            plt.matshow(real(mat(ht)), cmap="cool")
            plt.colorbar()
            plt.title("timestep=$(time[i]) \n distance=$(d)  Ω=$(round(sample_values(Ω,time[i]:time[i])[1],digits=3))  Δ=$(round(sample_values(Δ,time[i]:time[i])[1], digits=3))")
            if i < 10 
                plt.savefig("v_1.0\\Results\\Hamiltonian_test\\H_heatmaps\\Params\\H_t_0$(i).png")
            else
                plt.savefig("v_1.0\\Results\\Hamiltonian_test\\H_heatmaps\\Params\\H_t_$(i).png")
            end
            plt.close()
        end

    end

end

#omega=2π*[0.0, 6.0, 6.0, 0.0]
omega=2π*[4.0, 4.0, 4.0, 4.0]
#delta=2π*[-6.0, -6.0, 6.0, 6.0]
delta=2π*[10.1, 10.1, 10.1, 10.1]
phi=2π*[0.0, 0.0, 0.0]
clocks=[0.0, 0.4, 1.8, 2.2]
#d = 5.72
d = 10.8#5.7#,6.2#,6.7#,7.2,#7.9,#8.4,8.9#,9.4,9.9#10.4#10.8
#d = 7.4#4.9#,5.4#,5.9#,6.4#6.9,#7.4,#7.9,#8.4,#8.9
h_test(omega, delta, phi, clocks, d, true)

