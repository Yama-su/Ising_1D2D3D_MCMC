using Random
using Statistics # For the mean function
using Printf # For formatting output
using Plots # For plotting

# --- Common Accumulator Structure ---
# Based on the 3D code's Accumulator, specialized for scalar measurements
mutable struct Accumulator
    count::Dict{String,UInt64}
    data::Dict{String,Float64} # Fixed to Float64 for scalar measurements
end

function Accumulator()
    Accumulator(Dict{String,UInt64}(), Dict{String,Float64}())
end

function add!(acc::Accumulator, name::String, data::Float64)
    if haskey(acc.count, name)
        acc.count[name] += 1
        acc.data[name] += data
    else
        acc.count[name] = 1
        acc.data[name] = data
    end
end

function mean(acc::Accumulator, name::String)
    return acc.data[name] / acc.count[name]
end

# --- 1D Ising Model ---

# SpinState1D Structure
mutable struct SpinState1D
    num_spins::Int
    s::Array{Int8,1}
    energy::Float64 # Energy stored as a floating-point number
    tot_mag::Float64 # Magnetization also stored as a floating-point number
end

# Energy Calculation Function (1D)
function calculate_energy_1d(s::Array{Int8,1})
    N = length(s)
    energy = 0.0
    for i in 1:N
        # Periodic boundary conditions
        energy -= s[i] * s[mod1(i + 1, N)]
    end
    return energy
end

# Magnetization Calculation Function (1D)
total_magnetization_1d(s::Array{Int8,1}) = sum(s)

# SpinState1D Constructor
function SpinState1D(s::Array{Int8,1})
    num_spins = length(s)
    SpinState1D(num_spins, copy(s), calculate_energy_1d(s), Float64(total_magnetization_1d(s)))
end

# Sanity check (1D)
function sanity_check_1d(ss::SpinState1D)
    @assert abs(calculate_energy_1d(ss.s) - ss.energy) < 1e-9 "Energy mismatch for 1D!"
    @assert abs(total_magnetization_1d(ss.s) - ss.tot_mag) < 1e-9 "Magnetization mismatch for 1D!"
end

# Spin Update Function (1D - Metropolis Algorithm)
function update_1d!(ss::SpinState1D, β::Float64, niters::Int, rng::AbstractRNG)
    N = ss.num_spins
    s = ss.s
    
    for iter in 1:niters
        for _ in 1:N # Attempt to update each spin in random order
            i = rand(rng, 1:N)
            
            # Nearest neighbor spins (periodic boundary conditions)
            sl = s[mod1(i - 1, N)]
            sr = s[mod1(i + 1, N)]
            
            # Effective magnetic field
            h = sl + sr
            
            si_old = s[i]
            
            # Energy change from spin flip ΔE = 2 * h * S_old
            # Assuming J=1. Hamiltonian H = -J * sum(Si * Sj), so ΔE = (H_new - H_old) = -J * (Si_new*Sum(Nj) - Si_old*Sum(Nj)) = -J * (-Si_old*h - Si_old*h) = 2*J*h*Si_old
            delta_E = 2.0 * h * si_old
            
            # Metropolis acceptance probability
            if delta_E <= 0 || rand(rng) < exp(-β * delta_E)
                s[i] *= -1 # Flip spin
                
                # O(1) update of observables
                ss.energy += delta_E
                ss.tot_mag += (s[i] - si_old)
            end
        end
    end
end

# Simulation Execution Function (1D)
function solve_1d!(ss::SpinState1D, acc::Accumulator, β::Float64, nsweeps::Int, ntherm::Int, interval_meas::Int, rng::AbstractRNG)
    if mod(nsweeps, interval_meas) != 0
        error("nsweeps must be divisible by interval_meas!")
    end
    
    # Thermalization steps
    update_1d!(ss, β, ntherm, rng)
    
    # Measurement steps
    for imeas in 1:(nsweeps ÷ interval_meas)
        update_1d!(ss, β, interval_meas, rng)
        add!(acc, "E", ss.energy)
        add!(acc, "E2", ss.energy^2)
        add!(acc, "M2", ss.tot_mag^2) # Measure squared magnetization
    end
    sanity_check_1d(ss) # Final consistency check
end


# --- 2D Ising Model ---

# SpinState2D Structure
mutable struct SpinState2D
    num_spins::Int
    s::Array{Int8,2}
    energy::Float64 # Energy stored as a floating-point number
    tot_mag::Float64 # Magnetization also stored as a floating-point number
end

# Energy Calculation Function (2D)
function calculate_energy_2d(s::Array{Int8,2})
    L = size(s, 1)
    energy = 0.0
    for x in 1:L
        for y in 1:L
            # Calculate interaction with 4 nearest neighbors for each spin
            # Apply periodic boundary conditions (mod1 is convenient for Julia's 1-based indexing)
            energy -= s[x, y] * (
                s[mod1(x + 1, L), y] +
                s[mod1(x - 1, L), y] +
                s[x, mod1(y + 1, L)] +
                s[x, mod1(y - 1, L)]
            )
        end
    end
    # Each interaction is counted twice, so divide by 2
    return energy / 2.0
end

# Magnetization Calculation Function (2D)
total_magnetization_2d(s::Array{Int8,2}) = sum(s)

# SpinState2D Constructor
function SpinState2D(s::Array{Int8,2})
    num_spins = length(s)
    SpinState2D(num_spins, copy(s), calculate_energy_2d(s), Float64(total_magnetization_2d(s)))
end

# Sanity check (2D)
function sanity_check_2d(ss::SpinState2D)
    @assert abs(calculate_energy_2d(ss.s) - ss.energy) < 1e-9 "Energy mismatch for 2D!"
    @assert abs(total_magnetization_2d(ss.s) - ss.tot_mag) < 1e-9 "Magnetization mismatch for 2D!"
end

# Spin Update Function (2D - Metropolis Algorithm)
function update_2d!(ss::SpinState2D, β::Float64, niters::Int, rng::AbstractRNG)
    L = size(ss.s, 1)
    s = ss.s
    
    for iter in 1:niters
        for _ in 1:ss.num_spins # Update each spin in random order
            x, y = rand(rng, 1:L), rand(rng, 1:L)
            
            # Nearest neighbor spins (periodic boundary conditions)
            NN = s[mod1(x - 1, L), y]
            SS = s[mod1(x + 1, L), y]
            WW = s[x, mod1(y - 1, L)]
            EE = s[x, mod1(y + 1, L)]
            
            # Effective magnetic field
            h = NN + SS + WW + EE
            
            si_old = s[x, y]
            
            # Energy change from spin flip ΔE = 2 * h * S_old
            delta_E = 2.0 * h * si_old
            
            # Metropolis acceptance probability
            if delta_E <= 0 || rand(rng) < exp(-β * delta_E)
                s[x, y] *= -1 # Flip spin
                
                # O(1) update of observables
                ss.energy += delta_E
                ss.tot_mag += (s[x, y] - si_old)
            end
        end
    end
end

# Simulation Execution Function (2D)
function solve_2d!(ss::SpinState2D, acc::Accumulator, β::Float64, nsweeps::Int, ntherm::Int, interval_meas::Int, rng::AbstractRNG)
    if mod(nsweeps, interval_meas) != 0
        error("nsweeps must be divisible by interval_meas!")
    end
    
    # Thermalization steps
    update_2d!(ss, β, ntherm, rng)
    
    # Measurement steps
    for imeas in 1:(nsweeps ÷ interval_meas)
        update_2d!(ss, β, interval_meas, rng)
        add!(acc, "E", ss.energy)
        add!(acc, "E2", ss.energy^2)
        add!(acc, "M2", ss.tot_mag^2) # Measure squared magnetization
    end
    sanity_check_2d(ss) # Final consistency check
end


# --- 3D Ising Model ---
# SpinState3D Structure (unchanged)
mutable struct SpinState3D
    num_spins::Int
    s::Array{Int8,3}
    energy::Float64
    tot_mag::Float64
end

# Energy Calculation Function (3D) (unchanged)
function calculate_energy_3d(s::Array{Int8,3})
    L = size(s, 1)
    energy = 0.0
    for x in 1:L
        for y in 1:L
            for z in 1:L
                energy -= s[x, y, z] * (
                    s[mod1(x + 1, L), y, z] +
                    s[mod1(x - 1, L), y, z] +
                    s[x, mod1(y + 1, L), z] +
                    s[x, mod1(y - 1, L), z] +
                    s[x, y, mod1(z + 1, L)] +
                    s[x, y, mod1(z - 1, L)]
                )
            end
        end
    end
    return energy / 2.0
end

# Magnetization Calculation Function (3D) (unchanged)
total_magnetization_3d(s::Array{Int8,3}) = sum(s)

# SpinState3D Constructor (unchanged)
function SpinState3D(s::Array{Int8,3})
    num_spins = length(s)
    SpinState3D(num_spins, copy(s), calculate_energy_3d(s), Float64(total_magnetization_3d(s)))
end

# Sanity check (3D) (unchanged)
function sanity_check_3d(ss::SpinState3D)
    @assert abs(calculate_energy_3d(ss.s) - ss.energy) < 1e-9 "Energy mismatch!"
    @assert abs(total_magnetization_3d(ss.s) - ss.tot_mag) < 1e-9 "Magnetization mismatch!"
end

# Spin Update Function (3D - Metropolis Algorithm) (unchanged)
function update_3d!(ss::SpinState3D, β::Float64, niters::Int, rng::AbstractRNG)
    L = size(ss.s, 1)
    s = ss.s
    
    for iter in 1:niters
        for _ in 1:ss.num_spins
            x, y, z = rand(rng, 1:L), rand(rng, 1:L), rand(rng, 1:L)
            
            NN = s[mod1(x + 1, L), y, z]
            SS = s[mod1(x - 1, L), y, z]
            WW = s[x, mod1(y + 1, L), z]
            EE = s[x, mod1(y - 1, L), z]
            FF = s[x, y, mod1(z + 1, L)]
            BB = s[x, y, mod1(z - 1, L)]
            
            h = NN + SS + WW + EE + FF + BB
            
            si_old = s[x, y, z]
            
            delta_E = 2.0 * h * si_old
            
            if delta_E <= 0 || rand(rng) < exp(-β * delta_E)
                s[x, y, z] *= -1
                
                ss.energy += delta_E
                ss.tot_mag += (s[x, y, z] - si_old)
            end
        end
    end
end

# Simulation Execution Function (3D) (unchanged)
function solve_3d!(ss::SpinState3D, acc::Accumulator, β::Float64, nsweeps::Int, ntherm::Int, interval_meas::Int, rng::AbstractRNG)
    if mod(nsweeps, interval_meas) != 0
        error("nsweeps must be divisible by interval_meas!")
    end
    
    # Thermalization steps
    update_3d!(ss, β, ntherm, rng)
    
    # Measurement steps
    for imeas in 1:(nsweeps ÷ interval_meas)
        update_3d!(ss, β, interval_meas, rng)
        add!(acc, "E", ss.energy)
        add!(acc, "E2", ss.energy^2)
        add!(acc, "M2", ss.tot_mag^2)
    end
    sanity_check_3d(ss)
end


# --- Simulation Execution and Plotting ---

println("--- 1D Ising Model Simulation ---")
L_1d = 100 # Lattice size
num_spins_1d = L_1d
nsweeps_1d = 100000 # Number of Monte Carlo steps at each temperature
ntherm_1d = 20000 # Number of thermalization steps
interval_meas_1d = 10 # Measurement interval

temperatures_1d = range(0.1, stop=3.0, length=30)
specific_heats_1d = Float64[]
magnetic_susceptibilities_1d = Float64[]

rng_1d = MersenneTwister(1234)

for T in temperatures_1d
    println(@sprintf(" 1D: Simulating at Temperature T = %.2f...", T))
    s0_1d = rand(rng_1d, Int8[-1, 1], L_1d)
    ss_1d = SpinState1D(s0_1d)
    acc_1d = Accumulator()
    
    solve_1d!(ss_1d, acc_1d, 1.0/T, nsweeps_1d, ntherm_1d, interval_meas_1d, rng_1d)
    
    E_avg_1d = mean(acc_1d, "E")
    E2_avg_1d = mean(acc_1d, "E2")
    M2_avg_1d = mean(acc_1d, "M2")
    
    C_1d = (E2_avg_1d - E_avg_1d^2) / (T^2 * num_spins_1d) # Divide by N
    push!(specific_heats_1d, C_1d)
    
    Chi_1d = M2_avg_1d / (T * num_spins_1d) # Divide by N
    push!(magnetic_susceptibilities_1d, Chi_1d)
end

println("1D Simulation complete.")
println("Calculated Specific Heat (1D, C/N): ", specific_heats_1d)
println("Calculated Magnetic Susceptibility (1D, Chi/N): ", magnetic_susceptibilities_1d)

# Plot 1D Specific Heat
p1_1d = plot(temperatures_1d, specific_heats_1d, 
             xlabel="Temperature T", 
             ylabel="Specific Heat C/N", 
             title="1D Ising Model Specific Heat", 
             legend=false,
             marker=:circle)
savefig(p1_1d, "specific_heat_1d_ising.png")
println("1D specific heat plot saved as specific_heat_1d_ising.png.")

# Plot 1D Magnetic Susceptibility
p2_1d = plot(temperatures_1d, magnetic_susceptibilities_1d, 
             xlabel="Temperature T", 
             ylabel="Magnetic Susceptibility Chi/N", 
             title="1D Ising Model Magnetic Susceptibility", 
             legend=false,
             marker=:circle)
savefig(p2_1d, "magnetic_susceptibility_1d_ising.png")
println("1D magnetic susceptibility plot saved as magnetic_susceptibility_1d_ising.png.")


println("\n--- 2D Ising Model Simulation ---")
L_2d = 32 # Lattice size (L x L)
num_spins_2d = L_2d^2
nsweeps_2d = 200000 # Number of Monte Carlo steps at each temperature (adjusted)
ntherm_2d = 50000 # Number of thermalization steps (adjusted)
interval_meas_2d = 10 # Measurement interval

# Critical temperature for 2D Ising model is T_c ≈ 2.269 (for J=1, k_B=1)
temperatures_2d = range(1.5, stop=3.0, length=25) # Range including critical temperature
specific_heats_2d = Float64[]
magnetic_susceptibilities_2d = Float64[]

rng_2d = MersenneTwister(5678)

for T in temperatures_2d
    println(@sprintf(" 2D: Simulating at Temperature T = %.2f...", T))
    s0_2d = rand(rng_2d, Int8[-1, 1], L_2d, L_2d)
    ss_2d = SpinState2D(s0_2d)
    acc_2d = Accumulator()
    
    solve_2d!(ss_2d, acc_2d, 1.0/T, nsweeps_2d, ntherm_2d, interval_meas_2d, rng_2d)
    
    E_avg_2d = mean(acc_2d, "E")
    E2_avg_2d = mean(acc_2d, "E2")
    M2_avg_2d = mean(acc_2d, "M2")
    
    C_2d = (E2_avg_2d - E_avg_2d^2) / (T^2 * num_spins_2d) # Divide by N
    push!(specific_heats_2d, C_2d)
    
    Chi_2d = M2_avg_2d / (T * num_spins_2d) # Divide by N
    push!(magnetic_susceptibilities_2d, Chi_2d)
end

println("2D Simulation complete.")
println("Calculated Specific Heat (2D, C/N): ", specific_heats_2d)
println("Calculated Magnetic Susceptibility (2D, Chi/N): ", magnetic_susceptibilities_2d)

# Plot 2D Specific Heat
p1_2d = plot(temperatures_2d, specific_heats_2d, 
             xlabel="Temperature T", 
             ylabel="Specific Heat C/N", 
             title="2D Ising Model Specific Heat", 
             legend=false,
             marker=:circle)
savefig(p1_2d, "specific_heat_2d_ising.png")
println("2D specific heat plot saved as specific_heat_2d_ising.png.")

# Plot 2D Magnetic Susceptibility
p2_2d = plot(temperatures_2d, magnetic_susceptibilities_2d, 
             xlabel="Temperature T", 
             ylabel="Magnetic Susceptibility Chi/N", 
             title="2D Ising Model Magnetic Susceptibility", 
             legend=false,
             marker=:circle)
savefig(p2_2d, "magnetic_susceptibility_2d_ising.png")
println("2D magnetic susceptibility plot saved as magnetic_susceptibility_2d_ising.png.")


println("\n--- 3D Ising Model Simulation ---")
L_3d = 8 # Lattice size (L x L x L)
num_spins_3d = L_3d^3
nsweeps_3d = 50000 # Number of Monte Carlo steps at each temperature
ntherm_3d = 10000 # Number of thermalization steps
interval_meas_3d = 10 # Measurement interval

# Critical temperature for 3D Ising model is T_c ≈ 4.51 (for J=1)
temperatures_3d = range(2.0, stop=6.0, length=20) # Range including critical temperature

specific_heats_3d = Float64[]
magnetic_susceptibilities_3d = Float64[]

rng_3d = MersenneTwister(4649)

for T in temperatures_3d
    println(@sprintf(" 3D: Simulating at Temperature T = %.2f...", T))
    s0_3d = rand(rng_3d, Int8[-1, 1], L_3d, L_3d, L_3d)
    ss_3d = SpinState3D(s0_3d)
    acc_3d = Accumulator()
    
    solve_3d!(ss_3d, acc_3d, 1.0/T, nsweeps_3d, ntherm_3d, interval_meas_3d, rng_3d)
    
    E_avg_3d = mean(acc_3d, "E")
    E2_avg_3d = mean(acc_3d, "E2")
    M2_avg_3d = mean(acc_3d, "M2")
    
    C_3d = (E2_avg_3d - E_avg_3d^2) / (T^2 * num_spins_3d)
    push!(specific_heats_3d, C_3d)
    
    Chi_3d = M2_avg_3d / (T * num_spins_3d)
    push!(magnetic_susceptibilities_3d, Chi_3d)
end

println("3D Simulation complete.")
println("Calculated Specific Heat (3D, C/N): ", specific_heats_3d)
println("Calculated Magnetic Susceptibility (3D, Chi/N): ", magnetic_susceptibilities_3d)

# Plot 3D Specific Heat
p1_3d = plot(temperatures_3d, specific_heats_3d, 
             xlabel="Temperature T", 
             ylabel="Specific Heat C/N", 
             title="3D Ising Model Specific Heat", 
             legend=false,
             marker=:circle)
savefig(p1_3d, "specific_heat_3d_ising.png")
println("3D specific heat plot saved as specific_heat_3d_ising.png.")

# Plot 3D Magnetic Susceptibility
p2_3d = plot(temperatures_3d, magnetic_susceptibilities_3d, 
             xlabel="Temperature T", 
             ylabel="Magnetic Susceptibility Chi/N", 
             title="3D Ising Model Magnetic Susceptibility", 
             legend=false,
             marker=:circle)
savefig(p2_3d, "magnetic_susceptibility_3d_ising.png")
println("3D magnetic susceptibility plot saved as magnetic_susceptibility_3d_ising.png.")

println("\nAll processes completed.")