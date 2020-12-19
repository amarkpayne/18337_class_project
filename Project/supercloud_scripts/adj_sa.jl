import Pkg
Pkg.activate("class_project_env")
ENV["CONDA_JL_HOME"] = "/home/gridsan/ampayne/.conda/envs/rms_env"
ENV["PYTHON"] = "/home/gridsan/ampayne/.conda/envs/rms_env/bin/python"

using BenchmarkTools
using DiffEqBase
using DiffEqSensitivity
using PyPlot
using ReactionMechanismSimulator
using Sundials

import ForwardDiff

# Parallel Settings
using Base.Threads
nthreads = Threads.nthreads()
println("Using $nthreads threads")

# Small Model
#phaseDict = readinput("chem.rms")
#phaseDict["phase"]["Species"]
#ig = IdealGas(phaseDict["phase"]["Species"], phaseDict["phase"]["Reactions"], name="gas")
#ic = Dict(["T"=>1350.0,"P"=>1e5,"ethane"=>1.0])
#domain, y0, all_params = ConstantTPDomain(phase=ig,initialconds=ic;sensitivity=false)
#tf = 1e-3

# Medium Model
phaseDict = readinput("medium_model/medium_model.rms")
phaseDict["phase"]["Species"]
ig = IdealGas(phaseDict["phase"]["Species"], phaseDict["phase"]["Reactions"], name="gas")
ic = Dict(["T"=>1000,"P"=>2e6,"n-octane"=>1.0])
domain, y0, all_params = ConstantTPDomain(phase=ig,initialconds=ic;sensitivity=false)
tf = 2e-1

# Setup parallel problem
spcs_list = [s.name for s in phaseDict["phase"]["Species"]]
spcs_list = spcs_list[5:end]  # First 4 species are in the actual model

# Calculate these outside of timed functions. We will time the first two steps but not the third for fair comparison
adjoint_react = Reactor(domain,y0,(0.0, tf); p=all_params)
adjoint_sol = solve(adjoint_react.ode, CVODE_BDF(), abstol=1e-20,reltol=1e-12)
adjoint_bsol = Simulation(adjoint_sol, domain)  # The extra time involved with this step is not a fair comparison to forward SA

function adjoint_parallelization_proceedure(rms_domain, u0, ts, params, spcs_list, atol, rtol)
    # Solve for solution variables first
    adjoint_react = Reactor(rms_domain,u0,ts; p=params)
    adjoint_sol = solve(adjoint_react.ode, CVODE_BDF(), abstol=1e-20,reltol=1e-12)

    # Loop over species to calculate all rows of the SA matrix
    solution_dictionary = Dict()

    @threads for spc in spcs_list[1:16]  # Only consider 64 species for medium model
        adj_sa = getadjointsensitivities(adjoint_bsol, spc, CVODE_BDF();
                                         sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),abstol=atol,reltol=rtol)
        solution_dictionary[spc] = adj_sa
    end
    return solution_dictionary
end


@btime adjoint_parallelization_proceedure(domain, y0, (0.0, tf), all_params, spcs_list, 1e-6, 1e-6)
adjoint_parallelization_proceedure(domain, y0, (0.0, tf), all_params, spcs_list, 1e-6, 1e-6)
