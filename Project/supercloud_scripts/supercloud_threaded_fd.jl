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
#domain, y0, all_params = ConstantTPDomain(phase=ig,initialconds=ic;sensitivity=true)

# Medium Model
phaseDict = readinput("medium_model/medium_model.rms")
phaseDict["phase"]["Species"]
ig = IdealGas(phaseDict["phase"]["Species"], phaseDict["phase"]["Reactions"], name="gas")
ic = Dict(["T"=>1000,"P"=>2e6,"n-octane"=>1.0])
domain, y0, all_params = ConstantTPDomain(phase=ig,initialconds=ic;sensitivity=true)



#function BaselineReactor(domain::T,y0::Array{W,1},tspan::Tuple,interfaces::Z=[];p::X=DiffEqBase.NullParameters(),forwardsensitivities=false) where {T<:AbstractDomain,W<:Real,Z<:AbstractArray,X}
#    # Native RMS Code
#    dydt(dy::X,y::T,p::V,t::Q) where {X,T,Q<:Real,V} = dydtreactor!(dy,y,t,domain,interfaces,p=p)
#
#
#    # Baseline setup of forward SA problem without giving the jacobian
#    odefcn = ODEFunction(dydt)
#
#
#    ode = ODEForwardSensitivityProblem(odefcn,y0,tspan,p)
#    recsolver = Sundials.CVODE_BDF(linear_solver=:GMRES)
#
#    return Reactor(domain,ode,recsolver,forwardsensitivities)
#end

#baseline_react = BaselineReactor(domain,y0,(0.0,1e-3); forwardsensitivities=true, p=all_params)
#baseline_sol = solve(baseline_react.ode, CVODE_BDF(), abstol=1e-20,reltol=1e-12)

#function baseline_proceedure(rms_domain, u0, ts, params)
#    baseline_react = BaselineReactor(rms_domain,u0,ts; forwardsensitivities=true, p=params)
#    baseline_sol = solve(baseline_react.ode, CVODE_BDF(), abstol=1e-20,reltol=1e-12)
#end

#println("Profiling baseline")
#@btime baseline_proceedure(domain, y0, (0.0, 1e-3), all_params)




function ThreadedFDReactor(domain::T,y0::Array{W,1},tspan::Tuple, interfaces::Z=[];p::X=DiffEqBase.NullParameters(), p_batch_size=8) where {T<:AbstractDomain,W<:Real,Z<:AbstractArray,X}
    # Native RMS Code
    dydt(dy::X,y::T,p::V,t::Q) where {X,T,Q<:Real,V} = dydtreactor!(dy,y,t,domain,interfaces,p=p)

    # Create list of Reactor objects (containing ODEForwardSensitivityProblem objects) for each batch of parameters
    reactor_list = []
    n_batches = Int64(round(length(p)/p_batch_size, RoundUp))
    ## Enclose the "fixed" parameters for each batch
    odefcn = ODEFunction((dy, y, batch_p, t) -> dydt(dy, y, [batch_p; p[p_batch_size+1:end]], t))
    ode = ODEForwardSensitivityProblem(odefcn,y0,tspan,p[1:p_batch_size])
    recsolver = Sundials.CVODE_BDF(linear_solver=:GMRES)
    push!(reactor_list, Reactor(domain,ode,recsolver,true))

    for i in 2:n_batches-1
        odefcn = ODEFunction((dy, y, batch_p, t) -> dydt(dy, y, [p[1:(i-1)*p_batch_size];batch_p; p[i*p_batch_size+1:end]], t))
        ode = ODEForwardSensitivityProblem(odefcn,y0,tspan,p[(i-1)*p_batch_size+1:i*p_batch_size])
        recsolver = Sundials.CVODE_BDF(linear_solver=:GMRES)
        push!(reactor_list, Reactor(domain,ode,recsolver,true))
    end

    odefcn = ODEFunction((dy, y, batch_p, t) -> dydt(dy, y, [p[1:(n_batches-1)*p_batch_size]; batch_p], t))
    ode = ODEForwardSensitivityProblem(odefcn,y0,tspan,p[(n_batches-1)*p_batch_size+1:end])
    recsolver = Sundials.CVODE_BDF(linear_solver=:GMRES)
    push!(reactor_list, Reactor(domain,ode,recsolver,true))

    return reactor_list
end


function threaded_fd_proceedure(rms_domain, u0, ts, params; batch_size=8)
    threaded_fd_react_list = ThreadedFDReactor(rms_domain,u0,ts; p=params, p_batch_size=batch_size)
    solution_dictionary = Dict()

    @threads for i in 1:length(threaded_fd_react_list)
      sol = solve(threaded_fd_react_list[i].ode, CVODE_BDF(), abstol=1e-20,reltol=1e-12)
      solution_dictionary[i] = sol
    end
    return solution_dictionary
end

function threaded_fd_proceedure_limited(rms_domain, u0, ts, params; batch_size=8)
    threaded_fd_react_list = ThreadedFDReactor(rms_domain,u0,ts; p=params, p_batch_size=batch_size)
    solution_dictionary = Dict()

    last_batch = Int64(64/batch_size)
    @threads for i in 1:last_batch
      sol = solve(threaded_fd_react_list[i].ode, CVODE_BDF(), abstol=1e-20,reltol=1e-12)
      solution_dictionary[i] = sol
    end
    return solution_dictionary
end

tf = 2e-1

b_size = 2
println("Profiling threaded proceedure for $nthreads threads and a b_size of $b_size")
@btime threaded_fd_proceedure_limited(domain, y0, (0.0, tf), all_params; batch_size=b_size)

b_size = 4
println("Profiling threaded proceedure for $nthreads threads and a b_size of $b_size")
@btime threaded_fd_proceedure_limited(domain, y0, (0.0, tf), all_params; batch_size=b_size)

b_size = 8
println("Profiling threaded proceedure for $nthreads threads and a b_size of $b_size")
@btime threaded_fd_proceedure_limited(domain, y0, (0.0, tf), all_params; batch_size=b_size)

b_size = 16
println("Profiling threaded proceedure for $nthreads threads and a b_size of $b_size")
@btime threaded_fd_proceedure_limited(domain, y0, (0.0, tf), all_params; batch_size=b_size)




# Results on Personal laptop: 2 threads, 16 GB RAM
# b_size 32: 26.815 s (74588670 allocations: 31.74 GiB)
# b_size 16: 12.261 s (44988400 allocations: 18.54 GiB)
# b_size 8: 7.811 s (29134813 allocations: 11.58 GiB)
# b_size 4: 5.886 s (23357265 allocations: 8.59 GiB)
# b_size 2: 5.939 s (24291380 allocations: 7.93 GiB)
# b_size 1: 7.570 s (31433712 allocations: 8.98 GiB)


# Verify that the data matches
#t_sens = 1e-4
#threaded_fd_sol_dict = threaded_fd_proceedure(domain, y0, (0.0, 1e-3), all_params; batch_size=b_size)
#println(isapprox(threaded_fd_sol_dict[1](t_sens), baseline_sol(t_sens)[1:n_spcs+b_size*n_spcs]))
#println(isapprox(threaded_fd_sol_dict[2](t_sens)[1:n_spcs], baseline_sol(t_sens)[1:n_spcs]))
#println(isapprox(threaded_fd_sol_dict[2](t_sens)[n_spcs+1:end], baseline_sol(t_sens)[n_spcs+1*b_size*n_spcs+1:n_spcs+2*b_size*n_spcs]))
#println(isapprox(threaded_fd_sol_dict[3](t_sens)[28:end], baseline_sol(t_sens)[n_spcs+2*b_size*n_spcs+1:n_spcs+3*b_size*n_spcs]))
#n_batches = length(threaded_fd_sol_dict)
#final_sens = threaded_fd_sol_dict[n_batches](t_sens)[n_spcs+1:end]
#println(isapprox(final_sens, baseline_sol(t_sens)[end-length(final_sens)+1:end]))
