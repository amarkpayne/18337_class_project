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

# Medium Model
phaseDict = readinput("medium_model/medium_model.rms")
phaseDict["phase"]["Species"]
ig = IdealGas(phaseDict["phase"]["Species"], phaseDict["phase"]["Reactions"], name="gas")
ic = Dict(["T"=>1000,"P"=>2e6,"n-octane"=>1.0])
domain, y0, all_params = ConstantTPDomain(phase=ig,initialconds=ic;sensitivity=false)


function ThreadedSA(domain::T,y0::Array{W,1},tspan::Tuple, fsol, interfaces::Z=[];p::X=DiffEqBase.NullParameters()) where {T<:AbstractDomain,W<:Real,Z<:AbstractArray,X}
    # Previously implement in RMS
    jacy!(J::Q2,y::T,p::V,t::Q) where {Q2,T,Q<:Real,V} = jacobiany!(J,y,p,t,domain,interfaces,nothing)
    jacp!(J::Q2,y::T,p::V,t::Q) where {Q2,T,Q<:Real,V} = jacobianp!(J,y,p,t,domain,interfaces,nothing)

    # My implementation
    function dsdt(p_index)
        function dsdt!(ds, s, local_params, t)
            jy = zeros(length(y0), length(y0))
            jp = zeros(length(y0), length(p))
            jacy!(jy, fsol(t), p, t)
            jacp!(jp, fsol(t), p, t)
            c = jp[:, p_index]
            ds .= jy*s .+ c
        end
        return dsdt!
    end

    # Create list of ODEProblems for each batch of parameters
    sa_list = []

    for i in 1:length(p)
        odefcn = ODEFunction(dsdt(i))
        prob = ODEProblem(odefcn, zeros(length(y0)),tspan,0)
        push!(sa_list, prob)
    end
    return sa_list
end


function threaded_sa_proceedure(rms_domain, u0, ts, params)
    f_react = Reactor(rms_domain,u0,ts; p=params)
    f_sol = solve(f_react.ode, CVODE_BDF(), abstol=1e-20,reltol=1e-12)

    sa_list = ThreadedSA(rms_domain, u0, ts, f_sol; p=params)

    # Parallelize the SA calculations
    solution_dictionary = Dict()

    @threads for i in 1:256  # Only do 256 parameters for medium model
        s = solve(sa_list[i], CVODE_BDF(), abstol=1e-20,reltol=1e-12)
        solution_dictionary[i] = s
    end

    return solution_dictionary
end

@btime threaded_sa_proceedure(domain, y0, (0.0, 1e-3), all_params)
