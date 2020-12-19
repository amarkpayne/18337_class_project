import Pkg
Pkg.activate("class_project_env")
ENV["CONDA_JL_HOME"] = "/home/ampayne/anaconda2/envs/rms_env"
ENV["PYTHON"] = "/home/ampayne/anaconda2/envs/rms_env/bin/python"
ENV["PYTHONPATH"]

using ReactionMechanismSimulator
using DiffEqBase
using Sundials
using DiffEqSensitivity
using PyPlot

phaseDict = readinput("chem.inp";
              spcdict="species_dictionary.txt")

phaseDict["phase"]["Species"]

ig = IdealGas(phaseDict["phase"]["Species"], phaseDict["phase"]["Reactions"], name="gas")

ic = Dict(["T"=>1350.0,"P"=>1e5,"ethane"=>1.0])

domain, y0, all_params = ConstantTPDomain(phase=ig,initialconds=ic;sensitivity=true)



react = Reactor(domain,y0,(0.0,1.0))

sol = solve(react.ode, CVODE_BDF(), abstol=1e-20,reltol=1e-12)

sol(0.0)

function ParallelReactor(domain::T,y0::Array{W,1},tspan::Tuple,interfaces::Z=[];p::X=DiffEqBase.NullParameters(),forwardsensitivities=false) where {T<:AbstractDomain,W<:Real,Z<:AbstractArray,X}
    dydt(dy::X,y::T,p::V,t::Q) where {X,T,Q<:Real,V} = dydtreactor!(dy,y,t,domain,interfaces,p=p)
    jacy!(J::Q2,y::T,p::V,t::Q) where {Q2,T,Q<:Real,V} = jacobiany!(J,y,p,t,domain,interfaces,nothing)
    jacyforwarddiff!(J::Q2,y::T,p::V,t::Q) where {Q2,T,Q<:Real,V} = jacobianyforwarddiff!(J,y,p,t,domain,interfaces,nothing)
    jacp!(J::Q2,y::T,p::V,t::Q) where {Q2,T,Q<:Real,V} = jacobianp!(J,y,p,t,domain,interfaces,nothing)
    jacpforwarddiff!(J::Q2,y::T,p::V,t::Q) where {Q2,T,Q<:Real,V} = jacobianpforwarddiff!(J,y,p,t,domain,interfaces,nothing)

    if domain isa Union{ConstantTPDomain,ConstantVDomain,ConstantPDomain,ParametrizedTPDomain,ParametrizedVDomain,ParametrizedPDomain,ConstantTVDomain,ParametrizedTConstantVDomain,ConstantTADomain}
        if !forwardsensitivities
            odefcn = ODEFunction(dydt;jac=jacy!,paramjac=jacp!)
        else
            odefcn = ODEFunction(dydt;paramjac=jacp!)
        end
    else
        odefcn = ODEFunction(dydt;jac=jacyforwarddiff!,paramjac=jacpforwarddiff!)
    end
    if forwardsensitivities
        ode = ODEForwardSensitivityProblem(odefcn,y0,tspan,p)
        recsolver = Sundials.CVODE_BDF(linear_solver=:GMRES)
    else
        ode = ODEProblem(odefcn,y0,tspan,p)
        recsolver  = Sundials.CVODE_BDF()
    end
    return Reactor(domain,ode,recsolver,forwardsensitivities), dydt
end

react, my_dydt = ParallelReactor(domain,y0,(0.0,1.0))

my_dys = zeros(27)
my_ys = zeros(27)
my_ys[5] = 0.1

a = my_dydt(my_dys, y0, all_params, 0.01)

function dydt_enclosed(dy,y,p,t)
    params = all_params
    params = zeros(96)
    return my_dydt(dy, y, params, t)
end

b = dydt_enclosed(my_dys, y0, 5e100, 0.01)

isapprox(a, b)
