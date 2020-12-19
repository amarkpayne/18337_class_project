# Parallelization strategy 2: Forward SA threaded (with Jacobian)

```julia
function ThreadedReactor(domain::T,y0::Array{W,1},tspan::Tuple, interfaces::Z=[];p::X=DiffEqBase.NullParameters(), p_batch_size=8) where {T<:AbstractDomain,W<:Real,Z<:AbstractArray,X}
    dydt(dy::X,y::T,p::V,t::Q) where {X,T,Q<:Real,V} = dydtreactor!(dy,y,t,domain,interfaces,p=p)
    jacp!(J::Q2,y::T,p::V,t::Q) where {Q2,T,Q<:Real,V} = jacobianp!(J,y,p,t,domain,interfaces,nothing)

    function jacp_batch!(J,y,p,t, i)
        if i == 1


    # Create list of Reactor objects (containing ODEForwardSensitivityProblem objects) for each batch of parameters
    reactor_list = []
    n_batches = Int64(round(length(p)/p_batch_size, RoundUp))
    ## Enclose the "fixed" parameters for each batch
    odefcn = ODEFunction((dy, y, batch_p, t) -> dydt(dy, y, [batch_p; p[p_batch_size+1:end]], t);
                         paramjac=(J, y, batch_p, t) -> jacp!(J, y, [batch_p; p[p_batch_size+1:end]], t)[:, 1:p_batch_size]
                         )
    ode = ODEForwardSensitivityProblem(odefcn,y0,tspan,p[1:p_batch_size])
    recsolver = Sundials.CVODE_BDF(linear_solver=:GMRES)
    push!(reactor_list, Reactor(domain,ode,recsolver,true))

    for i in 2:n_batches-1
        odefcn = ODEFunction((dy, y, batch_p, t) -> dydt(dy, y, [p[1:(i-1)*p_batch_size];batch_p; p[i*p_batch_size+1:end]], t),
                             paramjac=(J, y, batch_p, t) -> jacp!(J, y, [p[1:(i-1)*p_batch_size];batch_p; p[i*p_batch_size+1:end]], t)[:, (i-1)*p_batch_size+1:i*p_batch_size]
                             )
        ode = ODEForwardSensitivityProblem(odefcn,y0,tspan,p[(i-1)*p_batch_size+1:i*p_batch_size])
        recsolver = Sundials.CVODE_BDF(linear_solver=:GMRES)
        push!(reactor_list, Reactor(domain,ode,recsolver,true))
    end

    odefcn = ODEFunction((dy, y, batch_p, t) -> dydt(dy, y, [p[1:(n_batches-1)*p_batch_size]; batch_p], t),
                         paramjac=(J, y, batch_p, t) -> jacp!(J, y, [p[1:(n_batches-1)*p_batch_size]; batch_p], t)[:, (n_batches-1)*p_batch_size+1:end]
    )
    ode = ODEForwardSensitivityProblem(odefcn,y0,tspan,p[(n_batches-1)*p_batch_size+1:end])
    recsolver = Sundials.CVODE_BDF(linear_solver=:GMRES)
    push!(reactor_list, Reactor(domain,ode,recsolver,true))

    return reactor_list
end

threaded_react_list = ThreadedReactor(domain,y0,(0.0,1e-3); p=all_params, p_batch_size=4)
threaded_ode1 = threaded_react_list[1]
threaded_sol = solve(threaded_ode1.ode, CVODE_BDF(), abstol=1e-20,reltol=1e-12)

# Use this proceedure to test out the small model
function threaded_proceedure(rms_domain, u0, ts, params; batch_size=8)
    threaded_fd_react_list = ThreadedReactor(rms_domain,u0,ts; p=params, p_batch_size=batch_size)
    solution_dictionary = Dict()

    @threads for i in 1:length(threaded_fd_react_list)
      sol = solve(threaded_fd_react_list[i].ode, CVODE_BDF(), abstol=1e-20,reltol=1e-12)
      solution_dictionary[i] = sol
    end
    return solution_dictionary
end

# Use this proceedure to test out the medium model, as the timing for all 1000+ parameters takes too long
function threaded_proceedure_limited(rms_domain, u0, ts, params; batch_size=8)
    threaded_fd_react_list = ThreadedReactor(rms_domain,u0,ts; p=params, p_batch_size=batch_size)
    solution_dictionary = Dict()

    # Do 64 parameters in total, so adjust number of iteration to match based on batch size
    last_batch = Int64(64/batch_size)
    @threads for i in 1:last_batch
      sol = solve(threaded_fd_react_list[i].ode, CVODE_BDF(), abstol=1e-20,reltol=1e-12)
      solution_dictionary[i] = sol
    end
    return solution_dictionary
end

b_size = 1
n_spcs = length(y0)
tf = 1e-3
threaded_proceedure(domain, y0, (0.0, tf), all_params; batch_size=b_size)
@btime threaded_fd_proceedure(domain, y0, (0.0, tf), all_params; batch_size=b_size)
# Results on Personal laptop: 2 threads, 16 GB RAM
# b_size 32:
# b_size 16:
# b_size 8:
# b_size 4:
# b_size 2:
# b_size 1:


# b_size 32: 26.815 s (74588670 allocations: 31.74 GiB)
# b_size 16: 12.261 s (44988400 allocations: 18.54 GiB)
# b_size 8: 7.811 s (29134813 allocations: 11.58 GiB)
# b_size 4: 5.886 s (23357265 allocations: 8.59 GiB)
# b_size 2: 5.939 s (24291380 allocations: 7.93 GiB)
# b_size 1: 7.570 s (31433712 allocations: 8.98 GiB)


# Verify that the data matches
t_sens = 1e-4
threaded_fd_sol_dict = threaded_fd_proceedure(domain, y0, (0.0, 1e-3), all_params; batch_size=b_size)
isapprox(threaded_fd_sol_dict[1](t_sens), baseline_sol(t_sens)[1:n_spcs+b_size*n_spcs])
isapprox(threaded_fd_sol_dict[2](t_sens)[1:n_spcs], baseline_sol(t_sens)[1:n_spcs])
isapprox(threaded_fd_sol_dict[2](t_sens)[n_spcs+1:end], baseline_sol(t_sens)[n_spcs+1*b_size*n_spcs+1:n_spcs+2*b_size*n_spcs])
isapprox(threaded_fd_sol_dict[3](t_sens)[28:end], baseline_sol(t_sens)[n_spcs+2*b_size*n_spcs+1:n_spcs+3*b_size*n_spcs])
n_batches = length(threaded_fd_sol_dict)
final_sens = threaded_fd_sol_dict[n_batches](t_sens)[n_spcs+1:end]
isapprox(final_sens, baseline_sol(t_sens)[end-length(final_sens)+1:end])
# ALl checks passed, regardless of model or batch size
