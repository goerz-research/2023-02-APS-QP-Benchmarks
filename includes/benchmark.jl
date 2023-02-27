using BenchmarkTools
using QuantumControl: propagate
using QuantumControlTestUtils.RandomObjects: random_dynamic_generator, random_state_vector
using ProgressMeter
using DataFrames


RNG = StableRNG(248221371);


function _propagate(propagator, tlist)
    @assert propagator.t == 0.0
    for interval = 1:length(tlist)-1
        Ψ = prop_step!(propagator)
    end
    return propagator.state
end


function run_propagation_benchmark(;
    N,
    precision,
    generator_args = Dict(),
    exact_propagation_args = Dict(),
    tune_propagation_args,
    rng = RNG,
    callback = nothing,
    callback_column = nothing,
    convert_generator = (H -> H),
    convert_state = (Ψ -> Ψ),
    dt = 1.0,
    nt = 1001,
    tune_benchmark_seconds = (args...) -> 5.0,
    warn_if_runtime_exceeds = 300.0,
    stop_if_runtime_exceeds = 1800.0,
    kwargs...,
)
    if N isa Vector
        N_values::Vector{Int64} = N
    else
        N_values = Int64[N]
    end
    if precision isa Vector
        precision_values::Vector{Float64} = precision
    else
        precision_values = Float64[precision]
    end
    data = Dict{Symbol,Vector{Any}}(:N => [], :precision => [], :timing => [])
    if !isnothing(callback) && !isnothing(callback_column)
        if callback_column isa Tuple
            for col_name in callback_column
                data[col_name] = []
            end
        else
            data[callback_column] = []
        end
    end
    tlist = collect(range(0, step=dt, length=nt))
    progressmeter = Progress(length(N_values) * length(precision_values))
    for N ∈ N_values
        H = convert_generator(random_dynamic_generator(N, tlist; rng, generator_args...))
        Ψ₀ = convert_state(random_state_vector(N; rng))
        res_exact = @timed propagate(Ψ₀, H, tlist; merge(kwargs, exact_propagation_args)...)
        if res_exact.time > warn_if_runtime_exceeds
            @warn "Propagation for N=$N takes $(res_exact.time) seconds"
        end
        if res_exact.time > stop_if_runtime_exceeds
            @error "Propagation for N=$N takes $(res_exact.time) seconds. Stopping."
            break
        end
        Ψ_exact = res_exact.value
        for precision in precision_values
            tuned_propagation_args =
                tune_propagation_args(Ψ₀, H, tlist, Ψ_exact, precision, kwargs)
            seconds = tune_benchmark_seconds(N, precision)
            t = @belapsed _propagate(propagator, $tlist) setup=(
                    propagator = init_prop($Ψ₀, $H, $tlist; $tuned_propagation_args...)
                ) evals = 1 seconds = seconds
            if !isnothing(callback)
                cb_data = callback(
                    Ψ₀, H, tlist;
                    N, precision, seconds, tuned_propagation_args...
                )
                if !isnothing(callback_column)
                    if cb_data isa Tuple
                        for (val, col_name) in zip(cb_data, callback_column)
                            push!(data[col_name], val)
                        end
                    else
                        push!(data[callback_column], cb_data)
                    end
                end
            end
            push!(data[:N], N)
            push!(data[:precision], precision)
            push!(data[:timing], t)
            next!(progressmeter)
        end
    end
    return DataFrame(data)
end


function tune_cheby(
    Ψ₀,
    H,
    tlist,
    Ψ_exact,
    precision::Float64,
    kwargs,
    verbose = false;
    cheby_coeffs_limit_candidates::Vector{Float64} = [
        1e-2,
        1e-3,
        1e-4,
        1e-5,
        1e-6,
        1e-7,
        1e-8,
        1e-9,
        1e-10,
        1e-11,
        1e-12,
        1e-13,
        1e-14,
        1e-15,
    ],
    nthreads = Threads.nthreads(),
)
    i::Int = 0
    N = length(cheby_coeffs_limit_candidates)
    while i < N
        tasks = []
        # determine error for a chunk of candidates, in parallel
        for thread_id = 1:nthreads
            i += 1
            if i <= N
                cheby_coeffs_limit::Float64 = cheby_coeffs_limit_candidates[i]
            else
                break
            end
            task = Threads.@spawn begin
                tuned_args = merge(kwargs, Dict(:cheby_coeffs_limit => $cheby_coeffs_limit))
                Ψ = propagate(Ψ₀, H, tlist; tuned_args...)
                task_error = norm(Ψ - Ψ_exact)
                ($cheby_coeffs_limit, task_error)
            end
            push!(tasks, task)
        end
        # check results to see if a candidate was below the error threshold
        results = [fetch(task) for task in tasks]
        for (cheby_coeffs_limit, task_error) in results
            if task_error ≤ precision
                if verbose
                    println(
                        "Tuned cheby: precision $precision with cheby_coeffs_limit=$cheby_coeffs_limit",
                    )
                end
                tuned_args = merge(kwargs, Dict(:cheby_coeffs_limit => cheby_coeffs_limit))
                return tuned_args
            end
        end
    end
    error("Could not tune cheby")
end
