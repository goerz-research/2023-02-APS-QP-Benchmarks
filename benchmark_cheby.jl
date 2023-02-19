# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Julia 1.8.0
#     language: julia
#     name: julia-1.8
# ---

using StableRNGs
using QuantumControl
using BenchmarkTools
using LinearAlgebra
using FileIO: FileIO
using DataFrames
using CSV
using Plots
using ProgressMeter

using QuantumControlTestUtils.RandomObjects: random_dynamic_generator, random_state_vector

RNG = StableRNG(248221371);

projectdir(path...) = joinpath(@__DIR__, path...)
datadir(path...) = projectdir("data", path...)
mkpath(datadir())

function run_propagation_benchmark(;
    N=10,
    error=[1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10],
    generator_args=Dict(),
    exact_propagation_args=Dict(),
    tune_propagation_args,
    rng=RNG,
    kwargs...
)
    if N isa Vector
        N_values = N
    else
        N_values = [N, ]
    end
    if error isa Vector
        error_values = error
    else
        error_values = error
    end
    data = Dict{Symbol,Vector{Any}}(
        :N => [],
        :error => [],
        :timing => [],
    )
    tlist = collect(range(0, 1000, length=1001))
    progressmeter = Progress(length(N_values) * length(error_values))
    for N ∈ N_values
        H = random_dynamic_generator(N, tlist; rng, generator_args...)
        Ψ₀ = random_state_vector(N; rng)
        Ψ_exact = propagate(Ψ₀, H, tlist; merge(kwargs, exact_propagation_args)...)
        for error in error_values
            tuned_propagation_args = tune_propagation_args(Ψ₀, H, tlist, Ψ_exact, error, kwargs)
            t = @belapsed propagate($Ψ₀, $H, $tlist; $tuned_propagation_args...)
            push!(data[:N], N)
            push!(data[:error], error)
            push!(data[:timing], t)
            next!(progressmeter)
        end
    end
    return DataFrame(data)
end


function tune_cheby(Ψ₀, H, tlist, Ψ_exact, error, kwargs, verbose=false)
    for cheby_coeffs_limit in [
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
        1e-15
    ]
        tuned_args = merge(kwargs, Dict(:cheby_coeffs_limit => cheby_coeffs_limit))
        Ψ = propagate(Ψ₀, H, tlist; tuned_args...)
        if norm(Ψ - Ψ_exact) ≤ error
            if verbose
                println("Tuned cheby: error $error with cheby_coeffs_limit=$cheby_coeffs_limit")
            end
            return tuned_args
        end
    end
    error("Could not tune cheby")
end

load_csv(f) = DataFrame(CSV.File(f))

data_cheby = run_or_load(datadir("benchmark_cheby.csv"); load=load_csv) do
    run_propagation_benchmark(;
        N=[10, 100],
        method=:cheby,
        generator_args=Dict(:exact_spectral_envelope => true),
        exact_propagation_args=Dict(:cheby_coeffs_limit => 1e-15),
        tune_propagation_args=tune_cheby
    )
end

for group in groupby(data_cheby, :N)
    N = group.N[1]
    fig = plot(
        group.error, group.timing; marker=true, label="N=$N",
        xaxis=:log, xlabel="error", ylabel="runtime (seconds)"
    )
    display(fig)
end
