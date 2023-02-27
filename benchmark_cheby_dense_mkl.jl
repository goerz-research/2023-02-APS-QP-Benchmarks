# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
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
using CSV
using Plots
using DrWatson: savename, parse_savename

using MKL
BLAS.get_config()

using QuantumControlTestUtils.RandomObjects: random_dynamic_generator, random_state_vector

projectdir(path...) = joinpath(@__DIR__, path...)
datadir(path...) = projectdir("data", path...)
mkpath(datadir())

include("includes/benchmark.jl")
include("includes/qdyn.jl")

# ## Sanity check for N=1000

N = 1000;
cheby_coeffs_limit=1e-12;
tlist = collect(range(0, step=1.0, length=1001));
H = random_dynamic_generator(N, tlist; rng=RNG, exact_spectral_envelope=true);
Ψ₀ = random_state_vector(N; rng=RNG);

@assert ishermitian(H.ops[1])
@assert ishermitian(H.ops[2])

@benchmark init_prop($Ψ₀, $H, $tlist)

propagator = init_prop(Ψ₀, H, tlist; cheby_coeffs_limit)
@benchmark prop_step!(propagator)

propagator = init_prop(Ψ₀, H, tlist; cheby_coeffs_limit)
Ψ_out = _propagate(propagator, tlist)
propagator = init_prop(Ψ₀, H, tlist; cheby_coeffs_limit)
@time _propagate(propagator, tlist);

# ## Benchmark

load_csv(f) = DataFrame(CSV.File(f))

PRECISION =  [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
BENCHMARK_SECONDS = Dict(
    10 => 5.0,
    100 =>  5.0,
    1000 => 300.0,
)

data_cheby = run_or_load(datadir("benchmark_dense_mkl_cheby.csv"); load=load_csv, force=false) do
    run_propagation_benchmark(;
        N=[1_000, 100, 10],  # slowest ones first
        precision=PRECISION,
        method=:cheby,
        generator_args=Dict(:exact_spectral_envelope => true, :hermitian => true),
        exact_propagation_args=Dict(:cheby_coeffs_limit => 1e-15),
        tune_benchmark_seconds=((N, precision) -> BENCHMARK_SECONDS[N]),
        tune_propagation_args=tune_cheby,
    )
end

data_dense = load_csv(datadir("benchmark_dense_cheby.csv"))

data_static = filter(:N => _N->(_N == 10), load_csv(datadir("benchmark_mstatic_cheby.csv")))

for group in groupby(data_dense, :N)
    N = group.N[1]
    fig = plot(
        group.precision, group.timing; marker=true, label="Julia",
        xaxis=:log,
        xlabel=raw"precision (absolute error)",
        ylabel="runtime (seconds)",
        xticks=PRECISION, title="Cheby (dense) – Hilbert space dimension N=$N",
    )
    group_mkl = filter(:N => _N->(_N == N), data_cheby)
    plot!(
        fig, group_mkl.precision, group_mkl.timing; marker=true,
        label="Julia (MKL)"
    )
    plot!(
        fig, group.precision, group.QDYN_ifort; shape=:utriangle,
        label="Fortran (ifort)"
    )
    #plot!(
    #    fig, group.precision, group.QDYN_ifort_fast; shape=:utriangle,
    #    label="Fortran (ifort-fast)"
    #)
    plot!(
        fig, group.precision, group.QDYN_gfortran; shape=:utriangle,
        label="Fortran (gfortran)"
    )
    if N == 10
        plot!(
            fig, data_static.precision, data_static.timing; shape=:rect,
            label="Julia (static)"
        )
    end
    y_limits = ylims(fig)
    plot!(fig; ylims=(0, y_limits[2]))
    display(fig)
end


