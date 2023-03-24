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
using StaticArrays

using QuantumControlTestUtils.RandomObjects: random_dynamic_generator, random_state_vector

projectdir(path...) = joinpath(@__DIR__, path...)
datadir(path...) = projectdir("data", path...)
mkpath(datadir())

include("includes/benchmark.jl")
include("includes/qdyn.jl")

# ## Sanity check for N=10

# +
function ham_to_static(H)
    N = size(H.ops[1])[1]
    QuantumControl.Generators.Generator(
        [MMatrix{N,N}(op) for op in H.ops],
        H.amplitudes
    )
end

function psi_to_static(Ψ)
    N = length(Ψ)
    MVector{N}(Ψ)
end
# -

N = 10;
cheby_coeffs_limit=1e-12;
tlist = collect(range(0, step=1.0, length=1001));
H_dense = random_dynamic_generator(N, tlist; rng=RNG, exact_spectral_envelope=true);
H = ham_to_static(H_dense)
Ψ₀ = psi_to_static(random_state_vector(N; rng=RNG));

@assert ishermitian(H.ops[1])
@assert ishermitian(H.ops[2])

@benchmark init_prop($Ψ₀, $H, $tlist)

propagator = init_prop(Ψ₀, H, tlist; cheby_coeffs_limit)
@benchmark prop_step!(propagator)

propagator = init_prop(Ψ₀, H, tlist; cheby_coeffs_limit)
Ψ_out = _propagate(propagator, tlist)
propagator = init_prop(Ψ₀, H, tlist; cheby_coeffs_limit)
@time _propagate(propagator, tlist);

benchmark_cheby_with_qdyn(
    Array(Ψ₀), Array(H), tlist;
    rf=datadir("cheby_mstatic_sanity"),
    cheby_coeffs_limit
)

Ψ_out_qdyn = pyconvert(Vector{ComplexF64},
    qdyn.io.read_psi_amplitudes(
        datadir("cheby_mstatic_sanity", "final_state.dat.1"),
        N;
        normalize=false
    )
);

norm(Ψ_out_qdyn - Ψ_out)

# ## Benchmark

load_csv(f) = DataFrame(CSV.File(f))

PRECISION =  [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]

data_static = run_or_load(datadir("benchmark_mstatic_cheby.csv"); load=load_csv, force=false) do
    run_propagation_benchmark(;
        N=10,
        precision=PRECISION,
        method=:cheby,
        generator_args=Dict(:exact_spectral_envelope => true, :hermitian => true),
        convert_generator=ham_to_static,
        convert_state=psi_to_static,
        exact_propagation_args=Dict(:cheby_coeffs_limit => 1e-15),
        tune_propagation_args=tune_cheby,
    )
end

data_dense = groupby(load_csv(datadir("benchmark_dense_cheby.csv")), :N)[3]

fig = plot(
    data_dense.precision, data_dense.timing; marker=true, label="Julia",
    xaxis=:log,
    xlabel=raw"precision (absolute error)",
    ylabel="runtime (seconds)",
    xticks=PRECISION, title="Cheby – Hilbert space dimension N=$N",
)
plot!(
    fig, data_static.precision, data_static.timing; marker=true, label="Julia (mstatic)",
    xaxis=:log, xticks=PRECISION
)
plot!(
    fig, data_static.precision, data_dense.QDYN_ifort; shape=:utriangle, label="Fortran (ifort)",
    xaxis=:log, xticks=PRECISION, ylim=(0, 0.01)
)


