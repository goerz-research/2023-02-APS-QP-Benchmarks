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
using CUDA

CUDA.versioninfo()

using QuantumControlTestUtils.RandomObjects: random_dynamic_generator, random_state_vector

projectdir(path...) = joinpath(@__DIR__, path...)
datadir(path...) = projectdir("data", path...)
mkpath(datadir())

include("includes/benchmark.jl")
include("includes/qdyn.jl")

# ## Sanity check for N=1000

# +
function ham_to_gpu(H)
    QuantumControl.Generators.Generator(
        [CuArray(op) for op in H.ops],
        #[CuArray(ampl) for ampl in H.amplitudes]
        H.amplitudes
    )
end

function ham_from_gpu(H)
    QuantumControl.Generators.Generator(
        [Array(op) for op in H.ops],
        #[Array(ampl) for ampl in H.amplitudes]
        H.amplitudes
    )
end

function op_to_gpu(H)
    QuantumControl.Generators.Operator(
        [CuArray(op) for op in H.ops],
        H.coeffs
    )
end

function op_from_gpu(H)
    QuantumControl.Generators.Operator(
        [Array(op) for op in H.ops],
        H.coeffs
    )
end

psi_to_gpu(Ψ) = CuArray(Ψ)

psi_from_gpu(Ψ) = Array(Ψ)
# -

N = 1000;
cheby_coeffs_limit=1e-12;
tlist = collect(range(0, step=1.0, length=1001));
H_dense = random_dynamic_generator(N, tlist; rng=RNG, exact_spectral_envelope=true);
H = ham_to_gpu(H_dense)
Ψ₀_dense = random_state_vector(N; rng=RNG)
Ψ₀ = psi_to_gpu(Ψ₀_dense);

norm(Ψ₀_dense - Array(Ψ₀))

@assert ishermitian(H.ops[1])
@assert ishermitian(H.ops[2])

import QuantumControl: init_prop

# + active=""
# import QuantumControl.Controls: get_controls, discretize, discretize_on_midpoints

# + active=""
# get_controls(ampl::AbstractArray) = (ampl,)

# + active=""
# function discretize(control::CuArray, tlist)
#     if length(control) == length(tlist)
#         return control
#     elseif length(control) == length(tlist) - 1
#         # convert `control` on intervals to values on `tlist`
#         # cf. pulse_onto_tlist in Python krotov package
#         vals = CuArray(zeros(eltype(control), length(control) + 1))
#         vals[1] = control[1]
#         vals[end] = control[end]
#         for i = 2:length(vals)-1
#             vals[i] = 0.5 * (control[i-1] + control[i])
#         end
#         return vals
#     else
#         throw(ArgumentError("control array must be defined on intervals of tlist"))
#     end
# end
#
# function discretize_on_midpoints(control::CuArray, tlist)
#     if length(control) == length(tlist) - 1
#         return control
#     elseif length(control) == length(tlist)
#         vals = CuArray(zeros(eltype(control), length(control) - 1))
#         vals[1] = control[1]
#         vals[end] = control[end]
#         for i = 2:length(vals)-1
#             vals[i] = 2 * control[i] - vals[i-1]
#         end
#         return vals
#     else
#         throw(ArgumentError("control array must be defined on the points of tlist"))
#     end
# end
# -

function init_prop(state::ST, H::GT, tlist, method::Val{:cheby}; kwargs...) where {ST<:CuArray,GT}
    propagator = init_prop(psi_from_gpu(state), ham_from_gpu(H), tlist, method; kwargs...)
    gpu_genop = QuantumControl.Controls.evaluate(H, tlist, 1)
    OT = typeof(gpu_genop)
    wrk = QuantumControl.QuantumPropagators.Cheby.ChebyWrk(
        state, propagator.wrk.Δ, propagator.wrk.E_min, propagator.wrk.dt;
        limit=propagator.wrk.limit  
    )
    return QuantumControl.QuantumPropagators.ChebyPropagator{GT,OT,ST}(
        H,
        state,
        propagator.t,
        propagator.n,
        tlist,
        propagator.parameters,
        propagator.controls,
        propagator.control_ranges,
        gpu_genop,
        wrk,
        propagator.backward,
        propagator.inplace,
        propagator.specrange_method,
        propagator.specrange_buffer,
        propagator.check_normalization,
        propagator.specrange_options,
    )
end

dense_propagator = init_prop(Ψ₀_dense, H_dense, tlist, Val(:cheby)); # DEBUG

propagator = init_prop(Ψ₀, H, tlist, Val(:cheby));

X = QuantumControl.QuantumPropagators._pwc_set_genop!(dense_propagator, 1)

Y = op_from_gpu(QuantumControl.QuantumPropagators._pwc_set_genop!(propagator, 1))

norm(X.ops[1] - Y.ops[1]), norm(X.ops[2] - Y.ops[2]), norm(X.coeffs .- Y.coeffs) 

copyto!(propagator.wrk.v0, Ψ₀)
Ψ = psi_from_gpu(propagator.wrk.v0);

norm(Array(Ψ₀) - Ψ₀_dense)

norm(Ψ - Ψ₀_dense)

Φ = copyto!(dense_propagator.wrk.v0, Ψ₀_dense);

norm(Φ - Ψ₀_dense)

x = prop_step!(dense_propagator);

y = prop_step!(propagator);

norm(Ψ₀_dense - Array(Ψ₀))

norm(x - Array(y))

typeof(propagator)

# + active=""
# @benchmark init_prop($Ψ₀, $H, $tlist)

# + active=""
# propagator = init_prop(Ψ₀, H, tlist; cheby_coeffs_limit)
# @benchmark prop_step!(propagator) evals=1 samples=900
# -

propagator = init_prop(Ψ₀, H, tlist; cheby_coeffs_limit)
Ψ_out = _propagate(propagator, tlist)
propagator = init_prop(Ψ₀, H, tlist; cheby_coeffs_limit)
@time _propagate(propagator, tlist);

benchmark_cheby_with_qdyn(
    Ψ₀_dense, H_dense, tlist;
    rf=datadir("cheby_gpu_sanity"),
    cheby_coeffs_limit
)

Ψ_out_qdyn = pyconvert(Vector{ComplexF64},
    qdyn.io.read_psi_amplitudes(
        datadir("cheby_gpu_sanity", "final_state.dat.1"),
        N;
        normalize=false
    )
);

norm(Array(Ψ_out))

norm(Ψ₀_dense - Array(Ψ_out))

norm(Ψ_out_qdyn - Array(Ψ_out))

# ## Benchmark

load_csv(f) = DataFrame(CSV.File(f))

PRECISION =  [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]

data_cheby = run_or_load(datadir("benchmark_gpu_cheby.csv"); load=load_csv, force=true) do
    run_propagation_benchmark(;
        N=[10, 100, 1_000],
        precision=PRECISION,
        method=:cheby,
        generator_args=Dict(:exact_spectral_envelope => true, :hermitian => true),
        exact_propagation_args=Dict(:cheby_coeffs_limit => 1e-15),
        convert_generator=ham_to_gpu,
        convert_state=psi_to_gpu,
        tune_propagation_args=tune_cheby
    )
end

for group in groupby(data_cheby, :N)
    N = group.N[1]
    fig = plot(
        group.precision, group.timing; marker=true, label="Julia",
        xaxis=:log,
        xlabel=raw"precision (absolute error)",
        ylabel="runtime (seconds)",
        xticks=PRECISION, title="Cheby (dense) – Hilbert space dimension N=$N",
    )
    y_limits = ylims(fig)
    plot!(fig; ylims=(0, y_limits[2]))
    display(fig)
end


