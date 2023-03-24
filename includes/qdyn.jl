using QuantumControl.Generators: Generator
using SparseArrays

using PythonCall
qdyn = pyimport("qdyn")

function QDYNPulse(ampl, tlist)
    pulse_tgrid = qdyn.pulse.pulse_tgrid(tlist[end], length(tlist))
    return qdyn.pulse.Pulse(pulse_tgrid, ampl, time_unit = "iu", ampl_unit = "iu")
end


function add_generator_to_qdyn_model(model, generator::Generator{Matrix{Float64}, Vector{Float64}}, tlist)
    n_drift = length(generator.ops) - length(generator.amplitudes)
    for (i, H) in enumerate(generator.ops)
        if i <= n_drift
            model.add_ham(H, label = "")
        else
            ampl = generator.amplitudes[i-n_drift]
            pulse = QDYNPulse(ampl, tlist)
            model.add_ham(H, pulse = pulse, op_type = "dip", label = "")
        end
    end

end


function add_generator_to_qdyn_model(model, generator::Generator{SparseArrays.SparseMatrixCSC{Float64, Int64}, Vector{Float64}}, tlist)
    n_drift = length(generator.ops) - length(generator.amplitudes)
    for (i, H) in enumerate(generator.ops)
        if i <= n_drift
            model.add_ham(H, label = "", sparsity_model="indexed")
        else
            ampl = generator.amplitudes[i-n_drift]
            pulse = QDYNPulse(ampl, tlist)
            model.add_ham(
                H, pulse = pulse, op_type = "dip", label = "",
                sparsity_model="indexed"
            )
        end
    end

end


function belapsed_qdyn_prop_traj(
    rf;
    samples = 10000,
    min_samples = 3,
    seconds = 5.0,
    qdyn_prop_traj,
)
    elapsed = 0.0
    rx = r"Completed traj \d+/\d+ in proc \d/\d:  ([\dE+-.]+) sec"is
    min_runtime::Float64 = Inf
    for sample = 1:samples
        final_state_file = joinpath(rf, "final_state.dat")
        res =
            @timed read(`$qdyn_prop_traj --write-final-state=$final_state_file $rf`, String)
        m = match(rx, res.value)
        @assert !isnothing(m)
        runtime = parse(Float64, m.captures[1])
        if runtime < min_runtime
            min_runtime = runtime
        end
        elapsed += res.time
        if (elapsed > seconds) && (sample ≥ min_samples)
            break
        end
    end
    return min_runtime
end


function benchmark_cheby_with_qdyn(
    Ψ,
    H,
    tlist;
    rf,
    seconds = 5.0,
    propagation_args...,
)
    cheby_coeffs_limit = propagation_args[:cheby_coeffs_limit]
    model = qdyn.model.LevelModel()
    model.add_state(Ψ, label = "")
    add_generator_to_qdyn_model(model, H, tlist)
    model.set_propagation(
        T = tlist[end],
        nt = length(tlist),
        time_unit = "iu",
        prop_method = "cheby",
    )
    mkpath(rf)
    model.write_to_runfolder(rf)
    config_data = qdyn.config.read_config_file(joinpath(rf, "config"))
    config_data["prop"]["cheby_prec"] = cheby_coeffs_limit
    qdyn.config.write_config(config_data, joinpath(rf, "config"))
    runtimes = []
    for qdyn_folder in ["qdyn_ifort", "qdyn_ifort_fast", "qdyn_gfortran"]
        @assert isdir(qdyn_folder)
        qdyn_prop_traj = joinpath(qdyn_folder, "utils", "qdyn_prop_traj")
        @assert isfile(qdyn_prop_traj) qdyn_prop_traj
        t = belapsed_qdyn_prop_traj(rf; qdyn_prop_traj, seconds)
        push!(runtimes, t)
    end
    return cheby_coeffs_limit, runtimes...
end
