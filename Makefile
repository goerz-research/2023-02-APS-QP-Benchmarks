.PHONY: init help clean distclean

.DEFAULT_GOAL := help

define PRINT_HELP_JLSCRIPT
rx = r"^([a-z0-9A-Z_-]+):.*?##[ ]+(.*)$$"
for line in eachline()
    m = match(rx, line)
    if !isnothing(m)
        target, help = m.captures
        println("$$(rpad(target, 20)) $$help")
    end
end
endef
export PRINT_HELP_JLSCRIPT

JULIA ?= julia

help:  ## Show this help
	@$(JULIA) -e "$$PRINT_HELP_JLSCRIPT" < $(MAKEFILE_LIST)

init: Manifest.toml qdyn_ifort/utils/qdyn_prop_traj  ## Initialize the repo

Manifest.toml: Project.toml
	$(JULIA) --project=. -e 'using Pkg; Pkg.instantiate()'

qdyn_ifort/utils/qdyn_prop_traj:
	$(JULIA) --project=. scripts/compile_qdyn.jl

# Note: benchmarks can run in parallel, but make sure there are enough cores
benchmarks: Manifest.toml qdyn_ifort/utils/qdyn_prop_traj  ## Run all benchmarks
	GKS_WSTYPE=100 $(JULIA) --project=. -t auto benchmark_cheby_dense.jl
	GKS_WSTYPE=100 $(JULIA) --project=. -t auto benchmark_cheby_sparse.jl
	GKS_WSTYPE=100 $(JULIA) --project=. -t auto benchmark_cheby_static.jl
	GKS_WSTYPE=100 $(JULIA) --project=. -t auto benchmark_cheby_dense_mkl.jl
	GKS_WSTYPE=100 $(JULIA) --project=. -t auto benchmark_cheby_sparse_mkl.jl

clean:  ## Remove generated files

distclean: clean  ## Restore clean repository state
	rm -rf .CondaPkg
	rm -rf qdyn*
	rm -rf .ipynb_checkpoints
	rm -rf data/*
