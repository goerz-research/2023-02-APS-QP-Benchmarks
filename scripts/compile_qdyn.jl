using CondaPkg
CondaPkg.resolve()

python = CondaPkg.which("python")
QDYN_URL = "git@gitlabph.physik.fu-berlin.de:ag-koch/qdyn.git"
QDYN_BRANCH = "benchmark-prop-traj"


VERSIONS = [
    ("qdyn_ifort", `$python ./configure --fc=ifort --python=$python`),
    ("qdyn_ifort_fast", `$python ./configure --fc=ifort --fast --python=$python`),
    ("qdyn_gfortran", `$python ./configure --fc=gfortran --python=$python`),
]


for (folder, cmd) in VERSIONS
    if !isdir(folder)
        run(`git clone $QDYN_URL $folder`)
        cd(folder) do
            run(`git checkout $QDYN_BRANCH`)
            run(cmd)
            run(`make`)
            run(`make utils`)
        end
    else
        @info "$folder already exists"
    end
end
