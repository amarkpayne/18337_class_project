import Pkg
Pkg.activate("class_project_env")

Pkg.add("Conda")
Pkg.add("PyCall")

import Conda
ENV["CONDA_JL_HOME"] = "/home/ampayne/anaconda2/envs/rms_env"
isfile(Conda.conda)


Pkg.build("Conda")
ENV["PYTHON"] = "/home/ampayne/anaconda2/envs/rms_env/bin/python"
Pkg.build("PyCall")

Pkg.develop(path="DifferentialEquations.jl/")
Pkg.build("DifferentialEquations")

Pkg.develop(path="ReactionMechanismSimulator.jl/")
Pkg.build("ReactionMechanismSimulator")

Pkg.test("ReactionMechanismSimulator")
