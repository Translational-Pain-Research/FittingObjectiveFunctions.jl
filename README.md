# FittingObjectiveFunctions.jl

[<img src="FittingObjectiveFunctions-docs.svg" style="height: 2em;">](https://translational-pain-research.github.io/FittingObjectiveFunctions-documentation/)

A lightweight [Julia](https://julialang.org/) package without dependencies to create objective functions for model fitting. This package does not include optimizers/samplers.

## Installation

The package can be installed with the following commands

```julia
using Pkg
Pkg.Registry.add()
Pkg.Registry.add(RegistrySpec(url = "https://github.com/Translational-Pain-Research/Translational-Pain-ResearchRegistry"))
Pkg.add("FittingObjectiveFunctions")
```
Since the package is not part of the `General` registry the commands install the additional registry `Translational-Pain-ResearchRegistry` first.

After the installation, the package can be used like any other package:
```julia
using FittingObjectiveFunctions
```

## Resources

* **Documentation:** [https://translational-pain-research.github.io/FittingObjectiveFunctions-documentation/](https://translational-pain-research.github.io/FittingObjectiveFunctions-documentation/)
