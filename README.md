# FittingObjectiveFunctions.jl

[<img src="FittingObjectiveFunctions-docs.svg" style="height: 2em;">](https://antibodypackages.github.io/FittingObjectiveFunctions-documentation/)

A lightweight [Julia](https://julialang.org/) package without dependencies to create objective functions for model fitting. This package does not include optimizers/samplers.

## Installation

The package can be installed with the following commands

```julia
using Pkg
Pkg.Registry.add()
Pkg.Registry.add(RegistrySpec(url = "https://github.com/AntibodyPackages/AntibodyPackagesRegistry"))
Pkg.add("FittingObjectiveFunctions")
```
Since the package is not part of the `General` registry the commands install the additional registry `AntibodyPackagesRegistry` first.

After the installation, the package can be used like any other package:
```julia
using FittingObjectiveFunctions
```

## Resources

* **Documentation:** [https://antibodypackages.github.io/FittingObjectiveFunctions-documentation/](https://antibodypackages.github.io/FittingObjectiveFunctions-documentation/)
