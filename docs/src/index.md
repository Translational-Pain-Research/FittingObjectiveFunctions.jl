# FittingObjectiveFunctions

## About

[`FittingObjectiveFunctions.jl`](https://github.com/Translational-Pain-Research/FittingObjectiveFunctions.jl) is a lightweight package without dependencies to create objective functions for model fitting. This package does not include optimizers/samplers.


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
```@example 1
using FittingObjectiveFunctions
```


## Simple example

Consider the following example data-set:

```@example 1
using Plots

X = collect(1:10)
Y = [1.0, 1.78, 3.64, 3.72, 5.33, 2.73, 7.52, 9.19, 6.73, 8.95]
ΔY = [0.38, 0.86, 0.29, 0.45, 0.66, 2.46, 0.39, 0.45, 1.62, 1.54]

scatter(X,Y, yerror = ΔY, legend=:none, xlabel = "X", ylabel="Y")
```

Before objective functions can be created, the data needs to be summarized in a [`FittingData`](@ref) object:

```@example 1
data = FittingData(X,Y,ΔY)
nothing #hide
```

Information about the model needs to be summarized in a [`ModelFunctions`](@ref) object. Here we choose a simple linear model ``m(x,\lambda) = \lambda x ``:


```@example 1
model = ModelFunctions((x,λ) -> λ*x) 
nothing #hide
```

A weighted least squares objective can be be constructed as follows:

```@example 1
lsq = lsq_objective(data,model)
```

The following plot (not part of this package) shows the connection between the data points, the parameter `λ` of the model `m(x,λ)` and the least squares objective `lsq`:

```@setup 1
using ColorSchemes, Measures

cmap = cgrad(:thermal)

data_plot = plot(; xlabel = "X", ylabel="Y", xlims = [0,maximum(X)+0.5])
for δ in 0.4:-0.1:0.1
	plot!(x->(1+δ)x, label = "λ=$(1+δ)", color = cmap[0.5 + δ/0.8], linewidth= 3)
end
plot!(x->x, label = "λ=1", color = cmap[0.5], linewidth = 3)
for δ in 0.1:0.1:0.4
	plot!(x->(1-δ)x, label = "λ=$(1-δ)", color = cmap[0.5 - δ/0.8], linewidth = 3)
end
scatter!(X,Y, yerror = ΔY, color = 1, label = "data", markersize = 6)
annotate!(5,14,"m(x,λ) = λ⋅x")

lsq_plot = plot(; legend = :top, xlabel = "λ", ylabel = "lsq", xlims = [0.5,1.5]) 
plot!(λ-> lsq(λ), linewidth= 3, label = "lsq")
for δ in 0.4:-0.1:0.1
	scatter!([(1+δ)],[lsq(1+δ)], label = "λ=$(1+δ)", color = cmap[0.5 + δ/0.8], markersize = 5)
end
scatter!([(1)],[lsq(1)], label = "λ=1", color = cmap[0.5], markersize = 5)
for δ in 0.1:0.1:0.4
	scatter!([(1-δ)],[lsq(1-δ)], label = "λ=$(1-δ)", color = cmap[0.5 - δ/0.8], markersize = 5)
end

```

```@example 1
plot(data_plot,lsq_plot, layout = (1,2), size = (900,400), bottom_margin = 5mm, left_margin = 5mm) # hide
```