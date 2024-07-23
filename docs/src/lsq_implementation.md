# LSQ: How to implement

Consider the data and model from [Simple example](@ref Simple-example):
```@example 1
using FittingObjectiveFunctions, Plots #hide

X = collect(1:10)
Y = [1.0, 1.78, 3.64, 3.72, 5.33, 2.73, 7.52, 9.19, 6.73, 8.95]
ΔY = [0.38, 0.86, 0.29, 0.45, 0.66, 2.46, 0.39, 0.45, 1.62, 1.54]
data = FittingData(X,Y,ΔY)
model = ModelFunctions((x,λ)-> λ*x)

nothing #hide
```

## Objective functions

Use [`lsq_objective`](@ref) to construct the uncertainty-weighted least squares objective: 
```@example 1
weighted_lsq = lsq_objective(data,model)
```
The returned objective function takes the model parameter (array) `λ` as argument `weighted_lsq(λ)`.

To obtain the standard least squares objective, the errors must be set to `1`, e.g. by using the shortened constructor (see [The `FittingData` struct](@ref The-FittingData-struct)):

``` @example 1
data_no_errors = FittingData(X,Y)
standard_lsq = lsq_objective(data_no_errors,model)
w_lsq_plot = plot(weighted_lsq, label = "weighted lsq", xlims = [0,2], xlabel = "λ", ylabel = "lsq") #hide
plot!(standard_lsq, label = "standard lsq", xlims = [0,2], xlabel = "λ", ylabel = "lsq", color = 2) #hide
```


## [Partial derivatives and gradients](@id lsq_derivatives)

To obtain partial derivatives or the gradient of the least squares objective function, the partial derivatives of the model function need to be added to the [`ModelFunctions`](@ref) object (cf. [The `ModelFunctions` struct](@ref The-ModelFunctions-struct)):

``` @example 1
model = ModelFunctions((x,λ)->λ*x , partials = [(x,λ)-> x])
```

The partial derivatives of the least squares objective can be obtained with [`lsq_partials`](@ref)
``` @example 1
∂_weighted_lsq = lsq_partials(data,model)
```
Note that [`lsq_partials`](@ref) returns the partial derivatives as vector of abstract functions with `λ` as argument, even in the 1-dimensional case.

``` @example 1
∂_weighted_lsq[1](1.1)
```

The gradient of the least squares objective can be obtained with [`lsq_gradient`](@ref)
``` @example 1
∇_weighted_lsq = lsq_gradient(data,model) 
```
The returned gradient function has the signature `(grad_vector,λ)`. The argument `grad_vector` must be a vector of appropriate type and length, that can be mutated.
``` @example 1
∇_weighted_lsq([0.0],1.1) 
```

!!! info "Mutation of gradient vector"
	In some optimization algorithms, the gradient function is called multiple times during each iteration. Mutating an array allows to reduce the memory allocation overhead of creating new gradient arrays.
