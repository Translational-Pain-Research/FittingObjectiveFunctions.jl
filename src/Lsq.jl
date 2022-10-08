export lsq_objective, lsq_partials, lsq_gradient


# Least squares objective functions
####################################################################################################


"""
	lsq_objective(data::FittingData,model::ModelFunctions)
Return the least squares objective function `lsq(λ)`.


**Analytical expression**
	
* independent data points `x_i`
* dependent data points `y_i`
* errors `Δy_i` (defaulting to 1)
* model function `m`


```math
\\text{lsq}(\\lambda) = \\sum_{i=1}^N \\frac{(y_i - m(x_i,\\lambda))^2}{\\Delta y_i^2}
```
"""
function lsq_objective(data::FittingData,model::ModelFunctions)
	consistency_check(data,model)

	# Without explicit errors, the least squares can be calculated with a reduced formula.
	if ones(length(data.independent)) == data.errors
		return let X = data.independent, Y = data.dependent, f = model.model
			@inline function(λ,T::Type = Float64)
				return sum((Y[i]-f(X[i],λ))^2 for i in 1:length(X))
			end
		end
	else
		return let X = data.independent, Y = data.dependent, Σ = data.errors,  f = model.model
			@inline function(λ,T::Type = Float64)
				return sum(((Y[i]-f(X[i],λ))/Σ[i])^2 for i in 1:length(X))
			end
		end
	end
end













# Least squares objective-function partial derivatives
####################################################################################################

"""
	lsq_partials(data::FittingData,model::ModelFunctions)
Return the partial derivatives w.r.t. the parameters `[∂_1 ob(λ),…,∂_n ob(λ)]` of the least squares objective function `ob(λ)`.

* The partial derivatives `(∂_μ m)(x,λ)` of the model function must be specified in the `ModelFunctions` object `model`.


**Analytical expression**
	
* independent data points: `x_i`
* dependent data points: `y_i`
* errors: `Δy_i` (defaulting to 1)
* model function: `m`
* partial derivatives of model function in `λ`: `∂_μ m`


```math
\\partial_\\mu \\text{lsq}(\\lambda) = \\sum_{i=1}^N \\frac{ 2 \\cdot (m(x_i,\\lambda) - y_i) \\cdot (\\partial_\\mu m)((x_i,\\lambda)) }{\\Delta y_i^2}
```
"""
function lsq_partials(data::FittingData,model::ModelFunctions)
	if isnothing(model.partials)
		throw(ArgumentError("`model.partials` must not be `nothing` for `lsq_partials`."))
	end
	consistency_check(data,model)


	∂ = similar(model.partials,Function)
	# Without explicit errors, the least squares can be calculated with a reduced formula.
	if ones(length(data.independent)) == data.errors
		for k in eachindex(model.partials)
			∂_k  = let X = data.independent, Y = data.dependent, f = model.model, ∂_kf = model.partials[k]
				@inline function(λ)
					2 * sum(-(Y[i]-f(X[i],λ)) * ∂_kf(X[i],λ) for i in 1:length(X))
				end
			end
			∂[k] = ∂_k
		end
		return ∂
	else
		for k in eachindex(model.partials)
			∂_k  = let X = data.independent, Y = data.dependent,Σ2 = data.errors .^2, f = model.model, ∂_kf = model.partials[k]
				@inline function(λ)
					2 * sum(-(Y[i]-f(X[i],λ))* ∂_kf(X[i],λ)/Σ2[i] for i in 1:length(X))
				end
			end
			∂[k] = ∂_k
		end
		return ∂
	end
end




















"""
	lsq_gradient(data::FittingData,model::ModelFunctions)
Return the gradient function `grad!(gradient,λ)` for the least squares objective function `ob(λ)`.
The gradient function `grad!` mutates (for performance) and returns the `gradient`. The elements of `gradient` do not matter, but the type and length must fit.


* The partial derivatives `(∂_μ m)(x,λ)` of the model function must be specified in the `ModelFunctions` object `model`.

**Analytical expression**
	
* independent data points: `x_i`
* dependent data points: `y_i`
* errors: `Δy_i` (defaulting to 1)
* model function: `m`
* partial derivatives of model function in `λ`: `∂_μ m`


```math
\\nabla \\text{lsq}(\\lambda) =  \\sum_{\\mu}  \\sum_{i=1}^N \\frac{ 2 \\cdot (m(x_i,\\lambda) - y_i) \\cdot (\\partial_\\mu m)((x_i,\\lambda)) }{\\Delta y_i^2} \\ \\vec{e}_\\mu
```
"""
function lsq_gradient(data::FittingData,model::ModelFunctions)
	partials = lsq_partials(data,model)
	
	return @inline function(gradient,λ)
		for i in 1:length(partials)
			gradient[i] = partials[i](λ)
		end
		return gradient
	end
end