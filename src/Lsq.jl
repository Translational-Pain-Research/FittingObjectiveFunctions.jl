export lsq_objective, lsq_partials, lsq_gradient


# Least squares objective functions
####################################################################################################


"""
	lsq_objective(data::FittingData,model::ModelFunctions)
Return the least squares objective as function `λ -> lsq(λ)`.


**Analytical expression**
	
* independent data points ``x_i``
* dependent data points ``y_i``
* errors ``\\Delta y_i``
* model function ``m``


```math
\\text{lsq}(\\lambda) = \\sum_{i=1}^N \\frac{(y_i - m(x_i,\\lambda))^2}{\\Delta y_i^2}
```
"""
function lsq_objective(data::FittingData,model::ModelFunctions)
	consistency_check(data,model)

	# Without explicit errors, the least squares can be calculated with a reduced formula.
	if ones(length(data.independent)) == data.errors
		return let X = data.independent, Y = data.dependent, f = model.model
			@inline function(λ)
				return sum((Y[i]-f(X[i],λ))^2 for i in 1:length(X))
			end
		end
	else
		return let X = data.independent, Y = data.dependent, Σ = data.errors,  f = model.model
			@inline function(λ)
				return sum(((Y[i]-f(X[i],λ))/Σ[i])^2 for i in 1:length(X))
			end
		end
	end
end













# Least squares objective-function partial derivatives
####################################################################################################

"""
	lsq_partials(data::FittingData,model::ModelFunctions)
Return the partial derivatives of the least squares objective function `ob(λ)` as array of functions `[λ->∂_1 ob(λ),…,λ->∂_n ob(λ)]` .

* The partial derivatives ``\\frac{\\partial}{\\partial \\lambda_\\mu} m(x,\\lambda)`` of the model function must be specified in the [`ModelFunctions`](@ref) object `model`.


**Analytical expression**
	
* independent data points: ``x_i``
* dependent data points: ``y_i``
* errors: ``\\Delta y_i``
* model function: ``m``
* partial derivatives of model function in: ``\\frac{\\partial}{\\partial \\lambda_\\mu}m(x,\\lambda)``


```math
 \\frac{\\partial}{\\partial \\lambda_\\mu} \\text{lsq}(\\lambda) = \\sum_{i=1}^N \\frac{ 2 \\cdot (m(x_i,\\lambda) - y_i) \\cdot \\frac{\\partial}{\\partial \\lambda_\\mu} m(x,\\lambda)}{\\Delta y_i^2}
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
Return the gradient of the least squares objective function `ob(λ)` as function `(gradient,λ)->grad!(gradient,λ)` .

* The gradient function `grad!` mutates (for performance) and returns the `gradient`. The elements of `gradient` do not matter, but the type and length must fit.

* The partial derivatives ``\\frac{\\partial}{\\partial \\lambda_\\mu} m(x,\\lambda)`` of the model function must be specified in the [`ModelFunctions`](@ref) object `model`.


**Analytical expression**
	
* independent data points: ``x_i``
* dependent data points: ``y_i``
* errors: ``\\Delta y_i``
* model function: ``m``
* partial derivatives of model function in: ``\\frac{\\partial}{\\partial \\lambda_\\mu}m(x,\\lambda)``


```math
\\nabla \\text{lsq}(\\lambda) =  \\sum_{\\mu}  \\left(\\sum_{i=1}^N \\frac{ 2 \\cdot (m(x_i,\\lambda) - y_i) \\cdot \\frac{\\partial}{\\partial \\lambda_\\mu}m(x,\\lambda) }{\\Delta y_i^2} \\right) \\vec{e}_\\mu
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