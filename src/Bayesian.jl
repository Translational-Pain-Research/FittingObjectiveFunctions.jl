export posterior_objective, log_posterior_objective, log_posterior_partials, log_posterior_gradient


# Inline uniform prior
@inline function uniform_prior(λ)
 return 1
end

@inline function log_uniform_prior(λ)
 return 0
end





"""
	posterior_objective(data::FittingData, 
		model::Function,distribution::Function, 
		prior = λ-> 1
	)
Return the unnormalized posterior density as function `λ->p(λ)`.
	
Using the default prior `λ-> 1`, e.g. py passing only the first two arguments, leads to the likelihood objective for a maximum likelihood fit.


**Analytical expression**

* independent data points ``x_i``
* dependent data points ``y_i``
* errors ``\\Delta y_i``
* model function ``m``
* ``y``-uncertainty distributions: ``q_i``
* prior distribution: ``p_0``


```math
p(\\lambda) = p_0(\\lambda) \\cdot \\prod_{i=1}^N q_i(y_i,m(x_i,\\lambda),\\Delta y_i)
```
"""
function posterior_objective(data::FittingData,model::ModelFunctions, prior::Function = uniform_prior)
	consistency_check(data,model)

	return let  X = data.independent, Y = data.dependent, Σ = data.errors, f = model.model, p = data.distributions
		@inline function(λ)
			return prod(p[i](Y[i],f(X[i],λ), Σ[i]) for i in 1:length(X)) * prior(λ)
		end
	end
end








"""
	log_posterior_objective(data::FittingData,
		model::ModelFunctions, 
		log_prior::Function = log_uniform_prior
	)

Return the logarithmic posterior density as function `λ->L_p(λ)`. 
	
* The `y`-uncertainty distributions of the [`FittingData`](@ref) object `data` and `log-prior` must be specified in the logarithmic form.
	
* Using the default prior, e.g. py passing only the first two arguments, leads to the logarithmic likelihood objective for a maximum likelihood fit.


**Analytical expression**

* independent data points ``x_i``
* dependent data points ``y_i``
* errors ``\\Delta y_i``
* model function ``m``
* logarithmic ``y``-uncertainty distributions: ``L_i``
* logarithmic prior distribution: ``L_0``

```math
L_p(\\lambda) = L_0(\\lambda) + \\sum_{i=1}^N L_i(y_i,m(x_i,\\lambda),\\Delta y_i)
```

"""
function log_posterior_objective(data::FittingData,model::ModelFunctions, log_prior::Function = log_uniform_prior)
	consistency_check(data,model)
	# Let block to encapsule data in returned function, since data is mutable.
	return let  X = data.independent, Y = data.dependent, Σ = data.errors, f = model.model, p = data.distributions
		@inline function(λ)
			return sum(p[i](Y[i],f(X[i],λ), Σ[i]) for i in 1:length(X)) + log_prior(λ)
		end
	end

end



"""
	log_posterior_partials(data::FittingData,
		model::ModelFunctions,
		log_distribution_derivatives, 
		prior_partials::Union{Nothing,AbstractArray{Function,N}} = nothing
	) 

Return the partial derivatives of the log-posterior distribution `L_p(λ)` as array of functions `[λ->∂_1 L_p(λ),…,λ->∂_n L_p(λ)]`.

* The partial derivatives ``\\frac{\\partial}{\\partial \\lambda_\\mu} m(x,\\lambda)`` of the model function must be specified in the [`ModelFunctions`](@ref) object `model`.

* `log_distribution_derivatives` can either be a function ``\\frac{\\partial}{\\partial m} L(y,m,\\Delta y)`` (same derivative for all distributions), or an array of functions ``\\left[\\frac{\\partial}{\\partial m} L_1(y,m,\\Delta y),\\ldots,\\frac{\\partial}{\\partial m} L_n(y,m,\\Delta y) \\right]``

* `prior_partials` can either be `nothing` (for the log-likelihood), or an array of functions ``\\left[\\frac{\\partial}{\\partial \\lambda_1} L_0(λ),\\ldots,\\frac{\\partial}{\\partial \\lambda_n} L_0(λ) \\right]``.

**Analytical expression**

* independent data points ``x_i``
* dependent data points ``y_i``
* errors ``\\Delta y_i``
* model function ``m``
* logarithmic ``y``-uncertainty distributions: ``L_i``
* logarithmic prior distribution: ``L_0``
* partial derivatives of model function: ``\\frac{\\partial}{\\partial \\lambda_\\mu} m(x,\\lambda)``
* partial derivatives of the logarithmic ``y``-uncertainty distributions: ``\\frac{\\partial}{\\partial m} L_i(y,m,\\Delta y)``
* partial derivatives of the logarithmic prior: ``\\frac{\\partial}{\\partial \\lambda_\\mu} L_0(λ)``


```math
\\frac{\\partial}{\\partial \\lambda_\\mu} L_p(\\lambda) = \\frac{\\partial}{\\partial \\lambda_\\mu} L_0(\\lambda) + \\sum_{i=1}^N  \\frac{\\partial}{\\partial m} L_i(y_i, m(x_i,\\lambda), \\Delta y_i)\\cdot \\frac{\\partial}{\\partial \\lambda_\\mu} m(x_i,\\lambda)
```
"""
function log_posterior_partials(data::FittingData,model::ModelFunctions, log_distribution_derivatives, prior_partials::Union{Nothing,AbstractArray{F,N}} = nothing) where {F <: Function, N}

	consistency_check(data,model)

	if isnothing(model.partials)
		throw(ArgumentError("`model.partials` must not be `nothing` for `lsq_partials`."))
	end

	if typeof(log_distribution_derivatives) <: Function
		log_distribution_derivatives = [log_distribution_derivatives for i in 1:length(data.distributions)]
	end

	if length(log_distribution_derivatives) != length(data.distributions)
		throw(ArgumentError("Number of log-distribution derivatives does not match the number of log-distributions."))
	end





	if isnothing(prior_partials)
		# Simplified version for prior_partials = nothing, i.e. [λ-> 0 for i in 1:length(model.partials)]
		∂ = similar(model.partials,Function)

		for k in eachindex(model.partials)
			∂_k  = let X = data.independent, Y = data.dependent, Σ = data.errors, f = model.model, ∂_kf = model.partials[k], ∂_p = log_distribution_derivatives
				@inline function(λ)
					sum(∂_p[i](Y[i],f(X[i],λ),Σ[i])*∂_kf(X[i],λ) for i in 1:length(X))
				end
			end
			∂[k] = ∂_k
		end
		return ∂


	else
		if length(prior_partials) != length(model.partials)
			throw(ArgumentError("Number of prior derivatives does not match number of `model.partials`."))
		end

		∂ = similar(model.partials,Function)

		for k in eachindex(model.partials)
			∂_k  = let X = data.independent, Y = data.dependent, Σ = data.errors, f = model.model, ∂_kf = model.partials[k], ∂_p = log_distribution_derivatives, ∂_kp_0 = prior_partials[k]
				@inline function(λ)
					sum(∂_p[i](Y[i],f(X[i],λ),Σ[i])*∂_kf(X[i],λ) for i in 1:length(X)) + ∂_kp_0(λ)
				end
			end
			∂[k] = ∂_k
		end
		return ∂
	end
end
















"""
	log_posterior_gradient(data::FittingData,
		model::ModelFunctions, 
		log_distribution_derivatives, 
		prior_partials::Union{Nothing,AbstractArray{Function,N}} = nothing
	) 
Return the gradient of the log-posterior distribution `L_p(λ)` as function `(gradient,λ)->grad!(gradient,λ)` .

* The gradient function `grad!` mutates (for performance) and returns the `gradient`. The elements of `gradient` do not matter, but the type and length must fit.

* The partial derivatives ``\\frac{\\partial}{\\partial \\lambda_\\mu} m(x,\\lambda)`` of the model function must be specified in the [`ModelFunctions`](@ref) object `model`.

* `log_distribution_derivatives` can either be a function ``\\frac{\\partial}{\\partial m} L(y,m,\\Delta y)`` (same derivative for all distributions), or an array of functions ``\\left[\\frac{\\partial}{\\partial m} L_1(y,m,\\Delta y),\\ldots,\\frac{\\partial}{\\partial m} L_n(y,m,\\Delta y) \\right]``

* `prior_partials` can either be `nothing` (for the log-likelihood), or an array of functions ``\\left[\\frac{\\partial}{\\partial \\lambda_1} L_0(λ),\\ldots,\\frac{\\partial}{\\partial \\lambda_n} L_0(λ) \\right]``.

**Analytical expression**

* independent data points ``x_i``
* dependent data points ``y_i``
* errors ``\\Delta y_i``
* model function ``m``
* logarithmic ``y``-uncertainty distributions: ``L_i``
* logarithmic prior distribution: ``L_0``
* partial derivatives of model function: ``\\frac{\\partial}{\\partial \\lambda_\\mu} m(x,\\lambda)``
* partial derivatives of the logarithmic ``y``-uncertainty distributions: ``\\frac{\\partial}{\\partial m} L_i(y,m,\\Delta y)``
* partial derivatives of the logarithmic prior: ``\\frac{\\partial}{\\partial \\lambda_\\mu} L_0(λ)``


```math
\\nabla L_p(\\lambda) = \\sum_{\\mu} \\left( \\frac{\\partial}{\\partial \\lambda_\\mu} L_0(\\lambda) + \\sum_{i=1}^N  \\frac{\\partial}{\\partial m} L_i(y_i, m(x_i,\\lambda), \\Delta y_i)\\cdot \\frac{\\partial}{\\partial \\lambda_\\mu} m(x_i,\\lambda) \\right) \\vec{e}_\\mu
```
"""
function log_posterior_gradient(data::FittingData,model::ModelFunctions, log_distribution_derivatives, prior_gradient::Union{Nothing,Function} = nothing)

	partials = log_posterior_partials(data,model,log_distribution_derivatives, nothing)
	
	if isnothing(prior_gradient)
		# Without prior_gradient, only partial derivatives (without prior_partials) are needed.
		return @inline function(gradient,λ)
			for i in eachindex(partials)
				gradient[i] = partials[i](λ)
			end
			return gradient
		end
	else
		return @inline function(gradient,λ)
			prior_grad =  prior_gradient(gradient,λ)
			for i in eachindex(partials)
				gradient[i] = partials[i](λ) + prior_grad[i]
			end
			return gradient
		end
	end
end