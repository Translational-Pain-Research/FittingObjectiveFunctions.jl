export FittingData, ModelFunctions, consistency_check



# Calculate sqrt term once, and not every time the normal distribution is calculated.
const sqrt_term = sqrt(2*pi)

# Default uncertainty distribution.
@inline function normal_distribution(x,μ,σ)
	return exp(-(x-μ)^2/(2*σ^2))/(sqrt_term * σ)
end


# Check types and lengths of the constructor arguments for FittingData.
# Also allows for more verbose error messages.
function constructor_checks_fitting_data(independent,dependent)
	if !(typeof(independent) <: AbstractArray)
		throw(ArgumentError("`independent` argument must be an array. Got $(typeof(independent))."))
	end

	if !(typeof(dependent) <: AbstractArray)
		throw(ArgumentError("`dependent` argument must be an array. Got $(typeof(dependent))."))
	end

	if !(length(independent)==length(dependent))
		throw(DimensionMismatch("Arrays for independent data and dependent data do not have the same length."))
	end

end



# Check types and lengths of the constructor arguments for FittingData.
# Also allows for more verbose error messages.
function constructor_checks_fitting_data(independent,dependent,errors,distributions)
	if !(typeof(independent) <: AbstractArray)
		throw(ArgumentError("`independent` argument must be an array. Got $(typeof(independent))."))
	end

	if !(typeof(dependent) <: AbstractArray)
		throw(ArgumentError("`dependent` argument must be an array. Got $(typeof(dependent))."))
	end

	if !(typeof(errors) <: AbstractArray)
		throw(ArgumentError("`errors` argument must be an array. Got $(typeof(errors))."))
	end


	if 0 in errors
		@warn("The errors contain at least once the value `0`. This can lead to `NaN` and `Inf` or unexpected behavior!")
	end

	# Test properties of distributions.

	if !(typeof(distributions) <: AbstractArray  && eltype(distributions)<: Function)
		throw(ArgumentError("`distributions` must be a `Function` or an array of `Function`. Got $(typeof(distributions))."))
	end


	if !(length(independent)==length(dependent)==length(errors)==length(distributions))
		throw(DimensionMismatch("Arrays for the independent data, dependent data, errors or distributions do not have the same length."))
	end

end



"""
	mutable struct FittingData

Data type for fitting data.

This struct is only a container to check consistency and is not performance relevant, hence the mutability.

**Fields**

* `independent`: Array of data points for the independent variable. 
* `dependent`: Array of data points for the dependent variable.
* `errors`: Array of measurement errors for the dependent variable.
* `distributions`: Distribution(s) for the uncertainty of the dependent variable. Can be a function or an array of functions (one for each data point). 

Elements with the same index belong together, i.e. define a measurement: 

	(independent[i], dependent[i], errors[i], distributions[i])


**Constructors**

```julia
FittingData(X,Y)
```

```julia
FittingData(X,Y,ΔY;distributions = (y,m,Δy) -> exp(-(y-m)^2/(2*Δy^2))/(sqrt(2*pi) * Δy))
```

**Distributions**

The distributions must have the signature `(y,m,Δy)`, where `y` is the dependent variable, `m` is the result of the model function and `Δy` is the error of the dependent variable. If the distributions are not specified, a normal distribution is used:

	(y,m,Δy) -> exp(-(y-m)^2/(2*Δy^2))/(sqrt(2*pi) * Δy)
"""
mutable struct FittingData
	independent::AbstractArray{R,N} where {R,N}
	dependent::AbstractArray{R,N} where {R,N}
	errors::AbstractArray{R,N} where {R,N}
	distributions::AbstractArray{F,N} where {F <: Function,N}

	# Constructors for convenience.
	function FittingData(independent,dependent; distributions = normal_distribution)
		# Check required user-passed arguments.
		constructor_checks_fitting_data(independent,dependent)

		# Allows to pass only a single distribution function, that is used for all data points.
		if typeof(distributions) <: Function
			distributions = Function[distributions for i in 1:length(independent)]
		end

		# Default errors.
		errors = ones(length(dependent))
		
		# Check extended arguments after transforming user-passed arguments.
		constructor_checks_fitting_data(independent,dependent,errors,distributions)

		return new(independent,dependent,errors,distributions)
	end

	function FittingData(independent, dependent,errors; distributions = normal_distribution)
		# Check required user-passed arguments.
		constructor_checks_fitting_data(independent,dependent)

		# Allows to pass only a single distribution function, that is used for all data points.
		if typeof(distributions) <: Function
			distributions = Function[distributions for i in 1:length(independent)]
		end

		# Check extended arguments after transforming user-passed arguments.
		constructor_checks_fitting_data(independent,dependent,errors,distributions)

		return new(independent,dependent,errors,distributions)
	end
end



# Check types of partials. Allows for more verbose error messages.
function constructor_checks_model_functions(model,partials)
	# Test properties of `partials`.
	if !isnothing(partials)
		if !(typeof(partials)<:AbstractArray)
			throw(ArgumentError("`partials` must be an array of `Function`. Got $(typeof(partials))."))
		end
		if !(eltype(partials)<: Function)
			throw(ArgumentError("`partials` must be an array of `Function`. Got $(typeof(partials))."))
		end
	end
end





"""
	mutable struct ModelFunctions
Mutable type to collect model functions (and the respective partial derivatives) to construct objective functions.

This struct is only a container to check consistency and is not performance relevant, hence the mutability.

**Fields**

* `model`: The model function. Must have the signature `(x,λ)`, where `x` is the independent variable, and `λ` is the parameter (array).
* `partials`: Array of partial derivative functions (one for each parameter array element). Must have the same signature `(x,λ)` as the model function.


**Constructor**

	ModelFunctions(model, partials = nothing)

**Examples**

```julia-repl
julia> ModelFunctions((x,λ)-> λ*x)	
```

```julia-repl
julia> ModelFunctions((x,λ)-> λ*x, partials = [(x,λ)-> x])	
```

```julia-repl
julia> ModelFunctions((x,λ)-> λ[1]*x+λ[2], partials = [(x,λ)-> x, (x,λ)-> 1])	
```

"""
mutable struct ModelFunctions
	model::Function
	partials::Union{Nothing, AbstractArray{F,N}} where {F <: Function,N}

	function ModelFunctions(model; partials = nothing)
		constructor_checks_model_functions(model,partials)

		return new(model,partials)
	end
end





"""
	consistency_check(fitting_data::FittingData,model::ModelFunctions)
Test `fitting_data` and `model`, e.g. after mutation. 
"""
function consistency_check(fitting_data::FittingData,model::ModelFunctions)
	
	# Repeat constructor tests, because of mutability

	constructor_checks_model_functions(model.model,model.partials)

	# After construction, errors are required (set to 1 by default).
	constructor_checks_fitting_data(fitting_data.independent,fitting_data.dependent,fitting_data.errors,fitting_data.distributions)

end








"""
	consistency_check(fitting_data::FittingData,model::ModelFunctions,λ)
Test if all functions can be evaluated with the parameter (array) `λ`. Also, test `fitting_data` and `model`, e.g. after mutation. 
"""
function consistency_check(fitting_data::FittingData,model::ModelFunctions,λ)
	
	# Length checks
	consistency_check(fitting_data,model)



	try
		for x in fitting_data.independent
			model.model(x,λ)
		end
	catch
		throw("Could not evaluate the model function.")
	end	



	# Test partial derivative evaluation
	try
		if !isnothing(model.partials) 
			for ∂ in model.partials, x in fitting_data.independent
				∂(x,λ)
			end
		end
	catch
		throw(ArgumentError("Could not evaluate a partial derivative function"))
	end



	# Test uncertainty distributions. Instead of model value, use data point value (which should be valid value for reasonable models).
	try
		for i in 1:length(fitting_data.independent), d in fitting_data.distributions
			d(fitting_data.independent[i],fitting_data.independent[i],fitting_data.errors[i])
		end
	catch
		throw(ArgumentError("Could not evaluate a distribution function"))
	end


end