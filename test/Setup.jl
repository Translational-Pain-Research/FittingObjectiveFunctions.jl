using FittingObjectiveFunctions
using Test




# Bundle test data in a single struct
####################################################################################################

# Struct to summarize the information about a data set.
struct test_data{NameType<:AbstractString,ModelType<:Function, DerivativesType, XType,YType,ΣType, TrueParameterType}
	name::NameType
	model::ModelType
	∂::DerivativesType
	X::XType
	Y::YType
	Σ::ΣType
	λ::TrueParameterType
end

# Generate dependent values from the model function to guarantee consistency.
# Set error to 1 to compare explicitly specified errors with the default errors (that should also be 1).
function test_data(name,model,∂,X,λ)
	Y = [model(x,λ) for x in X]
	Σ = ones(length(X)) 
	return test_data(name,model,∂,X,Y,Σ,λ)
end

# Test combinations for one-dimensional and multidimensional cases.
data_1d_1d = test_data("1-dim data, 1-dim parameter", (x,λ)-> λ*x, [(x,λ)-> x], [1,2,3],1)
data_2d_1d = test_data("2-dim data, 1-dim parameter", (x,λ)-> λ*(x[1] + x[2]), [(x,λ)-> (x[1]+x[2])], [[1,1],[2,2],[3,3]],1)
data_1d_2d = test_data("1-dim data, 2-dim parameter", (x,λ)-> λ[1]*x+λ[2], [(x,λ)-> x, (x,λ)-> 1], [1,2,3],[1,0])
data_2d_2d = test_data("2-dim data, 2-dim parameter", (x,λ)-> λ[1]*x[1] + λ[2]*x[2], [(x,λ)-> x[1], (x,λ)-> x[2]], [[1,1],[2,2],[3,3]],[1,1])


# Collect data sets in an array such that the test cases can be iterated.
data_sets = [data_1d_1d, data_2d_1d,data_1d_2d, data_2d_2d]














# Auxiliary functions 
####################################################################################################


# Normal distribution as uncertainty distribution for posterior objective functions.
normal_distribution(x,μ,σ) = 1/(sqrt(2*pi)*σ) * exp(-(x-μ)^2/(2*σ^2))

# Shortcut for logarithmic array.
function LogRange(start,stop,n)
	return 10 .^ LinRange(log10(start),log10(stop),n)
end

# Test if all elements of an array are approximately the same.
function all_equal(array::AbstractArray{R,N}) where {R<: Real, N}
	for i in 2:length(array)
		if !(array[i-1] ≈ array[i])
			return false
		end
	end
	return true
end

# Dispatch version for array of arrays.
function all_equal(array)
	for i in 2:length(array)
		if length(array[i-1]) != length(array[i])
			return false
		end
		for j in 1:length(array[i])
			if !(array[i-1][j] ≈ array[i][j])
				return false
			end
		end
	end
	return true
end






# Test functions
####################################################################################################

# Function to test if different objective functions are numerically equal (e.g. to test different constructors).
function test_objective_equality(objectives,λ::Real)
	equal = true
	for shift in LogRange(1,10^6,10^4)
		equal *= all_equal([ob(λ + shift) for ob in objectives])
		equal *= all_equal([ob(λ - shift) for ob in objectives])
	end
	return equal
end

function test_objective_equality(objectives,λ)
	equal = true
	for shift_1 in LogRange(1,10^6,10^2) ,shift_2 in LogRange(1,10^6,10^2)
		equal *= all_equal([ob(λ .+ [shift_1,shift_2]) for ob in objectives])
		equal *= all_equal([ob(λ .- [shift_1,shift_2]) for ob in objectives])
	end
	return equal
end



# Test positive semi-definiteness of objective functions (e.g. for least squares).
function test_objective_positivity(objective::Function, λ::Real)
	positive = true
	for shift in LogRange(1,10^6,10^4)
		positive *= objective(λ+shift) >= 0
		positive *= objective(λ+shift) >= 0
	end
	return positive
end


function test_objective_positivity(objective::Function, λ)
	positive = true
	for shift_1 in LogRange(1,10^6,10^2) ,shift_2 in LogRange(1,10^6,10^2)
		positive *= objective(λ .+ [shift_1,shift_2]) >= 0
		positive *= objective(λ .+ [shift_1,shift_2]) >= 0
	end
	return positive
end


# Test monotony, i.e. the further away the parameters are from the true parameters, the "worse" the objective function gets ("worse" determined by direction).
function test_objective_monotony(objective::Function,λ::Real,direction::Function)
	 monotonic = true
	 shift = LogRange(1,10^6,10^4)
	 for i in 2:length(shift)
		monotonic *= direction(objective(λ+shift[i]), objective(λ+ shift[i-1]))
		monotonic *= direction(objective(λ-shift[i]), objective(λ- shift[i-1]))
	 end
	 return monotonic
end

function test_objective_monotony(objective::Function,λ,direction::Function)
	 monotonic = true
	 shift = LogRange(1,10^6,10^4)
	 for i in 2:length(shift)
		monotonic *= direction(objective(λ .+ [shift[i],0]), objective(λ .+ [shift[i-1],0]))
		monotonic *= direction(objective(λ .-[shift[i],0]), objective(λ .- [shift[i-1],0]))
		monotonic *= direction(objective(λ .+ [0,shift[i]]), objective(λ .+ [0,shift[i-1]]))
		monotonic *= direction(objective(λ .-[0,shift[i]]), objective(λ .- [0,shift[i-1]]))
	 end
	 return monotonic
end


