@testset "Least squares methods" begin
	# Test all data sets, i.e. combination of multidimensional data/parameters.
	for data_set in data_sets
		@testset "$(data_set.name)" begin

			# Fitting data to create (weighted) least squares objectives that are numerically equal (errors = [1,...,1])
			fitting_data = [FittingData(data_set.X,data_set.Y), FittingData(data_set.X, data_set.Y, ones(length(data_set.X)))]
			model = ModelFunctions(data_set.model, partials = data_set.∂)

			# Create objective functions and derivatives.
			objectives = [lsq_objective(fd,model) for fd in fitting_data]
			partials = [lsq_partials(fd,model) for fd in fitting_data]
			gradients = [lsq_gradient(fd,model) for fd in fitting_data]














			@testset "Objective functions" begin

				@testset "Consistency check is called" begin
					fd = deepcopy(fitting_data[1])
					fd.independent = [1]
					@test_throws DimensionMismatch lsq_objective(fd,model)
				end

				@testset "Equality of implicit/explicit errors" begin
					@test test_objective_equality(objectives,data_set.λ)
				end

				# Because of the previously established equality, only objectives[1] needs to be tested.
				##################################################

				@testset "Sum of squares is positive" begin
					@test test_objective_positivity(objectives[1], data_set.λ)
				end

				@testset "Sum of squares monotonously increasing" begin
					# direction >= means "worse" values are "larger". 
					@test test_objective_monotony(objectives[1],data_set.λ,>=)
				end

				@testset "True parameter: Sum of squares is 0" begin
					@test objectives[1](data_set.λ) == 0
				end

				@testset "Comparison to explicit formula" begin
					explicit = function(λ)
						m = [data_set.model(x,λ) for x in data_set.X] 
						return sum(@. (data_set.Y - m)^2)
					end
					@test test_objective_equality([objectives[1],explicit],data_set.λ)
				end

			end
			












			@testset "Partial derivatives" begin

				@testset "Consistency check is called" begin
					fd = deepcopy(fitting_data[1])
					fd.independent = [1]
					@test_throws DimensionMismatch lsq_partials(fd,model)
				end

				@testset "model.partials must not be nothing" begin
					@test_throws ArgumentError lsq_partials(fitting_data[1],ModelFunctions((x,λ)-> λ*x))
				end

				@testset "Equality of implicit/explicit errors" begin
					@test prod(test_objective_equality([∂[i] for ∂ in partials], data_set.λ) for i in eachindex(data_set.λ))
				end

				# Because of the previously established equality, only partials[1] needs to be tested.
				##################################################

				@testset "True parameter: local extrema" begin
					# Distinguish between 1-dim and 2-dim parameters.
					@test [partials[1][i](data_set.λ) for i in eachindex(partials[1])] == zeros(length(partials[1]))
				end


				@testset "Comparison to explicit formula" begin
					explicit = [function(λ)
						m = [data_set.model(x,λ) for x in data_set.X] 
						∂m = [data_set.∂[i](x,λ) for x in data_set.X] 
						return sum(@. 2*(m-data_set.Y)*∂m)
					end for i in eachindex(data_set.λ) ]

					@test prod(test_objective_equality([partials[1][i], explicit[i]], data_set.λ) for i in eachindex(data_set.λ))
				end
			end















			@testset "Gradient" begin

				@testset "Consistency check is called" begin
					fd = deepcopy(fitting_data[1])
					fd.independent = [1]
					@test_throws DimensionMismatch lsq_gradient(fd,model)
				end

				@testset "model.partials must not be nothing" begin
					@test_throws ArgumentError lsq_gradient(fitting_data[1],ModelFunctions((x,λ)-> λ*x))
				end

				@testset "Compare implicit/explicit errors and test mutation" begin
					gradient_vectors = [ones(length(data_set.λ)) for i in 1:length(gradients)]
					returned_gradient_vectors = Vector{eltype(gradient_vectors)}(undef, length(gradient_vectors))

					for i in 1:length(gradients)
						returned_gradient_vectors[i] = gradients[i](gradient_vectors[i], data_set.λ)
					end

					# Test equality for returned/mutated gradient vectors respectively.
					@test all_equal(returned_gradient_vectors)
					@test all_equal(gradient_vectors)
					
					# Because of respective equality above, only first elements of returned/mutated gradients need to be tested.
					@test all_equal([gradient_vectors[1], returned_gradient_vectors[1]])
				end

				# Because of the previously established equality, only gradients[1] needs to be tested.
				##################################################

				@testset "True parameter: local extrema" begin
					@test gradients[1](ones(length(data_set.λ)), data_set.λ) == zeros(length(data_set.λ))
				end



				@testset "Comparison to explicit formula" begin
					# Since partial derivatives have been tested above.
					∂ = partials[1]
					explicit = function(λ)
						return [∂[i](λ) for i in eachindex(data_set.λ) ]
					end 
					@test prod(test_objective_equality([λ-> gradients[1](ones(length(λ)),λ), explicit], data_set.λ))
				end
			end
		end
	end
end