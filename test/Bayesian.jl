@testset "Posterior objectives" begin
	# Test all data sets, i.e. combination of multidimensional data/parameters.
	for data_set in data_sets
		@testset "$(data_set.name)" begin

			# FittingData to create objective functions from, that are numerically equal.
			# Comparing different specifications of distributions (always normal distributions)
			fitting_data = [FittingData(data_set.X,data_set.Y),
							FittingData(data_set.X,data_set.Y,distributions = normal_distribution),
							FittingData(data_set.X,data_set.Y,distributions = [normal_distribution for i in 1:length(data_set.X)])
							]

			model = ModelFunctions(data_set.model, partials = data_set.∂)

			posteriors = [posterior_objective(data,model) for  data in fitting_data]







			@testset "Posterior objective"	begin

				@testset "Consistency check is called" begin
					fd = deepcopy(fitting_data[1])
					fd.independent = [1]
					@test_throws DimensionMismatch posterior_objective(fd,model)
				end

				@testset "Equality of distribution specifications" begin
					@test test_objective_equality(posteriors,data_set.λ,)
				end


				# Because of the previously established equality of posterior functions, only posteriors[1] needs to be tested.
				##################################################


				@testset "Probabilities are positive" begin
					@test test_objective_positivity(posteriors[1],data_set.λ)
				end

				@testset "Probability monotonously decreasing" begin
					# direction <= means "worse" values are "smaller". 
					@test test_objective_monotony(posteriors[1],data_set.λ, <=)
				end

				@testset "True parameter probability" begin
					# Product of normal distributions at x=μ.
					@test posteriors[1](data_set.λ) ≈ 1/(sqrt(2*pi))^length(data_set.X)
				end

				@testset "Comparison to explicit formula" begin
					explicit = function(λ)
						m = [data_set.model(x,λ) for x in data_set.X] 
						return prod(@. exp(-(data_set.Y - m)^2/2)/sqrt(2*pi))
					end
					@test test_objective_equality([posteriors[1],explicit],data_set.λ)
				end
			end

















			# FittingData with distributions to replicate least squares.
			log_dist_fitting_data = [ FittingData(data_set.X,data_set.Y,distributions = (y,m,Δy) -> -(y-m)^2/Δy),
							FittingData(data_set.X,data_set.Y,distributions = [(y,m,Δy) -> -(y-m)^2/Δy for i in 1:length(data_set.X)])]

			log_posteriors = [log_posterior_objective(data,model) for  data in log_dist_fitting_data]

			
			@testset "Log-posterior objective" begin

				@testset "Consistency check is called" begin
					fd = deepcopy(fitting_data[1])
					fd.independent = [1]
					@test_throws DimensionMismatch log_posterior_objective(fd,model)
				end

				@testset "Equality of distribution specifications" begin
					@test test_objective_equality(log_posteriors,data_set.λ)
				end


				# Because of the previously established equality of log_posterior functions, only log_posteriors[1] needs to be tested.
				##################################################


				@testset "Log-probability monotonously decreasing" begin
					# direction <= means "worse" values are "smaller". 
					@test test_objective_monotony(log_posteriors[1],data_set.λ, <=)
				end

				@testset "True parameter probability" begin
					# log-distribution chosen, such that log_posterior = -lsq
					@test log_posteriors[1](data_set.λ) ≈ 0
				end

				@testset "Comparison to explicit formula" begin
					# Use lsq objective, which is already tested in Lsq.jl.
					explicit = lsq_objective(log_dist_fitting_data[1], model)
					@test test_objective_equality([λ-> -log_posteriors[1](λ),explicit],data_set.λ)
				end
			end
















			# FittingData for log-posterior partials with simpler logarithmic distributions.
			log_partials_fitting_data = [FittingData(data_set.X,data_set.Y,distributions = (y,m,Δy)-> -(y-m)^2/Δy),
										FittingData(data_set.X,data_set.Y,distributions = [(y,m,Δy)-> -(y-m)^2/Δy for i in 1:length(data_set.X)])]
			
			# Both methods to pass the partial derivatives of the log-distributions (single function and array of functions).
			distribution_partials = [(y,m,Δy) -> -2*(m-y)/Δy,[(y,m,Δy) -> -2*(m-y)/Δy for i in 1:length(data_set.X)]]
			
			log_partials = [log_posterior_partials(data,model,dist_partials) for data in log_partials_fitting_data for dist_partials in distribution_partials]


			@testset "Log-posterior derivatives" begin
				
				@testset "Test if consistency check is used" begin
					fd = deepcopy(log_partials_fitting_data[1])
					fd.independent = [1]
					@test_throws DimensionMismatch log_posterior_partials(fd,model,(y,m,Δy)-> 2*(m-y))
				end

				@testset "model.partials must not be nothing" begin
					@test_throws ArgumentError log_posterior_partials(log_partials_fitting_data[1],ModelFunctions((x,λ)-> λ*x), (y,m,Δy) -> 2*(m-y))
				end

				@testset "Length of distribution derivatives" begin
					@test_throws ArgumentError log_posterior_partials(log_partials_fitting_data[1], model,[(y,m,Δy) -> 2*(m-y) for i in 0:length(data_set.X)])
				end

				@testset "Length of prior derivatives" begin
					@test_throws ArgumentError log_posterior_partials(log_partials_fitting_data[1], model,(y,m,Δy) -> 2*(m-y), [λ -> λ for i in 0:length(model.partials)])
				end


				@testset "Equality of distribution specifications" begin
					@test prod(test_objective_equality([∂[i] for ∂ in log_partials], data_set.λ) for i in eachindex(data_set.λ))
				end


				# Because of the previously established equality, only log_partials[1] needs to be tested.

				@testset "True parameter: local extrema" begin
					# Distinguish between 1-dim and 2-dim parameters.
					@test [log_partials[1][i](data_set.λ) for i in eachindex(log_partials[1])] == zeros(length(log_partials[1]))
				end


				@testset "Comparison to explicit formula" begin
					# Use lsq partials, which is already tested in Lsq.jl.
					explicit = lsq_partials(log_partials_fitting_data[1],model)

					@test prod(test_objective_equality([λ-> -log_partials[1][i](λ), explicit[i]], data_set.λ) for i in eachindex(data_set.λ))
				end


				@testset "Prior derivative values" begin
					∂_p0 =  log_posterior_partials(log_partials_fitting_data[1], model,(y,m,Δy) -> 2*(m-y), [λ -> 1 for i in 1:length(model.partials)])

					# Because of local maximum without prior, only the prior derivatives contribute to the derivative.
					@test [∂_p0[i](data_set.λ) for i in eachindex(log_partials[1])] == ones(length(log_partials[1]))
				end
			end
















			log_gradients = [log_posterior_gradient(data,model,dist_partials) for data in log_partials_fitting_data for dist_partials in distribution_partials]


			@testset "Log-posterior Gradient" begin

				@testset "Test if consistency check is used" begin
					fd = deepcopy(log_partials_fitting_data[1])
					fd.independent = [1]
					@test_throws DimensionMismatch log_posterior_gradient(fd,model,(y,m,Δy)-> 2*(m-y))
				end

				@testset "model.partials must not be nothing" begin
					@test_throws ArgumentError log_posterior_gradient(log_partials_fitting_data[1],ModelFunctions((x,λ)-> λ*x), (y,m,Δy) -> 2*(m-y))
				end


				@testset "Length of distribution derivatives" begin
					@test_throws ArgumentError log_posterior_gradient(log_partials_fitting_data[1], model,[(y,m,Δy) -> 2*(m-y) for i in 0:length(data_set.X)])
				end



				@testset "Compare distribution specifications and test mutation" begin
					gradient_vectors = [ones(length(data_set.λ)) for i in 1:length(log_gradients)]
					returned_gradient_vectors = Vector{eltype(gradient_vectors)}(undef, length(gradient_vectors))

					for i in 1:length(log_gradients)
						returned_gradient_vectors[i] = log_gradients[i](gradient_vectors[i], data_set.λ)
					end

					# Test equality for returned/mutated gradient vectors respectively.
					@test all_equal(returned_gradient_vectors)
					@test all_equal(gradient_vectors)
					
					# Because of respective equality, only first elements of returned/mutated gradients need to be tested.
					@test all_equal([gradient_vectors[1], returned_gradient_vectors[1]])
				end

				# Because of the previously established equality of objective functions, only gradients[1] needs to be tested.
				##################################################

				@testset "True parameter: local extrema" begin
					@test log_gradients[1](ones(length(data_set.λ)), data_set.λ) == zeros(length(data_set.λ))
				end


				@testset "Comparison to explicit formula" begin
					# Since partial derivatives have been tested above.
					explicit = lsq_gradient(log_dist_fitting_data[1],model)
					@test prod(test_objective_equality([λ-> -log_gradients[1](ones(length(λ)),λ), λ-> explicit(ones(length(λ)), λ)], data_set.λ))
				end


				@testset "Prior derivative values" begin
					prior_gradient = function(gradient,λ) return gradient .+1 end
					gradient =  log_posterior_gradient(log_partials_fitting_data[1], model,(y,m,Δy) -> 2*(m-y), prior_gradient)
					# Because of local maximum without prior, only the prior derivatives contribute to the derivative.
					@test gradient(zeros(length(data_set.λ)), data_set.λ) == ones(length(data_set.λ))
				end
			end
		end
	end
end

