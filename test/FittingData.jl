@testset "FittingData struct" begin


	for data_set in data_sets
		@testset "2-argument constructor for $(data_set.name)" begin

			# Test that the data is assigned correctly
			fitting_data = FittingData(data_set.X,data_set.Y)
			@test fitting_data.independent == data_set.X
			@test fitting_data.dependent == data_set.Y
			@test fitting_data.errors == ones(length(data_set.X))

			# Default distributions should be an array of functions
			@test typeof(fitting_data.distributions) <: AbstractArray
			@test eltype(fitting_data.distributions) <: Function

			# A single function should be internally converted to an array of functions.
			fitting_data = FittingData(data_set.X,data_set.Y, distributions = normal_distribution )
			@test typeof(fitting_data.distributions) <: AbstractArray
			@test eltype(fitting_data.distributions) <: Function

			# Check that distributions array can be specified manually.
			fitting_data = FittingData(data_set.X,data_set.Y, distributions = [normal_distribution for i in 1:length(data_set.X)])
			@test typeof(fitting_data.distributions) <: AbstractArray
			@test eltype(fitting_data.distributions) <: Function
		end

		@testset "3-argument constructor for $(data_set.name)" begin

			# Test that the data is assigned correctly
			fitting_data = FittingData(data_set.X,data_set.Y, data_set.Σ)
			@test fitting_data.independent == data_set.X
			@test fitting_data.dependent == data_set.Y
			@test fitting_data.errors == data_set.Σ

			# Default distributions should be an array of functions
			@test typeof(fitting_data.distributions) <: AbstractArray
			@test eltype(fitting_data.distributions) <: Function

			# A single function should be internally converted to an array of functions.
			fitting_data = FittingData(data_set.X,data_set.Y, data_set.Σ, distributions = normal_distribution )
			@test typeof(fitting_data.distributions) <: AbstractArray
			@test eltype(fitting_data.distributions) <: Function

			# Check that distributions array can be specified manually.
			fitting_data = FittingData(data_set.X,data_set.Y, data_set.Σ, distributions = [normal_distribution for i in 1:length(data_set.X)])
			@test typeof(fitting_data.distributions) <: AbstractArray
			@test eltype(fitting_data.distributions) <: Function
		end
	end


	@testset "Test array checks" begin
		# Array checks should return an ArgumentError (with custom message), instead of a MethodError.

		# 2 arguments.
		@test_throws ArgumentError FittingData(:symbol,[1])
		@test_throws ArgumentError FittingData([1], :symbol)

		# 3 arguments.
		@test_throws ArgumentError FittingData(:symbol,[1],[1])
		@test_throws ArgumentError FittingData([1], :symbol, [1])
		@test_throws ArgumentError FittingData([1],[1], :symbol)
	end

	@testset "Test distribution type" begin
		# Distributions must be functions or an array of functions.
		@test_throws ArgumentError FittingData([1,2],[1,2], distributions = 1)
		@test_throws ArgumentError FittingData([1,2],[1,2], distributions = [x->x,1])
	end
	

	@testset "2-Argument constructor length check" begin
		# Iterate through all length combinations.
		for i in 1:2, j in 1:2, k in 1:2
			# Only test cases with at one length being different.
			if !(i==j==k)
				@test_throws DimensionMismatch FittingData(ones(i),ones(j), distributions = fill(normal_distribution,k))
			end
		end
	end

	@testset "3-Argument constructor length check" begin
		# Iterate through all length combinations.
		for i in 1:2, j in 1:2, k in 1:2, l in 1:2
			# Only test cases with at one length being different.
			if !(i==j==k==l)
				@test_throws DimensionMismatch FittingData(ones(i),ones(j),ones(k), distributions = fill(normal_distribution,l))
			end
		end
	end


	@testset "Warning if 0 in errors" begin
		@test_logs (:warn,) FittingData([1,2,3],[1,2,3],[1,0,1])
		@test_nowarn FittingData([1,1,1],[1,2,3],[1,1,1])
	end

end