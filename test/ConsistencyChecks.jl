@testset "consistency check" begin

	@testset "Mutating FittingData" begin
		model = ModelFunctions(x-> x)

		# Test length checks by mutating all fields individually.
		# fitting_data is reconstructed before each test.
		##################################################

		fitting_data = FittingData([1,2],[1,2])
		fitting_data.independent = [1]
		@test_throws DimensionMismatch consistency_check(fitting_data,model)

		fitting_data = FittingData([1,2],[1,2])
		fitting_data.dependent = [1]
		@test_throws DimensionMismatch consistency_check(fitting_data,model)

		fitting_data = FittingData([1,2],[1,2])
		fitting_data.errors = [1]
		@test_throws DimensionMismatch consistency_check(fitting_data,model)

		fitting_data = FittingData([1,2],[1,2])
		fitting_data.distributions = [normal_distribution]
		@test_throws DimensionMismatch consistency_check(fitting_data,model)


		# Test warning of zeros in errors.
		##################################################
		fitting_data = FittingData([1,2],[1,2])
		fitting_data.errors = [0,1]
		@test_logs (:warn,) consistency_check(fitting_data,model)
	end



	@testset "Argument checks" begin
		fitting_data = FittingData([1,2],[1,2],[1,2])

		# Model function evaluation should fail.
		@test_throws Any consistency_check(fitting_data, ModelFunctions(x->x),1)

		# Partial derivative evaluation should fail.
		@test_throws Any consistency_check(fitting_data,ModelFunctions((x,λ)-> λ*x, partials = [x->x]),1)
	end
end