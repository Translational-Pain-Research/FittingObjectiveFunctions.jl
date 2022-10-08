@testset "ModelFunctions struct" begin

		@testset "Constructor without partial derivatives" begin
			model = ModelFunctions((x,λ)-> λ* x)

			# Test field types.
			@test typeof(model.model) <: Function
			@test isnothing(model.partials)
		end

		@testset "Constructor with partial derivatives" begin
			model = ModelFunctions((x,λ)-> λ* x, partials = [(λ,x)-> x])

			# Test field types.
			@test typeof(model.model) <: Function
			@test typeof(model.partials) <: AbstractArray
			@test eltype(model.partials) <: Function
		end

		@testset "Partial derivatives checks" begin
			# Partials must be an array of functions (or nothing).
			@test_throws ArgumentError ModelFunctions(x->x, partials =  x->x)
			@test_throws ArgumentError ModelFunctions(x->x, partials = [[x-> x], [x-> x]])
		end

end
