include("Setup.jl")

@testset "FittingObjectiveFunctions.jl" begin
   include("FittingData.jl") 
   include("ModelFunctions.jl") 
   include("ConsistencyChecks.jl")
   include("Bayesian.jl")
   include("Lsq.jl")
end
