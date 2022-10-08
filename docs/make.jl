using Documenter, FittingObjectiveFunctions

makedocs(sitename="FittingObjectiveFunctions", pages = [
"FittingObjectiveFunctions"=>"index.md" ,
"FittingData and ModelFunctions"=>"fitting_data.md",
"Least squares objective"=>["Background"=>"lsq_background.md","How to implement" => "lsq_implementation.md"],
"Posterior probability"=> ["Background"=>"posterior_background.md", "How to implement"=>"posterior_implementation.md"],
"Logarithmic posterior probability"=>["Background"=>"log_posterior_background.md", "How to implement"=>"log_posterior_implementation.md"],
"API"=>  "API.md" ,
])