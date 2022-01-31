using JLD

include("functional_lr.jl")

xr = load("tensor_quantiles.jld")["tensor"]

xc, md = center(xr)

cu = 100.0 * ones(3)
cv = 100.0 * ones(3)
u, v = fit_flr_tensor(xc, cu, cv)
