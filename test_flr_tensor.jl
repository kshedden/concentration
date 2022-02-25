using Distributions, Printf, UnicodePlots

include("flr_tensor.jl")
include("flr_reg.jl")

#=
Simulate data for testing.  The data are a tensor with r
axes, each with length p.  The additive Gaussian error has
standard deviation s.
=#
function gen_tensor(p::Int, r::Int, s::Float64)

    q = r - 1
    di = fill(p, r)
    u = zeros(p, q)
    v = zeros(p, q)

    # Generate data with a random intercept and random
    # slope on this grid.
    grid = range(-1, 1, length = p)

    # The central axis
    ca = collect(2 * grid)

    for j = 1:q
        # u is linear with no intercept
        sl = randn()
        u[:, j] = sl * grid

        # v is linear
        ic, sl = randn(), randn()
        v[:, j] = ic .+ sl * grid
        v[:, j] ./= norm(v[:, j])
    end

    x = getfit_tensor(u, v, ca)

    # Additive noise
    x += s * randn(di...)

    return x, ca, u, v
end

function test_flr_tensor1()

    p = 11
    p2 = div(p, 2) + 1
    r = 4
    q = r - 1
    s = 0.2
    x0, ca0, u0, v0 = gen_tensor(p, r, s)

    X, Xp, Y = setup_tensor(x0)
    cu = 100 * ones(p)
    cv = 100 * ones(p)

    # Remove the central axis
    ca = mean(Y, dims = 1)[:]
    @assert isapprox(ca, ca0, atol = 1e-2, rtol = 1e-2)
    for j = 1:p
        Y[:, j] .-= ca[j]
    end

    pa = fitlr(X, Xp, Y, cu, cv)
    fv, _ = getfit(X, pa)
    rmse = sqrt(mean((fv - Y) .^ 2))
    @assert isapprox(s, rmse, atol = 1e-2, rtol = 1e-2)

    for j = 1:q
        # Expand the parameters to over the fixed zero
        # at the midpoint
        u = zeros(p)
        u[1:p2-1] = pa.beta[j][1:p2-1]
        u[p2+1:end] = pa.beta[j][p2:end]

        uv_est = u * pa.v[:, j]'
        uv_true = u0[:, j] * v0[:, j]'
        @assert cor(vec(uv_est), vec(uv_true)) > 0.95
    end
end

test_flr_tensor1()
