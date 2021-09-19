include("functional_lr.jl")

# Simulate data for testing.  The data are a tensor with r
# axes, each with length p.  The additive Gaussian error has
# standard deviation s.
function gen_tensor(p::Int, r::Int, s::Float64)

    q = r - 1
    di = fill(p, r)
    u = zeros(p, q)
    v = zeros(p, q)

    # Generate data with a random intercept and random
    # slope on this grid.
    grid = range(-1, 1, length = p)

    for j = 1:q
        ic, sl = randn(), randn()
        u[:, j] = ic .+ sl * grid
        ic, sl = randn(), randn()
        v[:, j] = ic .+ sl * grid

        # Center to make the representation identified
        if j > 1
            u[:, j] .-= mean(u[:, j])
        end

        # Scale to make the representation identified
        f = norm(u[:, j])
        u[:, j] /= f
        v[:, j] *= f

    end

    x = zeros(Float64, di...)
    for ii in product(Iterators.repeated(1:p, r)...)
        for k = 1:q
            x[ii...] += u[ii[k], k] * v[ii[end], k]
        end
    end

    # Additive noise
    x += s * randn(di...)

    return x, u, v

end

function check_reparameterize()

    p, q = 10, 4
    r = q + 1
    u = randn(p, q)
    v = randn(p, q)

    mu, u1, v1 = reparameterize(u, v)

    # Build a tensor from the first parameterization.
    di = fill(p, r)
    x1 = zeros(di...)
    for ii in product(Iterators.repeated(1:p, r)...)
        for k = 1:q
            x1[ii...] += u[ii[k], k] * v[ii[end], k]
        end
    end

    # Build a tensor from the second parameterization.
    x2 = zeros(di...)
    for ii in product(Iterators.repeated(1:p, r)...)
        x2[ii...] = mu[ii[end]]
        for k = 1:q
            x2[ii...] += u1[ii[k], k] * v1[ii[end], k]
        end
    end

    @assert maximum(abs.(x1 - x2)) < 1e-10

end

# This test is stochastic, but if the noise
# parameter is small it almost always passes.
function check_get_start()

    p = 10 # Number of grid points
    q = 3 # Number of covariates
    s = 0.01 # Additive noise

    # Rank of the tensor
    r = q + 1

    for i = 1:10

        x, u, v = gen_tensor(p, r, s)
        uh, vh = get_start(x[:], p, r)

        for j = 1:q
            # Correlate true and estimated values for
            # each component.
            m1 = u[:, j] * v[:, j]'
            m2 = uh[:, j] * vh[:, j]'
            @assert cor(vec(m1), vec(m2)) > 0.95
        end
    end

end

function check_grad_tensor_helper(; p = 10, r = 4, s = 1.0, e = 1e-5, pu = 10.0, pv = 10.0)

    x, u, v = gen_tensor(p, r, s)

    di = p * ones(Int, r)
    q = r - 1
    cu = pu * ones(q)
    cv = pv * ones(q)

    f, g! = _flr_fungrad_tensor(x[:], p, r, cu, cv)

    # Test the gradient at this point
    u0 = u + randn(p, q)
    v0 = v + randn(p, q)

    # Get the numerical gradient
    pa = vcat(u0[:], v0[:])
    f0 = f(pa)
    ng = zeros(length(pa))
    for i in eachindex(pa)
        pa[i] += e
        f1 = f(pa)
        ng[i] = (f1 - f0) / e
        pa[i] -= e
    end

    # Get the analytic gradient
    ag = zeros(length(pa))
    g!(ag, pa, project = false)

    d = maximum(abs.((ag - ng) ./ abs.(ng)))
    @assert d < 0.005

end

function check_grad_tensor()

    Random.seed!(3942)

    for r in [2, 4]
        for pe in [0.0, 5.0, 20.0]
            for j = 1:5
                check_grad_tensor_helper(p = 10, r = r, s = 1.0, e = 1e-5, pu = pe, pv = pe)
            end
        end
    end

end

function check_fit_tensor(; p = 10, r = 4, s = 1.0)

    s = 0.5

    # Number of covariates
    q = r - 1

    x, u, v = gen_tensor(p, r, s)

    # All parameters flattened
    pa = vcat(u[:], v[:])

    # Increasing levels of regularization
    for c in [100, 10000]
        cu = c * ones(q)
        cv = c * ones(q)
        uh, vh = fit_flr_tensor(x[:], p, r, cu, cv)

        for j = 1:q
            m1 = u[:, j] * v[:, j]'
            m2 = uh[:, j] * vh[:, j]'
            plt = scatterplot(vec(m1), vec(m2))
            println(plt)
        end
    end

end

# Fast tests
check_reparameterize()
check_get_start()

# Slow tests
check_fit_tensor()
check_grad_tensor()
