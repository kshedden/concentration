using Distributions, Printf, Combinatorics

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
        sl = randn()
        u[:, j] = sl * grid
        ic, sl = randn(), randn()
        v[:, j] = ic .+ sl * grid

        # Scale to make the representation identified
        f = norm(u[:, j])
        u[:, j] /= f
        v[:, j] *= f
    end

    x = getfit(u, v)

    # Additive noise
    x += s * randn(di...)

    return x, u, v

end

function check_getfitresid()

    p, q = 10, 2
    r = q + 1
    u = randn(p, q)
    v = randn(p, q)

    # Calculate the fitted values directly without using iterators
    xf = zeros(p, p, p)
    for i1 = 1:p
        for i2 = 1:p
            for j = 1:p
                xf[i1, i2, j] = u[i1, 1] * v[j, 1] + u[i2, 2] * v[j, 2]
            end
        end
    end

    xf_ = getfit(u, v)

    @assert maximum(abs, xf - xf_) < 1e-8

end

# Make sure that the starting values are permutation-invariant.
function check_get_start_perm()

    Random.seed!(3942)

    p = 11  # Number of grid points
    q = 3   # Number of covariates
    s = 1.0 # Additive noise

    # Rank of the tensor
    r = q + 1

    x, u, v = gen_tensor(p, r, s)
    uh1, vh1 = nothing, nothing

    for ii in permutations([1:q]...)

        # Permute the first q axes of x.
        jj = copy(ii)
        push!(jj, r)
        xx = permutedims(x, jj)

        uh, vh = get_start(xx)
        if isnothing(uh1)
            uh1, vh1 = uh, vh
        else
            @assert maximum(abs.(uh - uh1[:, ii])) < 1e-8
            @assert maximum(abs.(vh - vh1[:, ii])) < 1e-8
        end
    end
end

function check_get_start()

    Random.seed!(3942)

    p = 11   # Number of grid points
    q = 3    # Number of covariates
    s = 0.01 # Additive noise

    # Rank of the tensor
    r = q + 1

    for i = 1:10

        x, u, v = gen_tensor(p, r, s)
        uh, vh = get_start(x)

        for j = 1:q
            # Correlate true and estimated values for
            # each component.
            m1 = u[:, j] * v[:, j]'
            m2 = uh[:, j] * vh[:, j]'
            cr = cor(vec(m1), vec(m2))
            if cr < 0.95
                error(@sprintf("cor(vec(m1), vec(m2)) = %f < 0.95", cr))
            end
        end
    end
end

function check_grad_tensor_helper(; p = 10, r = 4, s = 1.0, e = 1e-6, pu = 10.0, pv = 10.0)

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
    g!(ag, pa)

    return maximum(abs.((ag - ng) ./ abs.(ng)))
end

function check_grad_tensor()

    Random.seed!(3942)

    npass, nfail = 0, 0
    for r in [2, 4]
        for pe in [0.0, 5.0, 20.0]
            for j = 1:5
                d = check_grad_tensor_helper(
                    p = 10,
                    r = r,
                    s = 1.0,
                    e = 1e-5,
                    pu = pe,
                    pv = pe,
                )
                if d > 0.01
                    println("Failed: d=$(d), r=$(r), pe=$(pe), j=$(j)")
                    nfail += 1
                else
                    npass += 1
                end
            end
        end
    end

    println("Failed $(nfail) out of $(nfail + npass) gradient checks")
end

function check_fit_tensor_perm(; p = 10, r = 4, s = 1.0)

    Random.seed!(3942)

    s = 0.5

    # Number of covariates
    q = r - 1

    x, u, v = gen_tensor(p, r, s)

    # All parameters flattened
    pa = vcat(u[:], v[:])

    uh1, vh1 = nothing, nothing
    c = 0.0
    cu = c * ones(q)
    cv = c * ones(q)

    for ii in permutations([1:q]...)

        # Permute the first q axes of x.
        jj = copy(ii)
        push!(jj, r)
        xx = permutedims(x, jj)

        uh, vh = fit_flr_tensor(xx, cu, cv)
        if isnothing(uh1)
            uh1, vh1 = uh, vh
        else
            @assert maximum(abs.(uh - uh1[:, ii])) < 1e-4
            @assert maximum(abs.(vh - vh1[:, ii])) < 1e-4
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
    for c in [0, 100, 10000]
        cu = c * ones(q)
        cv = c * ones(q)
        uh, vh = fit_flr_tensor(x, cu, cv)

        for j = 1:q
			@assert abs(norm(vh[:, j]) - 1) < 1e-6
            m1 = u[:, j] * v[:, j]'
            m2 = uh[:, j] * vh[:, j]'
            plt = scatterplot(vec(m1), vec(m2))
            println(plt)
        end
    end
end

function check_exact()

    # This tensor has an exact low-rank structure
    m = 11
    pg = collect(range(1 / m, 1 - 1 / m, length = m))
    xr = zeros(m, m, m, m)
    for i1 = 1:m
        for i2 = 1:m
            for i3 = 1:m
                for j = 1:m
                    f = quantile(Normal(), pg[i1])
                    f += quantile(Normal(), pg[i2])
                    f += quantile(Normal(), pg[i3])
                    xr[i1, i2, i3, j] = quantile(Normal(f, 1), pg[j])
                end
            end
        end
    end

    xc, md = center(xr)

    u = ones(m, 3)
    for i1 = 1:m
        u[i1, :] .= quantile(Normal(), pg[i1])
    end
    v = ones(m, 3)

    ft = getfit(u, v)
    @assert maximum(abs, ft .- xc) < 1e-6
end

# Fast tests
check_exact()
check_getfitresid()
check_get_start()
check_get_start_perm()

# Slow tests
check_grad_tensor()
check_fit_tensor()
check_fit_tensor_perm()
