using Statistics, Roots, LinearAlgebra

#=
'x' is a matrix whose columns contain approximately iid replicate
esimates of a target (e.g. obtained via some form of bootstrapping).
Let 'mu' denote the target, let 'mu_hat' denote the row-wise mean of
'x', and let 's' denote the row-wise SD of 'x'.  The goal is to
construct an interval of the form 'mu_hat +/- k*s' so that with
probability 'p', all elements of 'mu' are contained within this band.
=#
function simulband(x::AbstractMatrix; p::Float64 = 0.95)

    # The center of the band
    c = mean(x, dims = 2)

    # The simultaneous confidence band takes the form c +/- k*s
    # for a constant k.
    s = std(x, dims = 2)

    # We want to find the root of this function to get the proper
    # empirical coverage.
    f = function (k::Float64)
        m = 0
        for v in eachcol(x)
            if all((v .< c + k * s) .& (v .> c - k * s))
                m += 1
            end
        end
        return m / size(x, 2) - p
    end

    # Lower bound of a bracket
    lb = 2.0
    while f(lb) > 0
        lb /= 2
    end

    # Upper bound of a bracket
    ub = 3.0
    while f(ub) < 0
        ub *= 2
    end

    # Find the zero
    k0 = find_zero(f, (lb, ub), Bisection(), rtol = 1e-8, atol = 1e-8)

    return (c, s, k0)
end

function test_simulband()

    # Sample size of observed data
    n = 50

    # Number of replications in the simulation study
    # to assess simultaneous coverage.
    nrep = 1000

    # Number of bootstrap replications used to construct
    # the confidence band.
    nboot = 1000

    # Storage
    xm = ones(n, 2)
    xb = zeros(n, nboot)

    # Empirical coverage
    cover = 0

    for k = 1:nrep

        # Generate observed data from a linear model.
        x = randn(n)
        tx = 1 .- 0.5 * x
        xm[:, 2] = x
        y = tx + 2 * randn(50)

        # Use OLS to estimate linear model parameters
        b = (xm' * xm) \ xm' * y
        yhat = xm * b
        resid = y - yhat
        rmse = sqrt(mean(resid .^ 2))

        # Sampling covariance matrix of the linear model
        # parameters.
        cm = rmse^2 * inv(xm' * xm)

        # Parametric bootstrap pseudo-replicate estimates
        for k = 1:nboot
            bb = b + cholesky(cm).L * randn(2)
            xb[:, k] = xm * bb
        end

        # Get the band
        c, s, k = simulband(xb)

        # Check if we have simultaneous coverage
        if all(tx .<= c + k * s) && all(tx .>= c - k * s)
            cover += 1
        end
    end

    println(cover / nrep)
end
