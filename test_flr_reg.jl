using Statistics

include("flr_reg.jl")

# Make sure we can round-trip the packing and unpacking of
# parameters.
function test_split_join()
    d = [2, 3, 4]
    m = 4
    b = [[1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0]]
    v = randn(m, length(d))
    pa = Params(b, v)
    z = joinparams(pa)
    pa1 = splitparams(d, m, z)
    @assert isapprox(pa, pa1)
end

function genbasis(x, b, s)

    # Basis centers
    c = if b == 1
        [mean(x)]
    else
        range(minimum(x), maximum(x), length = b)
    end

    B = zeros(length(x), b)
    fl = []
    for j = 1:b
        f = function (x)
            y = (x .- c[j]) / s
            return exp.(-y .^ 2 / 2)
        end
        B[:, j] = f(x)
        push!(fl, f)
    end

    function g(z)
        x = zeros(length(z), length(fl))
        for j in eachindex(fl)
            x[:, j] = fl[j](z)
        end
        return x
    end

    return B, g
end

# Generate test data
function gendat(sigma)
    n = 1000
    m = 5
    s = 1.0
    d = [1, 2, 3]

    # The underlying true variable for each additive term
    X0 = [randn(n) for _ in d]

    # Basis functions for each additive term
    X, Xp = Matrix{Float64}[], Matrix{Float64}[]
    for (j, a) in enumerate(d)
        x, f = genbasis(X0[j], a, s)
        push!(X, x)
        z = collect(range(minimum(X0[j]), maximum(X0[j]), length = 100))
        push!(Xp, f(z))
    end
    beta = [randn(a) for a in d]
    for j in eachindex(d)
        s = std(X[j] * beta[j])
        beta[j] /= s
    end
    v = randn(m, length(d))
    for j = 1:size(v, 2)
        v[:, j] ./= norm(v[:, j])
    end
    q = sum(d) + m * length(d)

    Q = zeros(n, m)
    for j in eachindex(d)
        Q .+= X[j] * beta[j] * v[:, j]'
    end
    Q .+= sigma * randn(n, m)
    return (X, X0, Xp, Q, Params(beta, v))
end

function test_grad_helper(cux, cvx)

    sigma = 2.0
    X, X0, Xp, Q, pa = gendat(sigma)
    cu = [cux for _ = 1:length(X)]
    cv = [cvx for _ = 1:size(Q, 2)]

    d = [size(x, 2) for x in X]
    m = size(Q, 2)
    q = sum(d) + m * length(d)

    f, g! = flr_fungrad(X, Xp, Q, cu, cv)

    # Test the gradient at this point
    z = randn(q)

    # Get the numeric gradient
    e = 1e-8
    f0 = f(z)
    ngr = zeros(q)
    for i in eachindex(z)
        z[i] += e
        ngr[i] = (f(z) - f0) / e
        z[i] -= e
    end

    # Get the analytic gradient
    gr = zeros(q)
    g!(gr, z)

    return mean(abs.(gr - ngr) .< 1e-4 .+ (1 + 1e-4) * abs.(ngr))
end

function test_grad()
    for c in [0.0, 1.0]
        b = 0
        nrep = 100
        for k = 1:nrep
            b += test_grad_helper(c, c)
        end
        @assert b / nrep > 0.9
    end
end

function test_fit()

    sigma = 2
    X, X0, Xp, Q, pt = gendat(sigma)
    cu = [0.0 for _ = 1:length(X)]
    cv = [0.0 for _ = 1:size(Q, 2)]
    px = fitlr(X, Xp, Q, cu, cv)
    f, g! = flr_fungrad(X, Xp, Q, cu, cv)

    # The fitted values for the estimated and
    # true models.
    y = zeros(size(Q)...)
    y0 = zeros(size(Q)...)
    for j in eachindex(px.beta)
        fv = X[j] * px.beta[j] * px.v[:, j]'
        y += fv
        fv0 = X[j] * pt.beta[j] * pt.v[:, j]'
        y0 += fv0

        # Check the fitted values for each additive
        # term.
        @assert cor(vec(fv), vec(fv0)) > 0.8
    end

    # Check the overall fitted values.
    mse0 = mean((y - Q) .^ 2)
    mse1 = mean((y0 - Q) .^ 2)
    @assert abs(mse0 - sigma^2) / sigma^2 < 0.05
    @assert abs(mse1 - sigma^2) / sigma^2 < 0.05
end

test_split_join()
test_grad()
test_fit()
