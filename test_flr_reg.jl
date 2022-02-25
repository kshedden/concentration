using Statistics, UnicodePlots, IterTools

include("flr_reg.jl")
include("basis.jl")

function randparams(d, m)
    return Params([randn(b) for b in d], randn(m, length(d)))
end

function test_params()
    d = [1, 2, 3]
    beta = [randn(n) for n in d]
    v = randn(4, length(beta))
    pa = Params(beta, v)
end

# Generate test data
function gendat(sigma)
    n = 1000
    m = 5
    bs = 1.0
    d = [1, 2, 3]

    # The underlying true variable for each additive term
    X0 = [randn(n) for _ in d]

    # Basis functions for each additive term
    X, Xp = Matrix{Float64}[], Matrix{Float64}[]
    for (j, a) in enumerate(d)
        x, f = genbasis(X0[j], a - 1, bs; linear = true)
        push!(X, x)
        z = collect(range(minimum(X0[j]), maximum(X0[j]), length = 100))
        push!(Xp, f(z))
    end

    # Generate coefficients so that the true score pattern is smooth
    beta = [0.1 * randn(a) for a in d]
    for j in eachindex(d)
        beta[j][1] += 1 # The first basis function is linear
        s = std(X[j] * beta[j])
        beta[j] /= s
    end

    # Generate loadings that are smooth
    v = zeros(m, length(d))
    for j = 1:size(v, 2)
        v0 = collect(range(-1, 1, m))
        v[:, j] = randn() .+ randn() * v0
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

function test_fit_als()

    for sigma in [1.0, 2]
        X, X0, Xp, Q, pt = gendat(sigma)
        for c in [0.0, 10]
            cu = c * ones(length(X))
            cv = c * ones(length(X))

            px = fitlr(X, Xp, Q, cu, cv)

            # Check the overall fit
            fv, _ = getfit(X, px)
            ss1 = sum(abs2, Q - fv)
            ss2 = sigma^2 * prod(size(Q))
            @assert((ss1 - ss2) / ss2 < 0.05)

            # Check the fit for each factor
            for j in eachindex(X)
                uv1 = X[j] * px.beta[j] * px.v[:, j]'
                uv2 = X[j] * pt.beta[j] * pt.v[:, j]'
                ss = sum(abs2, uv1 - uv2) / sum(abs2, uv2)
                @assert ss < 0.1
                @assert cor(vec(uv1), vec(uv2)) > 0.9
            end
        end
    end
end

function test_als_reg()

    for sigma in [10.0]
        X, X0, Xp, Q, pt = gendat(sigma)
        for c in [0.0, 1, 10, 100, 1000]
            cu = c * ones(length(X))
            cv = c * ones(length(X))
            px = fitlr(X, Xp, Q, cu, cv)

            for j in [3] #1:size(px.v, 2)
                println("sigma=$(sigma), c=$(c), j=$(j)")
                u = Xp[j] * px.beta[j]
                plt = lineplot(u)
                println("u$(j):")
                println(plt)
                plt = lineplot(px.v[:, j])
                println("v$(j):")
                println(plt)
            end
        end
    end
end

test_params()
test_fit_als()
test_als_reg()
