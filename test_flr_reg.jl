using Statistics, UnicodePlots

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
    s = 1.0
    d = [1, 2, 3]

    # The underlying true variable for each additive term
    X0 = [randn(n) for _ in d]

    # Basis functions for each additive term
    X, Xp = Matrix{Float64}[], Matrix{Float64}[]
    for (j, a) in enumerate(d)
        x, f = genbasis(X0[j], a - 1, s; linear = true)
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

function test_invariance()

    sigma = 2
    cux = 2.0
    cvx = 2.0
    X, X0, Xp, Q, pa = gendat(sigma)
    cu = [cux for _ = 1:length(X)]
    cv = [cvx for _ = 1:length(X)]

    d = [size(x, 2) for x in X]
    m = size(Q, 2)
    q = sum(d) + m * length(d)

    f, g! = flr_fungrad(X, Xp, Q, cu, cv)

    # Test the gradient at this point
    for k = 1:10
        pa = randparams(d, m)
        f0 = f(pa)
        g0 = copy_params(pa)
        g!(g0, pa)
        for b in pa.beta
            b .*= 2
        end
        pa.v ./= 2
        f1 = f(pa)
        g1 = copy_params(pa)
        g!(g1, pa)
        @assert abs(f1 - f0) < 1e-8
        for j in eachindex(pa.beta)
            br = g1.beta[j] ./ g0.beta[j]
            @assert maximum(abs, br .- 0.5) < 1e-5
        end
        @assert maximum(abs, g1.v ./ g0.v .- 2) < 1e-5
    end
end

function test_grad_helper(sigma, cux, cvx)

    X, X0, Xp, Q, pa = gendat(sigma)
    cu = [cux for _ = 1:length(X)]
    cv = [cvx for _ = 1:length(X)]

    d = [size(x, 2) for x in X]
    m = size(Q, 2)
    q = sum(d) + m * length(d)

    f, g! = flr_fungrad(X, Xp, Q, cu, cv)

    # Test the gradient at this point
    pa = randparams(d, m)

    # Get the numeric gradient
    e = 1e-8
    f0 = f(pa)
    ngr = copy_params(pa)
    for i = 1:q
        pa.val[i] += e
        ngr.val[i] = (f(pa) - f0) / e
        pa.val[i] -= e
    end

    # Get the analytic gradient
    gr = Params(pa.beta, pa.v)
    g!(gr, pa; project = false)

    return all(abs.(gr.val - ngr.val) .< 1e-4 .+ (1 + 1e-4) * abs.(ngr.val))
end

function test_grad()
    for sigma in [0.1, 1, 5, 10]
        for c in [0.0, 1, 100]
            b = 0
            nrep = 100
            for k = 1:nrep
                b += test_grad_helper(sigma, c, c)
            end
            println(b / nrep)
            #@assert b / nrep > 0.8
        end
    end
end

function test_fit_helper(sigma, cux, cvx)

    X, X0, Xp, Q, pt = gendat(sigma)
    cu = [cux for _ = 1:length(X)]
    cv = [cvx for _ = 1:length(X)]
    px = fitlr(X, Xp, Q, cu, cv; show_trace = false)
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
        println(j, " ", cor(vec(fv), vec(fv0)))
        @assert cor(vec(fv), vec(fv0)) > 0.8
    end

    # Check the overall fitted values.
    mse0 = mean((y - Q) .^ 2)
    mse1 = mean((y0 - Q) .^ 2)
    @assert abs(mse0 - sigma^2) / sigma^2 < 0.05
    @assert abs(mse1 - sigma^2) / sigma^2 < 0.05
end

function test_fit()
    for sigma in [1, 2, 5]
        for c in [0.0]
            test_fit_helper(sigma, c, c)
        end
    end
end

function test_regularize()

    for sigma in [1]
        X, X0, Xp, Q, pt = gendat(sigma)
        for c in [0.0, 1, 10, 100, 1000, 10000]
            cu = [c for _ = 1:length(X)]
            cv = [c for _ = 1:length(X)]
            px = fitlr(X, Xp, Q, cu, cv)
            println(px.beta[3])

            for j in [3] #1:size(px.v, 2)
                z = collect(range(minimum(X0[j]), maximum(X0[j]), length = 100))
                u = Xp[j] * px.beta[j]
                #plt = lineplot(px.v[:, 1])
                plt = lineplot(z, u)
                println("sigma=$(sigma), c=$(c), j=$(j)")
                println(plt)
            end
        end
    end
end

#test_params()
#test_invariance()
#test_grad()
#test_fit()
test_regularize()
