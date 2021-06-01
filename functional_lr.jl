using Optim, LinearAlgebra, IterTools, Statistics, Random
using UnicodePlots # for testing only

# Construct a smoothing penalty matrix for a vector of length p.
function _fw(p)
    a = zeros(p-2, p)
    for i in 1:p-2
        a[i, i:i+2] = [1 -2 1]
    end
    return a
end

# Split the parameter vector for the additive low-rank tensor model into components.
# If there are q covariates and the grid for each has length p, then the data is a
# tensor with r = q + 1 axes and p points along each axis.  The _split function
# reshapes the parameters into two p x q arrays, u and v, that are views
# into the input argument z.
function _split(z, p, q)
    @assert(length(z) % 2 == 0)
    m = div(length(z), 2)
    @assert(m % p == 0)
    u = reshape(@view(z[1:m]), p, q)
    v = reshape(@view(z[m+1:end]), p, q)
    return tuple(u, v)
end


function _flr_fungrad_tensor(x::Array{Float64,1}, p::Int, r::Int, cu::Array{Float64,1}, cv::Array{Float64,1})

    @assert length(x) == p^r

    # The number of covariates
    q = r - 1

    # The size of the tensor representation of the data along each axis
    di = p * ones(Int, r)

    # Convert the data to a tensor
    xr = reshape(x, di...)

    # Storage
    ft = zeros(di...)
    rs = zeros(di...)

    # The smoothing penalty matrix.
    w = _fw(p)
    w2 = w' * w

    # Fitted values
    function _getfit(u, v)
        ft .= 0
        for ii in product(Iterators.repeated(1:p, r)...)
            for k in 1:q
                ft[ii...] += u[ii[k], k] * v[ii[end], k]
            end
        end
    end

    # The fitting function (sum of squared residuals).
    f = function(z::Array{Float64,1})

        u, v = _split(z, p, q)

        _getfit(u, v)

        # Residuals
        rs .= xr .- ft

        # Sum of squared residuals
        rv = sum(abs2, rs)

        # Penalty terms
        for j in 1:q
            rv += cu[j] * sum(abs2, w*u[:,j]) / sum(abs2, u[:,j])
            rv += cv[j] * sum(abs2, w*v[:,j]) / sum(abs2, v[:,j])
        end

        return rv

    end

    # The gradient of the fitting function
    g! = function(G, z; project=true)

        # Split the parameters
        u, v = _split(z, p, q)

        _getfit(u, v)

        # Residuals
        rs .= xr .- ft

        # Sum of squared residuals
        rv = sum(abs2, rs)

        # Split the gradient
        G .= 0
        ug, vg = _split(G, p, q)

        # Penalty gradients
        for j in 1:q
            ssu = sum(abs2, u[:,j])
            nu = sum(abs2, w*u[:,j])
            ssv = sum(abs2, v[:,j])
            nv = sum(abs2, w*v[:,j])
            ug[:,j] = 2 .* cu[j] .* (ssu .* w2 * u[:,j] - nu .* u[:,j]) ./ ssu^2
            vg[:,j] = 2 .* cv[j] .* (ssv .* w2 * v[:,j] - nv .* v[:,j]) ./ ssv^2
        end

        # Loss gradient
        for ii in product(Iterators.repeated(1:p, r)...)
            f = -2 * rs[ii...]
            for j in 1:q
                ug[ii[j], j] += f * v[ii[end], j]
                vg[ii[end], j] += f * u[ii[j], j]
            end
        end

        if project
            for j in 1:q
                f = dot(u[:, j], u[:, j])
                ug[:, j] .-= dot(ug[:, j], u[:, j]) * u[:, j] / f
            end
        end
    end

    return tuple(f, g!)

end

function get_start(x::Array{Float64,1}, p::Int, r::Int)

    x = copy(x)

    q = r - 1
    u = zeros(p, q)
    v = zeros(p, q)
    z = zeros(p, p)

    di = p * ones(Int, r)
    xr = reshape(x, di...)

    for j in 1:q

        z .= 0
        for ii in product(Iterators.repeated(1:p, r)...)
            z[ii[j], ii[end]] += xr[ii...]
        end
        z .= z ./ p^(q-1)

        uu, ss, vv = svd(z)
        f = sqrt(ss[1])
        u[:, j] = f * uu[:, 1]
        v[:, j] = f * vv[:, 1]

        f = norm(u[:, j])
        u[:, j] /= f
        v[:, j] *= f

        for ii in product(Iterators.repeated(1:p, r)...)
            xr[ii...] -= u[ii[j], j] * v[ii[end], j]
        end

    end

    return u, v

end

function check_get_start()

    p = 5
    r = 4
    q = r - 1
    s = 0.0

    x, u, v = gen_tensor(p, r, s)
    uh, vh = get_start(x[:], p, r)

    for j in 1:q
        m1 = u[:, j] * v[:, j]'
        m2 = uh[:, j] * vh[:, j]'
        @assert maximum(abs.(m1 - m2)) < 1e-4
    end

end


# Fit a functional low-rank model to the tensor x.  The input data x is a flattened representation
# of a tensor with r axes, each having length p.
function fit_flr_tensor(x::Array{Float64,1}, p::Int, r::Int, cu::Array{Float64,1}, cv::Array{Float64,1}; start=nothing)

    # Number of covariates
    q = r - 1

    # Starting values
    if !isnothing(start)
        @assert length(start) == 2 * p * q
        u, v = _split(start, p, q)
    else
        u, v = get_start(x, p, r)
    end

    pa = vcat(u[:], v[:])

    f, g! = _flr_fungrad_tensor(x, p, r, cu, cv)

    r = optimize(f, g!, pa, LBFGS(), Optim.Options(iterations=100, show_trace=true))
    println(r)

    if !Optim.converged(r)
        println("fit_flr_tensor did not converge")
    end

    z = Optim.minimizer(r)
    uh, vh = _split(z, p, q)
    return tuple(uh, vh)

end


function gen_tensor(p::Int, r::Int, s::Float64)

    q = r - 1
    di = p * ones(Int, r)
    u = zeros(p, q)
    v = zeros(p, q)
    grid = range(-1, 1, length=p)

    for j in 1:q
        ic, sl = randn(), randn()
        u[:, j] = ic .+ sl*grid
        ic, sl = randn(), randn()
        v[:, j] = ic .+ sl*grid
        if j > 1
            u[:, j] .-= mean(u[:, j])
        end
        f = norm(u[:, j])
        u[:, j] /= f
        v[:, j] *= f
    end

    x = zeros(Float64, di...)

    for ii in product(Iterators.repeated(1:p, r)...)
        for k in 1:q
            x[ii...] += u[ii[k], k] * v[ii[end], k]
        end
    end

    x += s * randn(di...)

    return x, u, v

end

function check_grad_tensor_helper(;p=10, r=4, s=1.0, e=1e-5, pu=10.0, pv=10.0)

    x, u, v = gen_tensor(p, r, s)

    di = p*ones(Int, r)
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
    g!(ag, pa, project=false)

    d = maximum(abs.((ag - ng) ./ abs.(ng)))
    @assert d < 0.005

end

function check_grad_tensor()

    Random.seed!(3942)

    for r in [2, 4]
        for pe in [0.0, 5.0, 20.0]
            for j in 1:5
                check_grad_tensor_helper(p=10, r=r, s=1.0, e=1e-5, pu=pe, pv=pe)
            end
        end
    end

end

function check_fit_tensor(; p=10, r=4, s=1.0)

    s = 1.0

    # Number of covariates
    q = r - 1

    x, u, v = gen_tensor(p, r, s)

    # All parameters flattened
    pa = vcat(u[:], v[:])

    # Increasing levels of regularization
    for c in [100, 100, 10000]
        cu = c * ones(q)
        cv = c * ones(q)
        uh, vh = fit_flr_tensor(x[:], p, r, cu, cv)
        #uh, vh = get_start(x[:], p, r)

        for j in 1:q
            m1 = u[:, j] * v[:, j]'
            m2 = uh[:, j] * vh[:, j]'
            plt = scatterplot(vec(m1), vec(m2))
            println(plt)
        end
    end

end


check_fit_tensor()
check_get_start()
check_grad_tensor()
