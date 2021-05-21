using Optim, LinearAlgebra, IterTools
using UnicodePlots # for testing only

# Construct a smoothing penalty matrix for a vector of length p.
function _fw(p)
    a = zeros(p-2, p)
    for i in 1:p-2
        a[i, i:i+2] = [1 -2 1]
    end
    return a
end

# Split the parameter vector for the multi-predictor model into components.
function _split(z)
    p = div(length(z), 6)
    ux = @view(z[1:p])
    vx = @view(z[p+1:2*p])
    um1 = @view(z[2*p+1:3*p])
    vm1 = @view(z[3*p+1:4*p])
    um2 = @view(z[4*p+1:5*p])
    vm2 = @view(z[5*p+1:6*p])
    return tuple(ux, vx, um1, vm1, um2, vm2)
end


function _flr_fungrad_multi(x::Array{Float64,4}, cu::Array{Float64,1}, cv::Array{Float64,1})

    p = size(x, 1)

    # Storage
    ft = zeros(p, p, p, p)
    rs = zeros(p, p, p, p)

    # The smoothing penalty matrix.
    w = _fw(p)
    w2 = w' * w

    # Fitted values
    function _getfit(ux, vx, um1, vm1, um2, vm2)
        for ii in product(Iterators.repeated(1:p, 4)...)
            ft[ii...] = ux[ii[1]]*vx[ii[4]] + um1[ii[2]]*vm1[ii[4]] + um2[ii[3]]*vm2[ii[4]]
        end
    end

    f = function(z::Array{Float64,1})

        ux, vx, um1, vm1, um2, vm2 = _split(z)

        _getfit(ux, vx, um1, vm1, um2, vm2)

        rs .= x .- ft # residuals

        rv = sum(abs2, rs)

        # Penalty terms
        rv = rv + cu[1] * sum(abs2, w*ux) / sum(abs2, ux)
        rv = rv + cu[2] * sum(abs2, w*um1) / sum(abs2, um1)
        rv = rv + cu[3] * sum(abs2, w*um2) / sum(abs2, um2)
        rv = rv + cv[1] * sum(abs2, w*vx) / sum(abs2, vx)
        rv = rv + cv[2] * sum(abs2, w*vm1) / sum(abs2, vm1)
        rv = rv + cv[3] * sum(abs2, w*vm2) / sum(abs2, vm2)

        return rv

    end


    g! = function(G, z; project=false)

        ux, vx, um1, vm1, um2, vm2 = _split(z)

        _getfit(ux, vx, um1, vm1, um2, vm2)

        rs .= x .- ft # residuals

        rv = sum(abs2, rs)

        uxg, vxg, um1g, vm1g, um2g, vm2g = _split(G)

        # Helper function for penalty gradients
        function h(u, v, gu, gv, cu, cv)
            ssu = sum(abs2, u)
            nu = sum(abs2, w*u)
            ssv = sum(abs2, v)
            nv = sum(abs2, w*v)
            gu .= 2 .* cu .* (ssu .* w2 * u - nu .* u) ./ ssu^2
            gv .= 2 .* cv .* (ssv .* w2 * v - nv .* v) ./ ssv^2
        end

        # Penalty gradients
        h(ux, vx, uxg, vxg, cu[1], cv[1])
        h(um1, vm1, um1g, vm1g, cu[2], cv[2])
        h(um2, vm2, um2g, vm2g, cu[3], cv[3])

        # Loss gradient
        for ii in product(Iterators.repeated(1:p, 4)...)
            r = -2 * rs[ii...]
            uxg[ii[1]] += r * vx[ii[4]]
            vxg[ii[4]] += r * ux[ii[1]]
            um1g[ii[2]] += r * vm1[ii[4]]
            vm1g[ii[4]] += r * um1[ii[2]]
            um2g[ii[3]] += r * vm2[ii[4]]
            vm2g[ii[4]] += r * um2[ii[3]]
        end

        if project

            proj = function(a, b)
                return a - dot(a, b) .* b / dot(b, b)
            end

            uxg .= proj(uxg, ux)
            um1g .= proj(um1g, um1)
            um2g .= proj(um2g, um2)

        end
    end

    return tuple(f, g!)

end

# Get starting values for the term u * v' in the decomposition of
# x, where u corresponds to the i^th axis of x and v corresponds to
# the fourth axis of x.
function get_start(x::Array{Float64,4}, i::Int)

    p = size(x, 1)
    z = zeros(p, p)

    # The indices that we average over.
    kk = [j for j in [1, 2, 3] if j != i]

    uu, vv = [], []
    for ii in product(Iterators.repeated(1:p, 2)...)

        for j1 in 1:p
            for j2 in 1:p
                jj = [0, 0, 0, j2]
                jj[i] = j1
                jj[kk[1]] = ii[1]
                jj[kk[2]] = ii[2]
                z[j1, j2] = x[jj...]
            end
        end

        u,s,v = svd(z)
        push!(uu, u[:, 1])
        push!(vv, v[:, 1])

    end

    uu = hcat(uu...)
    vv = hcat(vv...)

    u, _, _ = svd(uu)
    u = u[:, 1]
    v, _, _ = svd(vv)
    v = v[:, 1]

    return u, v

end

function check_get_start()

    p = 10
    u = range(-1, 1, length=10)
    u = u / norm(u)
    v = range(-1, 1, length=10).^2
    v = v / norm(v)

    x = zeros(p, p, p, p)
    for i in 1:p
        for j in 1:p
            x[i, :, j, :] = u * v' + 0.3*randn(p, p)
        end
    end

    uh, vh = get_start(x, 2)

    @assert abs(dot(u, uh)) > 0.95
    @assert abs(dot(v, vh)) > 0.95

end


# Fit a functional low-rank model to the matrix x.
function fit_flr_multi(x::Array{Float64,4}, cu::Array{Float64,1}, cv::Array{Float64,1}; start=nothing)

    p = size(x, 1)
    @assert size(x, 1) == size(x, 2) == size(x, 3) == size(x, 4)

    # Starting values
    if !isnothing(start)
        @assert length(start) == 6*p
        ux, vx, um1, vm1, um2, vm2 = _split(start)
    else
        ux, vx = get_start(x, 1)
        um1, vm1 = get_start(x, 2)
        um2, vm2 = get_start(x, 3)
    end

    pa = vcat(ux, vx, um1, vm1, um2, vm2)

    f, g! = _flr_fungrad_multi(x, cu, cv)

    r = optimize(f, g!, pa, LBFGS(), Optim.Options(iterations=100, show_trace=true))
    println(r)

    if !Optim.converged(r)
        println("fit_flr did not converge")
    end

    z = Optim.minimizer(r)
    uxh, vxh, um1h, vm1h, um2h, vm2h = _split(z)
    return tuple(uxh, vxh, um1h, vm1h, um2h, vm2h)

end


# Fit a rank-one approximation to a matrix x yielding x ~ u*v', where
# u and v are vectors.  The fit uses penalization so that u and v are
# smooth.  This only makes sense if u(i) is expected to be a smooth
# function of i, and similarly for v. The parameters cl and cr control
# the degree of smoothness in u and v, respectively.
function _flr_fungrad(x::Array{Float64,2}, cl::Float64, cr::Float64)

    p, q = size(x)

    # wl and wr are the smoothing penalty matrices for u and v.
    wl = _fw(p)
    wl2 = wl' * wl
    wr = _fw(q)
    wr2 = wr' * wr

    # Storage
    ft = zeros(p, q)
    rs = zeros(p, q)
    gl = zeros(p)
    gr = zeros(q)

    # The function to be minimized
    f = function(z)
        u = @view(z[1:p])     # left-side parameters
        v = @view(z[p+1:end]) # right-side parameters
        ft .= u * v'          # fitted values
        rs .= x .- ft         # residuals

        # Penalty terms
        ssu = sum(abs2, u)
        pl = sum(abs2, wl*u) / ssu
        ssv = sum(abs2, v)
        pr = sum(abs2, wr*v) / ssv

        return sum(abs2, rs) + cl*pl + cr*pr
    end

    # The gradient of f.  If project is false, the gradient is
    # calculated as if the parameters are free in R^{p+q}, if project
    # is true, the gradient is projected to S^p x R^q, which is the
    # natural domain for the parameters.  The gradient is stored in G.
    g! = function(G, z; project=true)
        u = @view(z[1:p])     # left-side parameters
        v = @view(z[p+1:end]) # right-side parameters
        ft .= u * v'          # fitted values
        rs .= x .- ft         # residuals

        ssu = sum(abs2, u)
        nl = sum(abs2, wl*u)
        ssv = sum(abs2, v)
        nr = sum(abs2, wr*v)
        gl .= 2 .* (ssu .* wl2 * u - nl .* u) ./ ssu^2
        gr .= 2 .* (ssv .* wr2 * v - nr .* v) ./ ssv^2

        G[1:p] = -2*rs*v + cl*gl       # left-side gradient
        G[p+1:end] = -2*rs'*u + cr*gr  # right-side gradient

        if project
            G[1:p] .= G[1:p] .- dot(G[1:p], u) .* u ./ dot(u, u)
        end
    end

    return tuple(f, g!)

end

# Fit a functional low-rank model to the matrix x.
function fit_flr(x, cl, cr)

    cl, cr = convert(Float64, cl), convert(Float64, cr)
    p, q = size(x)

    # Starting values
    u, s, v = svd(x)
    pa = sqrt(s[1]) .* vcat(u[:, 1],  v[:, 1])

    f, g! = _flr_fungrad(x, cl, cr)

    r = optimize(f, g!, pa, LBFGS())

    if !Optim.converged(r)
        println("fit_flr did not converge")
    end

    z = Optim.minimizer(r)
    return tuple(z[1:p], z[p+1:end])

end


function check_grad(;cl=1.0, cr=1.0, p=10, q=5, e=1e-5)

    u = range(1, stop=5, length=p)
    v = range(-2, stop=2, length=q)
    x = u * v' + 2.0 * randn(length(u), length(v))

    f, g! = _flr_fungrad(x, cl, cr)
    u0 = u + randn(length(u))
    v0 = v + randn(length(v))

    # Get the numerical gradient
    uv0 = vcat(u0, v0)
    f0 = f(uv0)
    ng = zeros(sum(size(x)))
    for i in 1:(p+q)
        uv0[i] = uv0[i] + e
        f1 = f(uv0)
        ng[i] = (f1 - f0) / e
        uv0[i] = uv0[i] - e
    end

    # Get the analytic gradient
    ag = zeros(p + q)
    g!(ag, uv0, project=false)

    d = maximum(abs.((ag - ng) ./ abs.(ng)))
    @assert d < 1e-3

end

function gen_multi(p, s)

    ux = range(1, stop=5, length=p)
    vx = range(-2, stop=2, length=p)
    um1 = range(1, stop=5, length=p)
    vm1 = range(-2, stop=2, length=p).^2
    um2 = range(1, stop=5, length=p)
    vm2 = range(1, stop=2, length=p)

    x = zeros(p, p, p, p)

    for ii in product(Iterators.repeated(1:p, 4)...)
        x[ii...] = ux[ii[1]]*vx[ii[4]] + um1[ii[2]]*vm1[ii[4]] + um2[ii[3]]*vm2[ii[4]]
    end

    x = x + s * randn(p, p, p, p)

    return x, ux, vx, um1, vm1, um2, vm2

end

function check_grad_multi(;p=10, s=1.0, e=1e-5)

    x, ux, vx, um1, vm1, um2, vm2 = gen_multi(p, s)

    cu = [10., 10., 10.]
    cv = [10., 10., 10.]

    f, g! = _flr_fungrad_multi(x, cu, cv)

    # Test the gradient at this point
    ux0 = ux + randn(p)
    vx0 = vx + randn(p)
    um10 = um1 + randn(p)
    vm10 = vm1 + randn(p)
    um20 = um2 + randn(p)
    vm20 = vm2 + randn(p)

    # Get the numerical gradient
    pa = vcat(ux0, vx0, um10, vm10, um20, vm20)
    f0 = f(pa)
    ng = zeros(6*p)
    for i in 1:6*p
        pa[i] = pa[i] + e
        f1 = f(pa)
        ng[i] = (f1 - f0) / e
        pa[i] = pa[i] - e
    end

    # Get the analytic gradient
    ag = zeros(6*p)
    g!(ag, pa, project=false)

    d = maximum(abs.((ag - ng) ./ abs.(ng)))
    @assert d < 0.005

end

function check_fit(;cl=1.0, cr=1.0, p=10, q=5, e=1e-5)

    u = range(1, stop=5, length=p)
    v = range(-2, stop=2, length=q)
    x = u * v' + 2.0 * randn(length(u), length(v))

    # With no penalization, the SVD gives the solution
    u0, s0, v0 = svd(x)
    u1, v1 = fit_flr(x, 0, 0)
    e = maximum(abs.(u1*v1' - s0[1]*u0[:, 1]*v0[:, 1]'))
    @assert e < 1e-9

    # Increasing levels of regularization
    for cl in [0.1, 1, 10, 100, 1000, 10000]
        u1, v1 = fit_flr(x, cl, 1)
        plt = lineplot(u1)
        println(plt)
    end

end

function check_fit_multi()

    p = 10
    s = 1.0
    x, ux, vx, um1, vm1, um2, vm2 = gen_multi(p, s)

    pa = vcat(ux, vx, um1, vm1, um2, vm2)

    # Increasing levels of regularization
    for c in [0.1, 1, 10, 100, 1000, 10000]
        cu = [c, c, c]
        cv = [c, c, c]
        uxh, vxh, um1h, vm1h, um2h, vm2h = fit_flr_multi(x, cu, cv)
        plt = scatterplot(ux, uxh)
        println(plt)
        plt = scatterplot(vx, vxh)
        println(plt)
        plt = scatterplot(um1, um1h)
        println(plt)
        plt = scatterplot(vm1, vm1h)
        println(plt)
        plt = scatterplot(um2, um2h)
        println(plt)
        plt = scatterplot(vm2, vm2h)
        println(plt)
    end

end


check_grad()
check_fit()
check_get_start()
check_grad_multi()
check_fit_multi()
