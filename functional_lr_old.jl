

# Construct a smoothing penalty matrix for a vector of length p.
function _fw(p)
    a = zeros(p-2, p)
    for i in 1:p-2
        a[i, i:i+2] = [1 -2 1]
    end
    return a
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

#check_grad()
#check_fit()
