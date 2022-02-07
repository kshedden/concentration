using LinearAlgebra, Optim

#=
Fit a low rank regression model by minimizing

\|Y - \sum_j X_jb_jv_j'\|^2 + penalty

Y is a n \times q matrix, e.g. where q quantiles have been estimated
for each of n observations.  The column-wise mean of Y is the "central
axis" and the columns of Y are subsequently centered so that the
factors explain deviations of the quantiles from the central axis.

X_j is a n \times d_j matrix of explanatory variables or basis
functions for the j^th additive factor.  b_j is a d_j dimensional
vector of regression coefficients and v_j is a q-vector of loadings.

It is equivalent to view the problem as minimizing

\|Y - \sum_j X_jF_j\|^2 + penalty

where F_j is a rank-1 matrix.  If the X_j are concatenated
horizontally to produce X = [X_1 X_2 ...] and the F_j are concatenated
vertically to obtain F = [F_1' F_2'...]' then the objective function
can be written

\|Y - XF\|^2 + penalty.

This is similar to a standard low-rank regression, however we constrain
each F_j to have rank exactly 1 whereas a standard low-rank regression
would control the overall rank of F.
=#

#=
The parameters for optimization.

'beta' is a vector since the X_j may have differing numbers
of columns.

'v' is a matrix with q rows and a column corresponding to
each X_j.
=#
mutable struct Params
    beta::Vector{Vector{Float64}}
    v::Matrix{Float64}
end

#=
Allow comparison of two 'Params' values.
=#
function Base.isapprox(p::Params, q::Params)::Bool
    if length(p.beta) != length(q.beta)
        return false
    end
    if size(p.v) != size(q.v)
        return false
    end
    for j = 1:length(p.beta)
        if !isapprox(p.beta[j], q.beta[j])
            return false
        end
    end
    return isapprox(p.v, q.v)
end

#=
Pack the parameters into a vector.
=#
function joinparams(pa::Params)::Vector{Float64}
    n = sum([length(b) for b in pa.beta]) + prod(size(pa.v))
    z = zeros(n)
    ii = 0
    for v in pa.beta
        z[ii+1:ii+length(v)] = v
        ii += length(v)
    end
    z[ii+1:end] = vec(pa.v)
    return z
end

#=
Unpack the parameters from their vectorized form.
=#
function splitparams(d::Vector{Int}, m::Int, z::Vector{Float64})::Params
    pa = Params([zeros(a) for a in d], zeros(m, length(d)))
    ii = 0
    for b in pa.beta
        b .= z[ii+1:ii+length(b)]
        ii += length(b)
    end
    pa.v .= reshape(z[ii+1:end], size(pa.v)...)
    return pa
end

#=
The model is not identified so scale each column
of 'v' (each vector of loadings) to have unit norm.
=#
function normalize!(pa::Params)
    for j in eachindex(pa.beta)
        f = norm(pa.v[:, j])
        pa.v[:, j] /= f
        pa.beta[j] *= f
    end
end

#=
Returns a matrix containing the fitted values corresponding to
a given parameter value.
=#
function getfit(X::Vector{Matrix{Float64}}, pa::Params)
    n = size(first(X), 1)
    m = size(pa.v, 1)
    fv = zeros(n, m)
    u = Vector{Vector{Float64}}()
    for j in eachindex(pa.beta)
        u0 = X[j] * pa.beta[j]
        push!(u, u0)
        fv += u0 * pa.v[:, j]'
    end
    return (fv, u)
end

function d2mat_grid(n)
    b = zeros(n - 2, n)
    for i = 1:n-2
        b[i, i:i+2] = [1, -2, 1]
    end
    return b
end

#=
Generate functions that calculate the function value and gradient
of the objective function.
=#
function flr_fungrad(
    X::Vector{Matrix{Float64}},
    Xp::Vector{Matrix{Float64}},
    Q::Matrix{Float64},
    cu::Vector{Float64},
    cv::Vector{Float64},
)

    n = size(Q, 1)
    q = length(X)
    m = size(Q, 2)
    d = [size(x, 2) for x in X]
    xtq = [x' * Q for x in X]

    # d2_u gives the second differences of Xp, by column
    d2_u = d2mat_grid(size(first(Xp), 1))
    wu = [d2_u * x for x in Xp]
    wu2 = [b' * b for b in wu]

    # d2v * v gives the second differences of v (by column).
    d2v = d2mat_grid(m)
    d2v2 = d2v' * d2v

    function f(z::Vector{Float64})::Float64
        pa = splitparams(d, m, z)

        # Squared error loss
        fv, u = getfit(X, pa)
        f = sum(abs2, Q - fv)

        # Penalty for u terms
        for j in eachindex(X)
            f += cu[j] * sum(abs2, wu[j] * pa.beta[j])
        end

        # Penalty for columns of v
        for j = 1:size(pa.v, 2)
            f += cv[j] * sum(abs2, d2v * pa.v[:, j])
        end

        return f
    end

    function g!(G::Vector{Float64}, z::Vector{Float64})
        pa = splitparams(d, m, z)
        gr = Params([zeros(a) for a in d], zeros(m, length(d)))
        xb = [X[j] * pa.beta[j] for j in eachindex(pa.beta)]

        # Gradient for loss function
        for j = 1:q
            gr.beta[j] = -2 * xtq[j] * pa.v[:, j]
            gr.v[:, j] = -2 * xtq[j]' * pa.beta[j]

            for k = 1:q
                if j == k
                    gr.beta[j] += 2 * sum(abs2, pa.v[:, j]) * X[j]' * xb[j]
                    gr.v[:, j] += 2 * sum(abs2, xb[j]) * pa.v[:, j]
                else
                    c = dot(pa.v[:, j], pa.v[:, k])
                    gr.beta[j] += c * X[j]' * X[k] * pa.beta[k]
                    gr.v[:, j] += dot(xb[j], xb[k]) * pa.v[:, k]
                end
            end
        end

        # Gradient for beta penalty
        for j = 1:q
            gr.beta[j] += 2 * cu[j] * wu2[j] * pa.beta[j]
        end

        # Gradient for beta penalty
        for j = 1:q
            gr.v[:, j] += 2 * cv[j] * d2v2 * pa.v[:, j]
        end

        G .= joinparams(gr)
    end

    return (f, g!)
end

#=
Get starting values by using least squares
to obtain a fit that is not rank-restricted,
then approximating the least squares coefficients
with a wank one matrix.
=#
function getstart(X, Q)::Params

    d = [size(x, 2) for x in X]
    m = size(Q, 2)

    # Use OLS to regress each column of Q on the
    # horizontally concatenated X_j's.
    Xh = hcat(X...)
    br = zeros(size(Xh, 2), m)
    u, s, v = svd(Xh)
    for j = 1:size(Q, 2)
        br[:, j] = v * ((u' * Q[:, j]) ./ s)
    end

    # Each block of the least squares coefficients,
    # corresponding to one X_j, is approximated with
    # a rank one array.
    beta = Vector{Vector{Float64}}()
    vm = zeros(m, length(d))
    ii = 0
    for (j, a) in enumerate(d)

        # uu * vv' is a rank-1 approximation to the OLS
        # coefficients.
        u, s, v = svd(br[ii+1:ii+a, :])
        uu = u[:, 1]
        vv = s[1] * v[:, 1]

        # Normalize to out standard form.
        f = norm(vv)
        vv ./= f
        uu .*= f
        push!(beta, uu)
        vm[:, j] = vv
        ii += a
    end

    return Params(beta, vm)
end

function fitlr(
    X::Vector{Matrix{Float64}},
    Xp::Vector{Matrix{Float64}},
    Q::Matrix{Float64},
    cu::Vector{Float64},
    cv::Vector{Float64},
)

    d = [size(x, 2) for x in X]
    m = size(Q, 2)
    f, g! = flr_fungrad(X, Xp, Q, cu, cv)

    ps = getstart(X, Q)
    pa = joinparams(ps)

    # Gradient descent
    opt = Optim.Options(iterations = 50, show_trace = false)
    r = optimize(
        f,
        g!,
        pa,
        GradientDescent(linesearch = Optim.LineSearches.BackTracking()),
        opt,
    )

    # Conjugate gradient may need many iterations.
    opt = Optim.Options(iterations = 3000, show_trace = false)
    r = optimize(f, g!, Optim.minimizer(r), BFGS(), opt)

    pa = splitparams(d, m, Optim.minimizer(r))
    normalize!(pa)
    return pa
end
