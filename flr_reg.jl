using LinearAlgebra, Printf
using UnicodePlots

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

'val' holds the actual parameter values, 'beta' and 'v'
are views into 'val'.
=#
mutable struct Params
    beta::Vector{AbstractArray{Float64,1}}
    v::AbstractArray{Float64,2}
    val::Vector{Float64}
end

function Params(beta::Vector{T}, v::AbstractMatrix) where {T<:AbstractVector}
    d = [length(b) for b in beta]
    m = size(v, 1)
    @assert size(v, 2) == length(d)

    val = zeros(sum(d) + m * length(d))
    beta1 = Vector{SubArray{Float64,1}}()
    ii = 0
    for j in eachindex(d)
        b = view(val, ii+1:ii+d[j])
        b .= beta[j]
        push!(beta1, b)
        ii += d[j]
    end
    v1 = view(val, ii+1:length(val))
    v1 = reshape(v1, m, length(d))
    v1 .= v
    return Params(beta1, v1, val)
end

#=
Copy the parameters
=#
function copy_params(pa::Params)::Params
    return Params([copy(b) for b in pa.beta], copy(pa.v))
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

# Returns a n - 2 x n matrix whose rows are the second differences
# at a sequence of offsets.
function d2grid(n)
    b = zeros(n - 2, n)
    for i = 1:n-2
        b[i, i:i+2] = [1, -2, 1]
    end
    return b
end

#=
Get starting values by using least squares
to obtain a fit that is not rank-restricted,
then approximating the least squares coefficients
with a rank one matrix.
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

        # Normalize
        f = norm(vv)
        vv ./= f
        uu .*= f
        push!(beta, uu)
        vm[:, j] = vv
        ii += a
    end

    return Params(beta, vm)
end

function updateG!(G, cu, d, X, d2X, v)
    f = length(d)
    n = size(first(X), 1)
    j1 = 0
    for k1 = 1:f
        j2 = 0
        for k2 = 1:f
            G[j1+1:j1+d[k1], j2+1:j2+d[k2]] .= X[k1]' * X[k2] * dot(v[:, k1], v[:, k2]) / n
            j2 += d[k2]
        end

        # Beta penalty
        G[j1+1:j1+d[k1], j1+1:j1+d[k1]] .+= cu[k1] * sum(abs2, v[:, k1]) * d2X[k1]
        j1 += d[k1]
    end
end

function updateC!(C, d, Q, X, v)
    f = length(d)
    n = size(first(X), 1)
    j = 0
    for k = 1:f
        C[j+1:j+d[k]] .= X[k]' * Q * v[:, k] / n
        j += d[k]
    end
end

function deriv2(x::Matrix{Float64})::Matrix{Float64}
    n, p = size(x)
    xd = zeros(n - 2, p)
    for j = 1:p
        for i = 1:n-2
            xd[i, j] = x[i, j] - 2 * x[i+1, j] + x[i+2, j]
        end
    end
    return xd
end

function normalizepar!(pa::Params)
    for j in eachindex(pa.beta)
        f = norm(pa.beta[j])
        pa.beta[j] ./= f
        pa.v[:, j] .*= f
    end
end

function updateV!(M, Q, X, d2, pa, cv)
    n = size(first(X), 1)
    m = size(pa.v, 1)
    f = length(pa.beta)
    ii = 0
    for k = 1:m
        for j = 1:f
            M[ii+1:ii+n, m*(j-1)+k] = X[j] * pa.beta[j]
        end
        ii += n
    end

    u, s, v = svd(M)
    xp = v * diagm(s .^ 2) * v'
    ii = 0
    for k = 1:f
        xp[ii+1:ii+m, ii+1:ii+m] .+= cv[k] * d2
        ii += m
    end

    pa.v .= reshape(xp \ (v * diagm(s) * u' * vec(Q)), m, f)
end

function fitlr(
    X::Vector{Matrix{Float64}},
    Xp::Vector{Matrix{Float64}},
    Q::Matrix{Float64},
    cu::Vector{Float64},
    cv::Vector{Float64};
    maxiter = 2000,
    args...,
)
    @assert all([size(Q, 1) == size(x, 1) for x in X])

    d = [size(x, 2) for x in X]
    n, m = size(Q)
    f = length(d)
    pa = getstart(X, Q)
    dd = sum(d)

    G = zeros(dd, dd)
    C = zeros(dd)
    M = zeros(n * m, m * f)

    # Second derivatives for beta
    D2x = [deriv2(x) for x in Xp]
    D2x = [size(b, 1) * b' * b for b in D2x]

    # Second derivatives for v
    d2 = d2grid(m)
    d2 = m * d2' * d2

    for itr = 1:maxiter
        p0 = copy_params(pa)

        # Update beta
        updateG!(G, cu, d, X, D2x, pa.v)
        updateC!(C, d, Q, X, pa.v)
        b = G \ C
        pa.val[1:length(b)] .= b

        # Update v
        updateV!(M, Q, X, d2, pa, cv)

        normalizepar!(pa)

        if norm(pa.val - p0.val) < 1e-10
            break
        end
    end

    return pa
end
