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
    @assert length(X) == length(Xp) == length(cu) == length(cv)

    n, m = size(Q)
    nm = n * m
    q = length(X)
    d = [size(x, 2) for x in X]
    xtq = [x' * Q for x in X]

    # d2_u gives the second differences of Xp, by column
	r = size(first(Xp), 1)
	rm = r * m
    d2_u = d2grid(r)
    wu = [d2_u * x for x in Xp]
    wu2 = [b' * b for b in wu]
    Xp2 = [x' * x for x in Xp]

    # d2v * v gives the second differences of v (by column).
    d2v = d2grid(m)
    d2v2 = d2v' * d2v

    function f(pa::Params)::Float64

        # Squared error loss
        fv, _ = getfit(X, pa)
        f = sum(abs2, Q - fv) / nm

        # Penalty for u terms
        f0 = f
        for j in eachindex(X)
            f += cu[j] * sum(abs2, pa.v[:, j]) * sum(abs2, wu[j] * pa.beta[j]) / rm
        end

        # Penalty for columns of v
        for j = 1:size(pa.v, 2)
           u = Xp[j] * pa.beta[j]
           f += cv[j] * sum(abs2, u) * sum(abs2, d2v * pa.v[:, j]) / rm
        end

        return f
    end

    function g!(gr::Params, pa::Params; project = true)
        xb = [X[j] * pa.beta[j] for j = 1:q]

        # Gradient for loss function
        for j = 1:q
            gr.beta[j] .= -2 * xtq[j] * pa.v[:, j] / nm
            gr.v[:, j] .= -2 * xtq[j]' * pa.beta[j] / nm

            for k = 1:q
               if j == k
                   gr.beta[j] .+= 2 * sum(abs2, pa.v[:, j]) * X[j]' * xb[j] / nm
                   gr.v[:, j] .+= 2 * sum(abs2, xb[j]) * pa.v[:, j] / nm
               else
                   c = dot(pa.v[:, j], pa.v[:, k])
                   gr.beta[j] .+= c * X[j]' * X[k] * pa.beta[k] / nm
                   gr.v[:, j] .+= dot(xb[j], xb[k]) * pa.v[:, k] / nm
               end
           end
        end

        # Gradient contributions from the penalty for u terms
        for j in eachindex(X)
            gr.beta[j] .+= 2 * cu[j] * sum(abs2, pa.v[:, j]) * wu2[j] * pa.beta[j] / rm
            gr.v[:, j] .+= 2 * cu[j] * sum(abs2, wu[j] * pa.beta[j]) * pa.v[:, j] / rm
        end

        # Gradient contributions from the penalty for v
        for j = 1:q
            u = Xp[j] * pa.beta[j]
            gr.v[:, j] .+= 2 * cv[j] * sum(abs2, u) * d2v2 * pa.v[:, j] / rm
            gr.beta[j] .+= 2 * cv[j] * sum(abs2, d2v * pa.v[:, j]) * Xp2[j] * pa.beta[j] / rm
        end

        # Project
        if project
            for j = 1:q
                dd = sum(abs2, pa.v[:, j])
                gr.v[:, j] .-= dot(gr.v[:, j], pa.v[:, j]) * pa.v[:, j] / dd
            end
        end
    end

    return (f, g!)
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

function fitlr(
    X::Vector{Matrix{Float64}},
    Xp::Vector{Matrix{Float64}},
    Q::Matrix{Float64},
    cu::Vector{Float64},
    cv::Vector{Float64};
    args...,
)
    d = [size(x, 2) for x in X]
    m = size(Q, 2)
    f, g! = flr_fungrad(X, Xp, Q, cu, cv)

    p0 = getstart(X, Q)
    grad = copy_params(p0)

    f0 = f(p0)
    for itr = 1:1000
        g!(grad, p0; project = true)
        p1 = copy_params(p0)
        step = 0.5
        f1 = nothing
        success = false
        while step > 1e-18
            p1.v = p0.v - step * grad.v
            for j in eachindex(p0.beta)
                p1.beta[j] = p0.beta[j] - step * grad.beta[j]
            end
            f1 = f(p1)
            if f1 < f0
                success = true
                p0 = p1
                f0 = f1
                break
            end
            step /= 2
        end

        if !success
            println("Failed to find downhill step")
            break
        end
    end

    println(@sprintf("|grad|=%f", norm(grad.val)))

    return p0
end
