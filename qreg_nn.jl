using JuMP
using Tulip
using Random
using Statistics

# Implement the nonparametric quantile regression approach described here:
# https://arxiv.org/abs/2012.01758

# Find the positions of the k nearest neighbors to each row of x, in
# the L2 norm.
function _nn1(x::Array{Float64,2}, k::Int)::Array{Int,2}

    n, p = size(x)
    d = zeros(n)
    nn = zeros(Int, n, k)

    for i in 1:n
        for j in 1:n
            d[j] = 0
            for l in 1:p
                d[j] = d[j] + (x[i, l] - x[j, l])^2
            end
        end
	d[i] = Inf
        nn[i, :] = sortperm(d)[1:k]
    end

    return nn

end

# Representation of a fitted model
mutable struct Qreg
    x::Array{Float64,2}
    nn::Array{Int,2}
    p::Float64
    fit::Array{Float64}
end

# Predict the quantile at the point z using k nearest neighbors.
function predict(qr::Qreg, z::Array{Float64}; k=5)::Float64

    nn = zeros(Int, k)
    x = qr.x
    n, r = size(x)
    d = zeros(n)
    for i in 1:n
        for j in 1:r
            d[i] = d[i] + (x[i,j] - z[j])^2
        end
    end
    ii = sortperm(d)[1:k]
    return mean(qr.fit[ii])

end

# Predict the quantile at the point z using local linear regression.
function predict_smooth(qr::Qreg, z::Array{Float64}, bw::Array{Float64})::Float64

    f = qr.fit
    x = qr.x
    n, r = size(x)
    xtx = zeros(r + 1, r + 1)
    xty = zeros(r + 1)
    xr = ones(r + 1)
    for i in 1:n
        e = 0.0
        for j in 1:r
            e = e + (x[i,j] - z[j]) / bw[j]
        end
        w = exp(-e*e / 2)
	xr[2:end] = x[i, :] - z
        xtx .= xtx + w * xr * xr'
        xty .= xty + w * f[i] * xr
    end

    b = xtx \ xty
    return b[1]

end


# Estimate the p'th quantiles for the population represented by data
# y, x.  lam is a penalty parameter controlling the smoothness of the
# fit, k is the number of nearest neighbors used for regularization.
function qreg_nn(y::Array{Float64}, x::Array{Float64,2}, p, lam; k=5)::Qreg

    n = length(y)
    nn = _nn1(x, k)

    model = Model(Tulip.Optimizer)

    # The estimated quantile for each row of the design matrix.
    @variable(model, rfit[1:n])

    # The residuals y - rfit are decomposed into their positive
    # and negative parts.
    @variable(model, rpos[1:n])
    @variable(model, rneg[1:n])

    # The distance between the fitted value of each point
    # and its nearest neighbor is bounded by dcap.
    @variable(model, dcap[1:n, 1:k])

    @constraint(model, rpos - rneg .== y - rfit)
    @constraint(model, rpos .>= 0)
    @constraint(model, rneg .>= 0)
    @constraint(model, dcap .>= 0)
    for j in 1:k
        @constraint(model, rfit - rfit[nn[:,j]] .<= dcap[:, j])
        @constraint(model, rfit[nn[:,j]] - rfit .<= dcap[:, j])
    end
    @objective(model, Min, sum(p*rpos + (1-p)*rneg) + lam*sum(dcap))
    optimize!(model)
    println(termination_status(model))
    fv = value.(rfit)

    return Qreg(x, nn, p, fv)

end

function check1()

    Random.seed!(342)
    n = 1000
    x = randn(n, 2)
    y = x[:, 1] + randn(n)

    qr1 = qreg_nn(y, x, 0.25, 0.1)
    qr2 = qreg_nn(y, x, 0.5, 0.1)
    qr3 = qreg_nn(y, x, 0.75, 0.1)

    yq = zeros(n, 3)
    yq[:, 1] = qr1.fit
    yq[:, 2] = qr2.fit
    yq[:, 3] = qr3.fit

    ax = [-0.67, 0, 0.67] # True intercepts
    for j in 1:3
        c = cov(yq[:,j], x[:, 1])
        b = c / var(x[:, 1])
        a = mean(yq[:,j]) - b*mean(x[:,1])
        @assert abs(b - 1) < 0.05 # True slope is 1
        @assert abs(a - ax[j]) < 0.1
    end

    bw = [1., 1.]
    @assert abs(predict_smooth(qr1, [0., 0.], bw) - ax[1]) < 0.1
    @assert abs(predict_smooth(qr2, [0., 0.], bw) - ax[2]) < 0.1
    @assert abs(predict_smooth(qr3, [0., 0.], bw) - ax[3]) < 0.1

end
