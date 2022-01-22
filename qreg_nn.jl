using JuMP
using Tulip
using Random
using Statistics
using NearestNeighbors
using LightGraphs
using UnicodePlots
using MathOptInterface

# Implement the nonparametric quantile regression approach described here:
# https://arxiv.org/abs/2012.01758

# Representation of a fitted model
mutable struct Qreg

    # The outcome variable
    y::Array{Float64,1}

    # The covariates used to define the nearest neighbors
    x::Array{Float64,2}

    # Indices of the nearest neighbors
    nn::Array{Int,2}

    # A tree for finding neighbors
    kt::KDTree

    # The optimization model
    model::JuMP.Model

    # The fitted values from the most recent call to fit
    fit::Array{Float64,1}

    # The probability point for the most recent fit
    p::Float64

    # Retain these references from the optimization model
    rpos::Array{JuMP.VariableRef,1}
    rneg::Array{JuMP.VariableRef,1}
    dcap::Array{JuMP.VariableRef,2}
    rfit::Array{JuMP.VariableRef,1}

end

# Returns the degrees of freedom of the fitted model, which is the
# number of connected components of the graph defined by all edges
# of the covariate graph that are fused in the regression fit.
function degf(qr::Qreg; e = 1e-2)::Int

    nn = qr.nn
    fv = qr.fit
    g = SimpleGraph(size(nn, 1))

    for i = 1:size(nn, 1)
        for j = 1:size(nn, 2)
            if abs(fv[i] - fv[nn[i, j]]) < e
                add_edge!(g, i, nn[i, j])
            end
        end
    end

    return length(connected_components(g))

end

# Returns the BIC for the given fitted model.
function bic(qr::Qreg)::Tuple{Float64,Int}
    d = degf(qr)
    p = qr.p
    resid = qr.y - qr.fit
    pos = sum(x -> clamp(x, 0, Inf), resid)
    neg = -sum(x -> clamp(x, -Inf, 0), resid)
    check = p * pos + (1 - p) * neg
    sig = (1 - abs(1 - 2 * p)) / 2
    n = length(qr.y)
    return tuple(2 * check / sig + d * log(n), d)
end


# Predict the quantile at the point z using k nearest neighbors.
function predict(qr::Qreg, z::Array{Float64}; k = 5)::Float64

    ii, _ = knn(qr.kt, z, k)
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
    for i = 1:n
        e = 0.0
        for j = 1:r
            e = e + (x[i, j] - z[j]) / bw[j]
        end
        w = exp(-e * e / 2)
        xr[2:end] = x[i, :] - z
        xtx .= xtx + w * xr * xr'
        xty .= xty + w * f[i] * xr
    end

    b = xtx \ xty
    return b[1]

end

# Construct a structure that can be used to perform quantile regression
# of y on x.  k is the number of nearest neighbors used for regularization.
function qreg_nn(y::Vector{Float64}, x::Matrix{Float64}; k::Int = 5)::Qreg

    n = length(y)

    # Build the nearest neighbor tree, exclude each point from its own
    # neighborhood.
    kt = KDTree(x')
    nx, _ = knn(kt, x', k + 1, true)
    nn = hcat(nx...)'
    nn = nn[:, 2:end]

    model = Model(Tulip.Optimizer)

    # The estimated quantile for each row of the design matrix.
    @variable(model, rfit[1:n])

    # The residuals y - rfit are decomposed into their positive
    # and negative parts.
    rpos = @variable(model, rpos[1:n])
    rneg = @variable(model, rneg[1:n])

    # The distance between the fitted value of each point
    # and its nearest neighbor is bounded by dcap.
    dcap = @variable(model, dcap[1:n, 1:k])

    @constraint(model, rpos - rneg .== y - rfit)
    @constraint(model, rpos .>= 0)
    @constraint(model, rneg .>= 0)
    @constraint(model, dcap .>= 0)
    for j = 1:k
        @constraint(model, rfit - rfit[nn[:, j]] .<= dcap[:, j])
        @constraint(model, rfit[nn[:, j]] - rfit .<= dcap[:, j])
    end

    return Qreg(y, x, nn, kt, model, Array{Float64,1}(), -1, rpos, rneg, dcap, rfit)
end


# Estimate the p'th quantiles for the population represented by the data
# in qr. lam is a penalty parameter controlling the smoothness of the
# fit.
function fit(qr::Qreg, p::Float64, lam::Float64)::Array{Float64,1}

    @objective(qr.model, Min, sum(p * qr.rpos + (1 - p) * qr.rneg) + lam * sum(qr.dcap))

    optimize!(qr.model)
    if termination_status(qr.model) != MathOptInterface.OPTIMAL
        error("fit did not converge")
    end
    qr.fit = value.(qr.rfit)
    return qr.fit

end


# Search for a tuning parameter based on BIC.  Starting from
# lambda=0.1, increase the tuning parameter sequentially
# by a factor of 'fac'.  The iterations stop when the current
# BIC is greater than the previous BIC, or when the degrees of
# freedom is less than or equal to 'dof_min', or when the value of
# lambda is greater than 'lam_max'. The path is returned as an array
# of triples containing the tuning parameter value, the BIC
# value, and the degrees of freedom.
function bic_search(qr::Qreg, p::Float64; fac = 1.2, lam_max = 1e6, dof_min = 2)

    pa = []

    lam = 0.1
    while lam < lam_max
        _ = fit(qr, p, lam)
        b, d = bic(qr)
        push!(pa, [lam, b, d])
        if (d <= dof_min) || (length(pa) > 1 && b > pa[end-1][2])
            break
        end
        lam = lam * fac
    end

    la = minimum([x[2] for x in pa])

    return tuple(la, pa)

end


function check1()

    Random.seed!(342)
    n = 1000
    x = randn(n, 2)
    y = x[:, 1] + randn(n)

    qr1 = qreg_nn(y, x)
    qr2 = qreg_nn(y, x)
    qr3 = qreg_nn(y, x)

    yq = zeros(n, 3)
    yq[:, 1] = fit(qr1, 0.25, 0.1)
    yq[:, 2] = fit(qr2, 0.5, 0.1)
    yq[:, 3] = fit(qr3, 0.75, 0.1)

    ax = [-0.67, 0, 0.67] # True intercepts
    for j = 1:3
        c = cov(yq[:, j], x[:, 1])
        b = c / var(x[:, 1])
        a = mean(yq[:, j]) - b * mean(x[:, 1])
        @assert abs(b - 1) < 0.05 # True slope is 1
        @assert abs(a - ax[j]) < 0.1
    end

    bw = [1.0, 1.0]
    @assert abs(predict_smooth(qr1, [0.0, 0.0], bw) - ax[1]) < 0.1
    @assert abs(predict_smooth(qr2, [0.0, 0.0], bw) - ax[2]) < 0.1
    @assert abs(predict_smooth(qr3, [0.0, 0.0], bw) - ax[3]) < 0.1

end

function check2()

    Random.seed!(342)
    n = 1000
    x = randn(n, 2)
    y = x[:, 1] .^ 2 + randn(n)

    qr = qreg_nn(y, x)
    p = 0.5

    lam = 0.1:0.05:1
    b = zeros(length(lam))
    for (i, la) in enumerate(lam)
        _ = fit(qr, p, la)
        b[i] = bic(qr)
    end

    lineplot(lam, b)

end

function check3()

    Random.seed!(342)
    nrep = 20
    n = 1000
    p = 0.5
    for j = 1:nrep
        for k in [1, 2]

            x = randn(n, k)
            y = x[:, 1] .^ 2 + randn(n)

            qr = qreg_nn(y, x)

            la, pa = bic_search(qr, p, lam_max = 1e6)
            x = [z[2] for z in pa]
            x = x .- minimum(x)
            plt = lineplot(x[3:end])
            println(plt)

        end
    end

end
