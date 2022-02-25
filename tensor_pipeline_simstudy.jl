using Distributions, UnicodePlots, DataFrames

include("flr_tensor.jl")
include("flr_reg.jl")
include("qreg_nn.jl")

#=
Returns a tensor of simulated data (estimated quantiles)
based on a given sample size (n) and number of covariates
(p), using the provided functions to define the mean and
standard deviation for Gaussian values.  Also returns
the x-values corresponding to each point along the covariate
axes (gr) and the probability points corresponding to the
probability axis (the last axis of the tensor).
=#
function gendat(n, p, meanfunc, sdfunc)

    # Covariates
    xm = randn(n, p)

    # Conditional mean of response given covariates
    ey = meanfunc.(xm[:, 1])

    # Conditional SD of response given covariates
    sd = sdfunc.(xm[:, 2])

    # Observed data
    y = ey + sd .* randn(n)

    # Number of axes in the tensor
    r = p + 1

    # Number of points along each axis of the tensor
    m = 11

    # The gridpoints for all covariate axes
    gr = collect(range(-2, 2, length = m))

    # The gridpoints for the probability axis
    pp = collect(range(0.1, 0.9, length = m))

    # Fill in the tensor with estimated quantiles.
    # For speed construct a separate QNN for each
    # probability point.
    qr = [qreg_nn(y, xm) for _ = 1:m]
    for j in eachindex(pp)
        fit(qr[j], pp[j], 0.1)
    end

    # Create a tensor whose entries are the estimated
    # conditional quantiles at various covariate values
    # and probability points.
    di = fill(m, r)
    xx = zeros(di...)
    for ii in product(Iterators.repeated(1:m, r)...)
        # The covariate where we are predicting
        z = [gr[j] for j in ii[1:end-1]]
        xx[ii...] = predict(qr[ii[end]], z)
    end

    return xx, gr, pp
end

# Generate one dataset and fit the QNN/FLR pipeline
# to it.
function simstudy_tensor_run1(n, p, meanfunc, sdfunc)
    xx, gr, pp = gendat(n, p, meanfunc, sdfunc)
    m = first(size(xx))
    X, Xp, Y = setup_tensor(xx)
    q = length(X)
    p = size(X[1], 2)

    # Regularization
    cu = 1 * ones(p)
    cv = 1 * ones(p)

    # Estimate the central axis and remove
    # it from the quantiles.
    ca = mean(Y, dims = 1)[:]
    for j = 1:size(Y, 2)
        Y[:, j] .-= ca[j]
    end

    # Fit the low rank model to the quantiles.
    pa = fitlr(X, Xp, Y, cu, cv)

    # The true central axis
    ca0 = [quantile(Normal(0, 4), p) for p in pp]
    rmse_ca = sqrt(mean((ca - ca0) .^ 2))

    # Get the true u*v' layers of the additive decomposition.
    uv_true1 = meanfunc.(gr) * ones(m)'
    uv_true2 = zeros(length(gr), length(pp))
    for i in eachindex(gr)
        s = sdfunc(gr[i])
        for j in eachindex(pp)
            uv_true2[i, j] = quantile(Normal(0, s), pp[j]) - quantile(Normal(0, 4), pp[j])
        end
    end
    uv_true = [uv_true1, uv_true2]

    cor_uv, rmse_uv = Float64[], Float64[]
    for j = 1:q
        # The estimated rank-1 term for the j'th layer.
        uv_est = Xp[j] * pa.beta[j] * pa.v[:, j]'
        push!(cor_uv, cor(uv_est[:], uv_true[j][:]))
        push!(rmse_uv, sqrt(mean(uv_est[:] - uv_true[j][:]) .^ 2))
    end

    return rmse_ca, cor_uv, rmse_uv
end


n = 1500
p = 2
nrep = 3
sdfunc = x -> sqrt((4 + x)^2)
rslt = DataFrame(
    :mq => Int[],
    :p => Int[],
    :rmse_ca_mean => Float64[],
    :rmse_ca_sd => Float64[],
    :cor1_mean => Float64[],
    :cor1_sd => Float64[],
    :rmse1_mean => Float64[],
    :rmse1_sd => Float64[],
    :cor2_mean => Float64[],
    :cor2_sd => Float64[],
    :rmse2_mean => Float64[],
    :rmse2_sd => Float64[],
)
for mq = 1:3
    meanfunc = x -> x^mq
    rmse_ca, cor_uv, rmse_uv = [], [], []
    for j = 1:nrep
        rmse_ca1, cor_uv1, rmse_uv1 = simstudy_tensor_run1(n, p, meanfunc, sdfunc)
        push!(rmse_ca, rmse_ca1)
        push!(cor_uv, cor_uv1)
        push!(rmse_uv, rmse_uv1)
    end
    rmse_ca = hcat(rmse_ca...)
    cor_uv = hcat(cor_uv...)
    rmse_uv = hcat(rmse_uv...)
    row = [mq, p, mean(rmse_ca), std(rmse_ca)]
    for j = 1:2
        push!(row, mean(cor_uv[j, :]))
        push!(row, std(cor_uv[j, :]))
        push!(row, mean(rmse_uv[j, :]))
        push!(row, std(rmse_uv[j, :]))
    end
    push!(rslt, row)
end
