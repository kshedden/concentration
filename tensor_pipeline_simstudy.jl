#=
Use simulation to assess performance of the analysis pipeline
that uses QNN and FLR to understand the conditional quantile
structure of a population.
=#

using Distributions, UnicodePlots, DataFrames, CSV

include("flr_tensor.jl")
include("flr_reg.jl")
include("qreg_nn.jl")

#=
Simulate data (y, xm) and use it to fill in a tensor of conditional
quantiles, estimated using QNN.  The dataset has sample size 'n' and
'p' covariates, and has conditional mean structure given by
'meanfunc(x[1])' and conditional standard deviation given by
'sdfunc(x[2])'.  Also returns the x-values corresponding to each point
along the covariate axes (gr) and the probability points corresponding
to the probability axis (the last axis of the tensor).
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

    return xm, y, xx, gr, pp
end

# Generate one dataset and fit the QNN/FLR pipeline
# to it.
function simstudy_tensor_run1(n, p, meanfunc, sdfunc)
    xm, y, xx, gr, pp = gendat(n, p, meanfunc, sdfunc)
    m = first(size(xx))
    X, Xp, Y = setup_tensor(xx)
    q = length(X)
    p = size(X[1], 2)

    # Regularization
    cu = 1 * ones(p)
    cv = 1 * ones(p)

    # Estimate the central axis and remove
    # it from the quantiles.
    qr = qreg_nn(y, xm)
    ca = zeros(m)
    for (j, p) in enumerate(pp)
        fit(qr, p, 0.1)
        ca[j] = predict(qr, zeros(size(xm, 2)))
    end
    for j = 1:size(Y, 2)
        Y[:, j] .-= ca[j]
    end

    # Fit the low rank model to the quantiles.
    pa = fitlr(X, Xp, Y, cu, cv)

    # The true central axis
    ca0 = [quantile(Normal(0, sdfunc(0)), p) for p in pp]
    rrmse_ca = sqrt(mean((ca - ca0) .^ 2)) / norm(ca0)

    # Get the true u*v' layers of the additive decomposition.
    uv_true1 = meanfunc.(gr) * ones(m)'
    uv_true2 = zeros(length(gr), length(pp))
    for i in eachindex(gr)
        s = sdfunc(gr[i])
        for j in eachindex(pp)
            q1 = quantile(Normal(0, s), pp[j])
            q2 = quantile(Normal(0, sdfunc(0)), pp[j])
            uv_true2[i, j] = q1 - q2
        end
    end
    uv_true = [uv_true1, uv_true2]
    xo = zeros(length(gr), length(pp))
    while length(uv_true) < q
        push!(uv_true, xo)
    end

    cor_uv, rrmse_uv = Float64[], Float64[]
    for j = 1:q
        # The estimated rank-1 term for the j'th layer.
        uv_est = Xp[j] * pa.beta[j] * pa.v[:, j]'
        push!(cor_uv, cor(uv_est[:], uv_true[j][:]))
        push!(rrmse_uv, sqrt(mean(uv_est[:] - uv_true[j][:]) .^ 2) / norm(uv_true[j][:]))
    end

    return rrmse_ca, cor_uv, rrmse_uv
end

function simstudy_tensor_run(n, p, meanfunc, sdfunc, nrep)
    rmse_ca, cor_uv, rmse_uv = [], [], []
    for j = 1:nrep
        rmse_ca1, cor_uv1, rmse_uv1 = simstudy_tensor_run1(n, p, meanfunc, sdfunc)
        push!(rmse_ca, rmse_ca1)
        push!(cor_uv, cor_uv1)
        push!(rmse_uv, rmse_uv1)
    end
    return rmse_ca, cor_uv, rmse_uv
end

n = 1500
nrep = 10
sdfunc = x -> sqrt((4 + x)^2)

function main()

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

    # Consider monomial mean structures
    for mq = 1:3
        for p in [2, 4]
            meanfunc = x -> x^mq
            rmse_ca, cor_uv, rmse_uv = simstudy_tensor_run(n, p, meanfunc, sdfunc, nrep)

            # Assess accuracy of each factor.
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
    end

    return rslt
end

rslt = main()
CSV.write("writing/tensor_pipeline_simstudy.csv", rslt)
