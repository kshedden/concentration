include("qreg_nn.jl")

function mediation_quantiles(yv, xm, pg, exq, m1q, m2q, bw)

    # Number of probability points to compute quantiles.
    m = length(pg)

    # The quantile regression model for y given an exposure
    # and two mediators.
    qr = qreg_nn(yv, xm)

    # Storage for estimated conditional quantiles.
    xr = zeros(m, m, m, m)

    # Storage for covariate vector.
    v = zeros(size(xm, 2))

    # Estimate the pg[j]'th quantile of y given the covariates.
    for j = 1:m

        _ = fit(qr, pg[j], 0.1) # important tuning parameter here

        for i1 = 1:m # Exposure
            for i2 = 1:m # First mediator
                for i3 = 1:m # Second mediator
                    v[3] = exq[i1]
                    v[4] = m1q[i2]
                    v[5] = m2q[i3]
                    xr[i1, i2, i3, j] = predict_smooth(qr, v, bw)
                end
            end
        end
    end

    return xr
end
