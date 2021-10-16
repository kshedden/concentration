include("qreg_nn.jl")

# A low rank representation of a quantile model.
mutable struct QModel

    # The estimated quantiles.
    xr::AbstractArray

    # The estimated quantiles after removing the
    # central axis.
    xc::AbstractArray

    # The central axis.
    md::AbstractArray

    # The left 
    u::AbstractArray

    # The right factors of the low rank representation
    # of the quantiles
    v::AbstractArray

    xm::AbstractArray

    xmn::AbstractArray

    xsd::AbstractArray

    xna::AbstractArray

    cq::Any
    hq::Any
    bq::Any

    gcbs::Any
    gaht::Any
    gabm::Any
end

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

function mediation_prep(outcome, sex, age1, age2, cbs, m, vlc, vla, bw)

    # Probability points
    pg = collect(range(1 / m, 1 - 1 / m, length = m))

    # This dataframe combines (within subjects) a childhood (age1) record
    # and an adult (age2) record.
    dr = gendat(outcome, sex, age1, age2, vlc, vla)

    # The childhood body size is in column 3 of xm.
    yv, xm, xmn, xsd, xna = regmat(outcome, dr, vlc, vla)

    # Transform from raw coordinates to Z-score coordinates
    gcbs = x -> (x - xmn[3]) / xsd[3]
    gaht = x -> (x - xmn[4]) / xsd[4]
    gabm = x -> (x - xmn[5]) / xsd[5]

    # Marginal quantiles for all variables of interest at the target ages
    cq = marg_qnt(cbs, age1, sex).(pg)
    hq = marg_qnt(:Ht_Ave_Use, age2, sex).(pg)
    bq = marg_qnt(:BMI, age2, sex).(pg)

    # Z-scores of the marginal quantiles
    zcq = gcbs.(cq)
    zhq = gaht.(hq)
    zbq = gabm.(bq)

    # Estimate quantiles for blood pressure given exposure
    # and two mediators.
    xr = mediation_quantiles(yv, xm, pg, zcq, zhq, zbq, bw)

    # Remove the quantiles at the median exposure and predictors
    xc, md = center(xr)

    # Fit a low rank model to the estimated quantiles
    u, v = fit_flr_tensor(xc, fill(1.0, 3), fill(1.0, 3))

    return QModel(xr, xc, md, u, v, xm, xmn, xsd, xna, cq, hq, bq, gcbs, gaht, gabm)
end

function marginal_qf(med, xm, c, cq, gcbs, bw, pg)

    # Conditional quantiles of the mediators given childhood body size.  
    # The rows of medcq are probability points of the mediator.
    # The columns or medcq are probability points of the exposure.
    # The mediator is in raw units, not standardized.
    # The columns of xm are [age1, age2, exposure, med1, med2].
    # All columns of xm are standardized.
    qr = qreg_nn(med, xm)
    m = length(pg)
    medcq = zeros(m, m)
    v = zeros(3)
    for i = 1:m
        _ = fit(qr, pg[i], 0.1) # important tuning parameter here
        for j = 1:m
            # cq is the unstandardized quantile of the exposure, gcbs 
            # standardizes it
            v[3] = gcbs(cq[j])
            medcq[i, j] = predict_smooth(qr, v, bw)
        end
    end

    # cdfc[j] is the conditional CDF of the mediator for people with 
    # the exposure fixed at quantile pg[j]
    cdfc = []
    for j = 1:m
        ii = sortperm(medcq[:, j])
        local f = LinearInterpolation(medcq[ii, j], pg[ii], extrapolation_bc = Line())
        push!(cdfc, f)
    end

    # Calculate the marginal CDF of the mediator,
    # for a population in which the exposure has
    # been perturbed by c.  This amounts to averaging
    # all of the conditional CDFs with appropriate
    # weights.
    cdfm = function (x)
        sn = Normal()
        q = quantile(sn, pg)
        w = pdf(sn, q .- c) ./ pdf(sn, q)
        w ./= sum(w)
        mc = 0.0
        for i = 1:m
            mc += w[i] * cdfc[i](x)
        end
        return mc
    end

    # Invert the marginal cumulative probabilities to quantiles.
    x = collect(range(minimum(med), maximum(med), length = m))
    y = [cdfm(z) for z in x]
    ii = sortperm(y)
    qf = LinearInterpolation(y[ii], x[ii], extrapolation_bc = Line())
    return qf
end
