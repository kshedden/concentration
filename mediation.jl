include("qreg_nn.jl")

# A low rank representation of a quantile model.
mutable struct QModel

	# The dataframe from which data are drawn
	dr::AbstractDataFrame

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

    # The design matrix for quantile regression.  The columns
    # are age1, age2, exposure, mediator 1, mediator 2.  Each
    # column is standardized.
    xm::AbstractArray

    # The means of the columns of xm, before standardizing.
    xmn::AbstractArray

    # The standard deviations of the columns of xm, before standardizing.
    xsd::AbstractArray

    # The names of the columns of xm.
    xna::AbstractArray

    # The quantiles of the exposure and mediators.
    qex::AbstractArray
    qm1::AbstractArray
    qm2::AbstractArray

    # Functions mapping from raw coordinates to Z-score coordinates.
    gex::Any
    gm1::Any
    gm2::Any
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
                    # v[1:2] always is zero, we are predicting
                    # at the target ages.
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

function mediation_prep(outcome, cbs, med1, med2, sex, age1, age2, m, vlc, vla, bw)

    # Probability points
    pg = collect(range(1 / m, 1 - 1 / m, length = m))

    # This dataframe combines (within subjects) a childhood (age1) record
    # and an adult (age2) record.
    dr = gendat(outcome, sex, age1, age2, vlc, vla)

    # The exposure is in column 3 of xm.
    yv, xm, xmn, xsd, xna = regmat(outcome, dr, vlc, vla)

    # Transform from raw coordinates to Z-score coordinates
    gex = x -> (x - xmn[3]) / xsd[3]
    gm1 = x -> (x - xmn[4]) / xsd[4]
    gm2 = x -> (x - xmn[5]) / xsd[5]

    # Marginal quantiles for all variables of interest at the target ages
    cq = marg_qnt(cbs, age1, sex).(pg)
    hq = marg_qnt(med1, age2, sex).(pg)
    bq = marg_qnt(med2, age2, sex).(pg)

    # Z-scores of the marginal quantiles
    zcq = gex.(cq)
    zhq = gm1.(hq)
    zbq = gm2.(bq)

    # Estimate quantiles for the outcome given exposure
    # and two mediators.
    xr = mediation_quantiles(yv, xm, pg, zcq, zhq, zbq, bw)

    # Remove the quantiles at the median exposure and predictors
    xc, md = center(xr)

    # Fit a low rank model to the estimated quantiles
    u, v = fit_flr_tensor(xc, fill(1.0, 3), fill(1.0, 3))

    return QModel(dr, xr, xc, md, u, v, xm, xmn, xsd, xna, cq, hq, bq, gex, gm1, gm2)
end

function marginal_qf(med, xm, c, cq, gex, bw, pg)

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
            # cq is the unstandardized quantile of the exposure, gex 
            # standardizes it
            v[3] = gex(cq[j])
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

mutable struct MediationResult{T<:Real}

	direct::Matrix{T}

	direct_score1::Vector{T}
	
	direct_score2::Vector{T}

	indirect1::Matrix{T}

	indirect2::Matrix{T}
end

function mediation(qrm::QModel)

    # Destructure (better syntax in 1.7)
    u, v = qrm.u, qrm.v
    xm, xmn, xsd = qrm.xm, qrm.xmn, qrm.xsd
    qex, qm1, qm2 = qrm.qex, qrm.qm1, qrm.qm2
    gex = qrm.gex

    # Estimate the direct effect
    sn = Normal()
    qn = quantile(sn, pg)
    pg1 = cdf(sn, qn .- 1)
    pg2 = cdf(sn, qn .+ 1)
    scoref = LinearInterpolation(pg, u[:, 1], extrapolation_bc = Line())
    score1 = [scoref(x) for x in pg1]
    score2 = [scoref(x) for x in pg2]
    de = (score2 - score1) * v[:, 1]'

    # Marginal quantile function of mediator 1 for perturbed populations
    # of the exposure.
    medh = xmn[4] .+ xsd[4] * xm[:, 4]
    q1f1 = marginal_qf(medh, xm[:, 1:3], -1.0, qex, gex, bw[1:3], pg)
    q1f2 = marginal_qf(medh, xm[:, 1:3], 1.0, qex, gex, bw[1:3], pg)

    # Marginal quantile function of mediator 2 for perturbed populations
    # of the exposure.
    medb = xmn[5] .+ xsd[5] * xm[:, 5]
    q2f1 = marginal_qf(medb, xm[:, 1:3], -1.0, qex, gex, bw[1:3], pg)
    q2f2 = marginal_qf(medb, xm[:, 1:3], 1.0, qex, gex, bw[1:3], pg)

    # Indirect effect through mediator 1
    scoref1 = LinearInterpolation(pg, u[:, 2], extrapolation_bc = Line())
    qm1f = LinearInterpolation(qm1, pg, extrapolation_bc = Line())
    q11 = [scoref1(qm1f(q1f1(p))) for p in pg]
    q12 = [scoref1(qm1f(q1f2(p))) for p in pg]
    ie1 = (q12 - q11) * v[:, 2]'

    # Indirect effect through mediator 2
    scoref2 = LinearInterpolation(pg, u[:, 3], extrapolation_bc = Line())
    qm2f = LinearInterpolation(qm2, pg, extrapolation_bc = Line())
    q21 = [scoref2(qm2f(q2f1(p))) for p in pg]
    q22 = [scoref2(qm2f(q2f2(p))) for p in pg]
    ie2 = (q22 - q21) * v[:, 3]'

	return MediationResult(de, score1, score2, ie1, ie2)
end
