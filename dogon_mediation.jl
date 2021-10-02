using DataFrames, GZip, CSV, Printf, Statistics, Distributions, UnicodePlots, Interpolations

include("qreg_nn.jl")
include("functional_lr.jl")
include("dogon_utils.jl")
include("mediation.jl")

df = GZip.open("/home/kshedden/data/Beverly_Strassmann/Cohort_2021.csv.gz") do io
    CSV.read(io, DataFrame)
end

# Analyze one sex, at specific childhood and adult ages
sex = "Female"
age1 = 5.0
age2 = 20.0

# Childhood body size variable (primary exposure)
# Possibilities: Ht_Ave_Use, WT, HAZ, WAZ
cbs = :HAZ

# Number of quantile points to track
m = 5

# Probability points
pg = collect(range(1 / m, 1 - 1 / m, length = m))

# Marginal quantiles for all variables of interest at the target ages
cq = marg_qnt(cbs, age1, sex).(pg)
hq = marg_qnt(:Ht_Ave_Use, age2, sex).(pg)
bq = marg_qnt(:BMI, age2, sex).(pg)

# Child childhood body size variable.
vl1 = [vs(cbs, missing, Inf)]

# Adult body size variables.
vl2 = [vs(:Ht_Ave_Use, missing, Inf), vs(:BMI, missing, Inf)]

# This dataframe combines (within subjects) a childhood (age1) record
# and an adult (age2) record.
dr = gendat(sex, age1, age2, vl1, vl2)

# The childhood body size is in column 3 of xm.
yv, xm, xmn, xsd, xna = regmat(:SBP, dr, vl1, vl2)

# Transform from raw coordinates to Z-score coordinates
gcbs = x -> (x - xmn[3]) / xsd[3]
gaht = x -> (x - xmn[4]) / xsd[4]
gabm = x -> (x - xmn[5]) / xsd[5]

# Z-scores of the marginal quantiles
zcq = gcbs.(cq)
zhq = gaht.(hq)
zbq = gabm.(bq)

# Bandwidth parameters for quantile smoothing
bw = fill(1.0, size(xm, 2))

# Estimate quantiles for blood pressure given exposure
# and two mediators.
xr = mediation_quantiles(yv, xm, pg, zcq, zhq, zbq, bw)

# Remove the quantiles at the median exposure and predictors
xc, md = center(xr)

# Fit a low rank model to the estimated quantiles
u, v = fit_flr_tensor(xc, fill(1.0, 3), fill(1.0, 3))

# Estimate the direct effect
sn = Normal()
qn = quantile(sn, pg)
pg1 = cdf(sn, qn .- 1)
pg2 = cdf(sn, qn .+ 1)
scoref = LinearInterpolation(pg, u[:, 1], extrapolation_bc = Line())
score1 = [scoref(x) for x in pg1]
score2 = [scoref(x) for x in pg2]
de = (score2 - score1) .* v[:, 1]

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

# Marginal quantile function of height for perturbed populations
# of childhood HAZ.
medh = xmn[4] .+ xsd[4] * xm[:, 4]
qfh1 = marginal_qf(medh, xm[:, 1:3], -1.0, cq, gcbs, bw[1:3], pg)
qfh2 = marginal_qf(medh, xm[:, 1:3], 1.0, cq, gcbs, bw[1:3], pg)

# Marginal quantile function of BMI for perturbed populations
# of childhood HAZ.
medb = xmn[5] .+ xsd[5] * xm[:, 5]
qfb1 = marginal_qf(medb, xm[:, 1:3], -1.0, cq, gcbs, bw[1:3], pg)
qfb2 = marginal_qf(medb, xm[:, 1:3], 1.0, cq, gcbs, bw[1:3], pg)

# Indirect effect through height
scorefh = LinearInterpolation(pg, u[:, 2], extrapolation_bc = Line())
hqf = LinearInterpolation(hq, pg, extrapolation_bc = Line())
qh1 = [scorefh(hqf(qfh1(p))) for p in pg]
qh2 = [scorefh(hqf(qfh2(p))) for p in pg]
ieh = (qh2 - qh1) .* v[:, 2]

# Indirect effect through BMI
scorefb = LinearInterpolation(pg, u[:, 3], extrapolation_bc = Line())
bqf = LinearInterpolation(bq, pg, extrapolation_bc = Line())
qb1 = [scorefb(bqf(qfb1(p))) for p in pg]
qb2 = [scorefb(bqf(qfb2(p))) for p in pg]
ieb = (qb2 - qb1) .* v[:, 3]
