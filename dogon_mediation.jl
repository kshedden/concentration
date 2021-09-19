using DataFrames, GZip, CSV, Printf, Statistics, Distributions, UnicodePlots, Interpolations

include("qreg_nn.jl")
include("functional_lr.jl")
include("dogon_utils.jl")

df = GZip.open("/home/kshedden/data/Beverly_Strassmann/Cohort_2021.csv.gz") do io
    CSV.read(io, DataFrame)
end

# Analyze one sex, at specific childhood and adult ages
sex = "Female"
age1 = 5.0
age2 = 20.0

# childhood body size variable (primary exposure)
# Possibilities: Ht_Ave_Use, WT, HAZ, WAZ
#cbs = :Ht_Ave_Use
cbs = :HAZ

# Number of quantile points to track
m = 5

# Probability points
pg = collect(range(1 / m, 1 - 1 / m, length = m))

# Marginal quantiles for all variables of interest at the target ages
cq = [marg_qnt(cbs, sex, p)(age1) for p in pg]
hq = [marg_qnt(:Ht_Ave_Use, sex, p)(age2) for p in pg]
bq = [marg_qnt(:BMI, sex, p)(age2) for p in pg]

# Child childhood body size variable.
vl1 = [vs(cbs, missing, Inf)]

# Adult body size variables.
vl2 = [vs(:Ht_Ave_Use, missing, Inf), vs(:BMI, missing, Inf)]

dr = gendat(sex, age1, age2, vl1, vl2)

# The childhood body size is in column 3 of xm.
yv, xm, xmn, xsd, xna = regmat(:SBP, dr, vl1, vl2)

gcbs = x -> (x - xmn[3]) / xsd[3]
gaht = x -> (x - xmn[4]) / xsd[4]
gabm = x -> (x - xmn[5]) / xsd[5]

# The quantile regression model for SBP given childhood and
# adult body size, and other controls.
qr = qreg_nn(yv, xm)

# Bandwidth parameters for quantile smoothing
bw = fill(1.0, size(xm, 2))

xr = zeros(m, m, m, m)

# Storage for covariate vector.
v = zeros(size(xm, 2))

# Fill in all the estimated quantiles.
for j = 1:m
    println(j)
    # Estimate a specific quantile of adult SBP
    _ = fit(qr, pg[j], 0.1) # important tuning parameter here

    for i1 = 1:m # Childhood body size
        for i2 = 1:m # Adult height
            for i3 = 1:m # Adult BMI
                v[3] = gcbs(cq[i1])
                v[4] = gaht(hq[i2])
                v[5] = gabm(bq[i3])
                xr[i1, i2, i3, j] = predict_smooth(qr, v, bw)
            end
        end
    end
end

u, v = fit_flr_tensor(vec(xr), m, 4, fill(1.0, 3), fill(1.0, 3))

mn, uu, vv = reparameterize(u, v)

# Direct effect
sn = Normal()
qn = quantile(sn, pg)
pg1 = cdf(sn, qn .- 1)
pg2 = cdf(sn, qn .+ 1)
scoref = LinearInterpolation(pg, uu[:, 1], extrapolation_bc = Line())
score1 = [scoref(x) for x in pg1]
score2 = [scoref(x) for x in pg2]
de = (score2 - score1) .* vv[:, 1]

# Marginal distributions of the mediators.
qr1 = qreg_nn(xm[:, 4], xm[:, 1:3])
marg1 = zeros(m, m)
v = zeros(3)
for i in 1:m
    _ = fit(qr1, pg[i], 0.1) # important tuning parameter here
	for j in 1:m
	    v[3] = gcbs(cq[j])
		marg1[i, j] = predict_smooth(qr1, v, bw[1:3])
	end
end
marg1 = marg1 * xsd[4] + xmn[4]

# Indirect effect
