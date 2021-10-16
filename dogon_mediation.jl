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

pg = collect(range(1 / m, 1 - 1 / m, length = m))

# Childhood body size variable.
vlc = [vspec(cbs, missing, Inf)]

# Adult body size variables.
vla = [vspec(:Ht_Ave_Use, missing, Inf), vspec(:BMI, missing, Inf)]

# Bandwidth parameters for quantile smoothing
bw = fill(1.0, 5)

qrm = mediation_prep(:SBP_MEAN, sex, age1, age2, cbs, m, vlc, vla, bw)

# Estimate the direct effect
sn = Normal()
qn = quantile(sn, pg)
pg1 = cdf(sn, qn .- 1)
pg2 = cdf(sn, qn .+ 1)
scoref = LinearInterpolation(pg, qrm.u[:, 1], extrapolation_bc = Line())
score1 = [scoref(x) for x in pg1]
score2 = [scoref(x) for x in pg2]
de = (score2 - score1) .* qrm.v[:, 1]

# Marginal quantile function of height for perturbed populations
# of childhood HAZ.
medh = qrm.xmn[4] .+ qrm.xsd[4] * qrm.xm[:, 4]
qfh1 = marginal_qf(medh, qrm.xm[:, 1:3], -1.0, qrm.cq, qrm.gcbs, bw[1:3], pg)
qfh2 = marginal_qf(medh, qrm.xm[:, 1:3], 1.0, qrm.cq, qrm.gcbs, bw[1:3], pg)

# Marginal quantile function of BMI for perturbed populations
# of childhood HAZ.
medb = qrm.xmn[5] .+ qrm.xsd[5] * qrm.xm[:, 5]
qfb1 = marginal_qf(medb, qrm.xm[:, 1:3], -1.0, qrm.cq, qrm.gcbs, bw[1:3], pg)
qfb2 = marginal_qf(medb, qrm.xm[:, 1:3], 1.0, qrm.cq, qrm.gcbs, bw[1:3], pg)

# Indirect effect through height
scorefh = LinearInterpolation(pg, qrm.u[:, 2], extrapolation_bc = Line())
hqf = LinearInterpolation(qrm.hq, pg, extrapolation_bc = Line())
qh1 = [scorefh(hqf(qfh1(p))) for p in pg]
qh2 = [scorefh(hqf(qfh2(p))) for p in pg]
ieh = (qh2 - qh1) .* qrm.v[:, 2]

# Indirect effect through BMI
scorefb = LinearInterpolation(pg, qrm.u[:, 3], extrapolation_bc = Line())
bqf = LinearInterpolation(qrm.bq, pg, extrapolation_bc = Line())
qb1 = [scorefb(bqf(qfb1(p))) for p in pg]
qb2 = [scorefb(bqf(qfb2(p))) for p in pg]
ieb = (qb2 - qb1) .* qrm.v[:, 3]
