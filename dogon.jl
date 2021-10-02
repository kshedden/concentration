using DataFrames, GZip, CSV, Printf, Statistics, UnicodePlots

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

# Get medians of all body size variables, we will focus
# our findings by controlling at these values.
cq = marg_qnt(cbs, age1, sex)
hq = marg_qnt(:Ht_Ave_Use, age2, sex)
bq = marg_qnt(:BMI, age2, sex)

# Control childhood body-size at the conditional median,
# with no caliper.
vl1 = [vs(cbs, cq(0.5), Inf)]

# Control adult body-size at their conditional medians, with
# calipers 10cm for height and 2 kg/m^2 for BMI.
vl2 = [vs(:Ht_Ave_Use, hq(0.5), 10), vs(:BMI, bq(0.5), 2)]

dr = gendat(sex, age1, age2, vl1, vl2)

# The childhood body size is in column 3 of xm.
yv, xm, xmn, xsd, xna = regmat(:SBP, dr, vl1, vl2)

# The quantile regression model for SBP given childhood and
# adult body size, and other controls.
qr = qreg_nn(yv, xm)

# Bandwidth parameters for quantile smoothing
bw = fill(1.0, size(xm, 2))

# Number of quantile points to track
m = 20

xr = zeros(m, m)

# Probability points
pg = collect(range(1 / m, 1 - 1 / m, length = m))

# Quantiles for childhood body size
xg = [marg_qnt(cbs, age1, sex)(p) for p in pg]
g = x -> (x - xmn[3]) / xsd[3]

# Z-score points for childhood body size
xgx = [g(x) for x in xg]

# Storage for covariate vector.  Covariates are
# set to Z=0 except for childhood body size.
v = zeros(size(xm, 2))

# Fill in all the estimated quantiles.
for j = 1:m
    _ = fit(qr, pg[j], 0.1) # important tuning parameter here
    for i = 1:m
        v[3] = xgx[i]
        xr[i, j] = predict_smooth(qr, v, bw)
    end
end

xc, md = center(xr)

u, v = fit_flr_tensor(vec(xc), m, 2, [2.0], [2.0])

println(lineplot(pg, md))
println(lineplot(xg, u[:, 1]))
println(lineplot(pg, v[:, 1]))
