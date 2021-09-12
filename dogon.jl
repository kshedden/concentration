using DataFrames, GZip, CSV, Printf, Statistics, UnicodePlots

include("qreg_nn.jl")
include("functional_lr.jl")
include("dogon_utils.jl")

df = GZip.open("/home/kshedden/data/Beverly_Strassmann/Cohort_2021.csv.gz") do io
    CSV.read(io, DataFrame)
end

# Analyze one sex, at specific childhood and adult ages
sex = "Female"
age1 = 5.
age2 = 20.

# childhood body size variable (primary exposure)
# Possibilities: Ht_Ave_Use, WT, HAZ, WAZ
#cbs = :Ht_Ave_Use
cbs = :HAZ

# Get medians of all body size variables, we will focus
# our findings by controlling at these values.
cq = marg_qnt(cbs, sex, 0.5)
hq = marg_qnt(:Ht_Ave_Use, sex, 0.5)
bq = marg_qnt(:BMI, sex, 0.5)

# Control childhood body-size at the conditional median,
# with no caliper.
vl1 = [vs(cbs, cq(age1), Inf)]

# Control adult body-size at their conditional medians, with
# calipers 10cm for height and 2 kg/m^2 for BMI.
vl2 = [vs(:Ht_Ave_Use, hq(age2), 10), vs(:BMI, bq(age2), 2)]

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
pg = collect(range(1/m, 1-1/m, length=m))

# Quantiles for childhood body size
xg = [marg_qnt(cbs, sex, p)(age1) for p in pg]
g = x -> (x - xmn[3]) / xsd[3]

# Z-score points for childhood body size
xgx = [g(x) for x in xg]

# Storage for covariate vector.  Covariates are
# set to Z=0 except for childhood body size.
v = zeros(size(xm, 2))

for j in 1:m
    _ = fit(qr, pg[j], 0.1) # important tuning parameter here
    for i in 1:m
        v[3] = xgx[i]
        xr[i, j] = predict_smooth(qr, v, bw)
    end
end

u, v = fit_flr_tensor(vec(xr), m, 2, [1.], [1.0])

mn, uu, vv = reparameterize(u, v)

println(lineplot(pg, mn))
println(lineplot(xg, uu[:,1]))
println(lineplot(pg, vv[:,1]), ylim=[-0.15, 0])
