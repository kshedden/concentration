using DataFrames, GZip, CSV, Printf, Statistics, UnicodePlots

include("qreg_nn.jl")
include("functional_lr.jl")

df = GZip.open("/nfs/kshedden/Beverly_Strassmann/Cohort_2021.csv.gz") do io
    CSV.read(io, DataFrame)
end

# Specify a variable to match on
mutable struct vs

    # The variable name
    name::Symbol

    # The target value for the variable
    target::Float64

    # Only consider observations within the caliper relative to the target
    caliper::Float64

end

# Get a map from subject id's to arrays containing
# all row indices with the subject id.
function gpix(df)::Dict{Int,Array{Int,1}}

    ix = Dict{Int,Array{Int,1}}()
    for (ri,id) in enumerate(df[:, :ID])
        if !haskey(ix, id)
            ix[id] = []
        end
        push!(ix[id], ri)
    end

    return ix

end

# Restrict the dataframe according to the variable specifications
function restrict(dx::AbstractDataFrame, vl::Array{vs,1})::AbstractDataFrame

    f = function(r)
        for v in vl
            if abs(r[v.name] - v.target) > v.caliper
                return false
            end
        end
        return true
    end

    return filter(f,  dx)

end

# Generate a dataset for two-point longitudinal quantile analysis.  Conditional quantiles
# of SBP at age2 are calculated given the values of the variables in vl1 at age1 and
# the values of the variables in vl2 at age2.
function gendat(sex::String, age1::Float64, age2::Float64, vl1::Array{vs,1}, vl2::Array{vs,1})

    # Always stratify by sex.
    dx = filter(r->r.Sex==sex, df)

    # Get all variable names that we need, except for SBP.
    na = [:ID, :Age_Yrs]
    push!(na, [x.name for x in vl1]...)
    push!(na, [x.name for x in vl2]...)
    na = unique(na)

    # Drop rows where any of the core variables are missing.
    # SBP may be missing in childhood, all other variables
    # must be observed.
    dx = dx[:, vcat(na, :SBP_MEAN)]
    dx = dx[completecases(dx[:, na]), :]

    # Split into childhood and adult datasets
    dx1 = filter(r->r.Age_Yrs<=10, dx)
    dx2 = filter(r->r.Age_Yrs>=12 && !ismissing(r.SBP_MEAN), dx)

    # Restrict to data close to the target ages
    dx1 = filter(r->abs(r.Age_Yrs - age1) <= 1.5, dx1)
    dx2 = filter(r->abs(r.Age_Yrs - age2) <= 1.5, dx2)

    dx1 = restrict(dx1, vl1)
    dx2 = restrict(dx2, vl2)

    ix1 = gpix(dx1)
    ix2 = gpix(dx2)

    idv, age1v, age2v, sbpv = Int[], Float64[], Float64[], Float64[]
    vd1 = [Float64[] for j in 1:length(vl1)]
    vd2 = [Float64[] for j in 1:length(vl2)]
    for (k,ii) in ix1
        if !haskey(ix2, k)
            continue
        end
        jj = ix2[k]

        for i in ii
            for j in jj
                push!(idv, k)
                push!(age1v, dx1[i, :Age_Yrs])
                push!(age2v, dx2[j, :Age_Yrs])
                push!(sbpv, dx2[j, :SBP_MEAN])

                for (l, v) in enumerate(vl1)
                    push!(vd1[l], dx1[i, v.name])
                end

                for (l, v) in enumerate(vl2)
                    push!(vd2[l], dx2[j, v.name])
                end

            end
        end
    end

    dr = DataFrame(:Id=>idv, :Age1=>age1v, :Age2=>age2v, :SBP=>sbpv)

    # Add control variables measured with childhood body size
    for (j, c) in enumerate(vl1)
        dr[:, Symbol(@sprintf("%s1", c.name))] = vd1[j]
    end

    # Add control variables measured with SBP and adult body size
    for (j, c) in enumerate(vl2)
        dr[:, Symbol(@sprintf("%s2", c.name))] = vd2[j]
    end

    return dr
end

# Build regression matrices for quantile regression.  Standardize
# all variables and return the mean/sd used for standardization
# so we can map between the original and standardized scales.
function regmat(dr::AbstractDataFrame, vl1::Array{vs,1}, vl2::Array{vs,1})

    xna = [:Age1, :Age2]
    push!(xna, [Symbol(@sprintf("%s1", x.name)) for x in vl1]...)
    push!(xna, [Symbol(@sprintf("%s2", x.name)) for x in vl2]...)
    xna = unique(xna)

    xm = dr[:, xna]
    xm = Array{Float64,2}(xm)

    xmn, xsd = [], []
    for j in 1:size(xm,2)
        m = mean(xm[:, j])
        s = std(xm[:, j])
        push!(xmn, m)
        push!(xsd, s)
        xm[:, j] = (xm[:, j] .- m) ./ s
    end

    yv = dr[:, :SBP]

    return (yv, xm, xmn, xsd, xna)

end


# Estimate marginal quantiles by age for 'trait'.
function mqa(trait::Symbol, sex::String, p::Float64; la::Float64=1.0, bw::Float64=1.0)

    dx = df[:, [:ID, :Sex, :Age_Yrs, trait]]
    dx = dx[completecases(dx), :]
    dx = filter(r->r.Sex==sex, dx)

    y = Array{Float64,1}(dx[:, trait])
    x = Array{Float64,1}(dx[:, :Age_Yrs])[:, :]
    qr = qreg_nn(y, x)
    _ = fit(qr, p, la)

    return a->predict_smooth(qr, [a], [bw])

end

# Analyze one sex, at specific childhood and adult ages
sex = "Female"
age1 = 5.
age2 = 20.

# childhood body size variable (primary exposure)
# Possibilities: Ht_Ave_Use, WT, HAZ, WAZ
#cbs = :Ht_Ave_Use
cbs = :HAZ

# Use these to determine where to control the adult body size values,
# and to select cases that are informative
cq = mqa(cbs, sex, 0.5)
hq = mqa(:Ht_Ave_Use, sex, 0.5)
bq = mqa(:BMI, sex, 0.5)

# Control childhood body-size
vl1 = [vs(cbs, cq(age1), Inf)]

# Control adult body-size
vl2 = [vs(:Ht_Ave_Use, hq(age2), 10), vs(:BMI, bq(age2), 2)]

dr = gendat(sex, age1, age2, vl1, vl2)

# The childhood body size is in column 3 of xm.
yv, xm, xmn, xsd, xna = regmat(dr, vl1, vl2)

qr = qreg_nn(yv, xm)

bw = 1.0 * ones(size(xm, 2))

m = 20
xr = zeros(m, m)

# Probability points for SBP
pg = collect(range(1/m, 1-1/m, length=m))

# Z-score points for childhood body size
xg = [mqa(cbs, sex, p)(age1) for p in pg]
g = x -> (x - xmn[3]) / xsd[3]
xg = [g(x) for x in xg]

# Covariate vector
v = zeros(size(xm, 2))

for j in 1:m
    _ = fit(qr, pg[j], 0.1) # important tuning parameter here
    for i in 1:m
        v[3] = xg[i]
        xr[i, j] = predict_smooth(qr, v, bw)
    end
end

xr0 = broadcast(-, xr, xr[div(m, 2), :]')

u, v = fit_flr(xr0, 10., 50.)

println(lineplot(xg, u))
println(lineplot(pg, v))
