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

    # Only consider observations within the caliper of the target
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


function gendat(sex::String, age1::Float64, age2::Float64, vl1::Array{vs,1}, vl2::Array{vs,1})

    # Always stratify by sex.
    dx = filter(r->r.Sex==sex, df)

    # Get all variable names that we need, except for SBP.
    # The variables in na must not be missing.
    na = [:ID, :Age_Yrs]
    push!(na, [x.name for x in vl1]...)
    push!(na, [x.name for x in vl2]...)
    na = unique(na)

    # Eliminate non-usable data
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

    # Add control variables measured with SBP
    for (j, c) in enumerate(vl2)
        dr[:, Symbol(@sprintf("%s2", c.name))] = vd2[j]
    end

    return dr
end

# Build regression matrices for quantile regression.
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

# Control childhood body-size
vl1 = [vs(:Ht_Ave_Use, 98, Inf)]

# Control adult body-size
vl2 = [vs(:Ht_Ave_Use, 158, 10), vs(:BMI, 20, 2)]

dr = gendat("Female", 5.0, 20.0, vl1, vl2)

yv, xm, xmn, xsd, xna = regmat(dr, vl1, vl2)

qr = qreg_nn(yv, xm)

bw = 1.0 * ones(size(xm, 2))

m = 20
xr = zeros(m, m)

# Probability points for SBP
pg = collect(range(1/m, 1-1/m, length=m))

# Z-score points for childhood height
xg = collect(range(-2, 2, length=m))

# Covariate vector
v = zeros(size(xm, 2))

for j in 1:m
    _ = fit(qr, pg[j], 0.2)
    for i in 1:m
        v[3] = xg[i]
        xr[i, j] = predict_smooth(qr, v, bw)
    end
end

xr0 = broadcast(-, xr, xr[div(m, 2), :]')

u, v = fit_flr(xr0, 10., 10.)

println(lineplot(xg, u))
println(lineplot(pg, v))
