using DataFrames, Printf

# Specify a variable to match on
mutable struct vspec

    # The variable name
    name::Symbol

    # The target value for the variable
    target::Union{Float64,Missing}

    # Only consider observations within the caliper relative to the target
    caliper::Float64
end

# Get a map from subject id's to arrays containing
# all row indices with the subject id.
function gpix(df)::Dict{Int,Array{Int,1}}

    ix = Dict{Int,Array{Int,1}}()
    for (ri, id) in enumerate(df[:, :ID])
        if !haskey(ix, id)
            ix[id] = Int[]
        end
        push!(ix[id], ri)
    end

    return ix
end

# Restrict the dataframe according to the variable specifications
function restrict(dx::AbstractDataFrame, vl::Array{vspec,1})::AbstractDataFrame

    f = function (r)
        for v in vl
            if ismissing(r[v.name]) ||
               (!ismissing(v.target) && abs(r[v.name] - v.target) > v.caliper)
                return false
            end
        end
        return true
    end

    return filter(f, dx)
end

# Generate a dataset for longitudinal quantile analysis at two time points.  
# Each row of the resulting dataframe combines the age1 data for the vl1
# variables with the age2 data for the vl2 variables.
function gendat(
    df::AbstractDataFrame,
    outcome::Symbol,
    sex::String,
    age1::Float64,
    age2::Float64,
    vl1::Array{vspec,1},
    vl2::Array{vspec,1};
    child_age_caliper = 1.5,
    adult_age_caliper = 1.5,
    single = false,
)

    # Always stratify by sex.
    dx = filter(r -> r.Sex == sex, df)

    # Get all variable names that we need, except for the outcome.
    na = [:ID, :Age_Yrs]
    push!(na, [x.name for x in vl1]...)
    push!(na, [x.name for x in vl2]...)
    na = unique(na)

    # Drop unused columns and drop rows where any of the core 
    # demographic variables are missing.
    dx = dx[:, vcat(na, outcome)]
    dx = dx[completecases(dx[:, [:ID, :Age_Yrs]]), :]

    # Split into childhood and adult datasets
    dx1 = filter(r -> r.Age_Yrs <= 10, dx)
    dx2 = filter(r -> r.Age_Yrs >= 12, dx)

    # Restrict to data close to the target ages
    dx1 = filter(r -> abs(r.Age_Yrs - age1) <= child_age_caliper, dx1)
    dx2 = filter(r -> abs(r.Age_Yrs - age2) <= adult_age_caliper, dx2)

    # Restrict based on non-missingness and place calipers on other variables
    dx1 = restrict(dx1, vl1)
    dx2 = restrict(dx2, vl2)

    # Outcome is needed for adults
    dx2 = filter(r -> !ismissing(r[outcome]), dx2)

    ix1 = gpix(dx1)
    ix2 = gpix(dx2)

    idv, age1v, age2v, outv = Int[], Float64[], Float64[], Float64[]
    vd1 = [Float64[] for _ = 1:length(vl1)]
    vd2 = [Float64[] for _ = 1:length(vl2)]
    for (k, ii) in ix1
        if !haskey(ix2, k)
            continue
        end
        jj = ix2[k]

        if single
            # Find the best match for each person.
            _, iq = findmin(abs2, dx1[ii, :Age_Yrs] .- age1)
            _, jq = findmin(abs2, dx2[jj, :Age_Yrs] .- age2)

            push!(idv, k)
            push!(age1v, dx1[iq, :Age_Yrs])
            push!(age2v, dx2[jq, :Age_Yrs])
            push!(outv, dx2[jq, outcome])

            for (l, v) in enumerate(vl1)
                push!(vd1[l], dx1[iq, v.name])
            end

            for (l, v) in enumerate(vl2)
                push!(vd2[l], dx2[jq, v.name])
            end
        else
            # Consider all pairings of a childhood observation and an 
            # adult observation, within one person.
            for i in ii
                for j in jj
                    push!(idv, k)
                    push!(age1v, dx1[i, :Age_Yrs])
                    push!(age2v, dx2[j, :Age_Yrs])
                    push!(outv, dx2[j, outcome])

                    for (l, v) in enumerate(vl1)
                        push!(vd1[l], dx1[i, v.name])
                    end

                    for (l, v) in enumerate(vl2)
                        push!(vd2[l], dx2[j, v.name])
                    end
                end
            end
        end
    end

    dr = DataFrame(:ID => idv, :Age1 => age1v, :Age2 => age2v, outcome => outv)

    # Include variables measured in childhood (childhood body size)
    for (j, c) in enumerate(vl1)
        dr[:, Symbol(@sprintf("%s1", c.name))] = vd1[j]
    end

    # Include variables in adulthood (adult body size)
    for (j, c) in enumerate(vl2)
        dr[:, Symbol(@sprintf("%s2", c.name))] = vd2[j]
    end

    return dr
end

# Build regression matrices for quantile regression with dependent
# variable 'dvar'.  Standardize all variables and return the mean/sd
# used for standardization so we can map between the original and
# standardized scales.
function regmat(
    dvar::Symbol,
    dr::AbstractDataFrame,
    vl1::Array{vspec,1},
    vl2::Array{vspec,1},
)

    xna = [:Age1, :Age2]
    push!(xna, [Symbol(@sprintf("%s1", x.name)) for x in vl1]...)
    push!(xna, [Symbol(@sprintf("%s2", x.name)) for x in vl2]...)
    xna = unique(xna)

    xm = dr[:, xna]
    xm = Array{Float64,2}(xm)

    xmn = mean(xm, dims = 1)
    xsd = std(xm, dims = 1)
    for j = 1:size(xm, 2)
        xm[:, j] = (xm[:, j] .- xmn[j]) ./ xsd[j]
    end

    yv = dr[:, dvar]

    return tuple(yv, xm, xmn, xsd, xna)
end

# Estimate marginal quantiles for 'trait' at a given age.
# The returned function f(age, p) returns the marginal
# quantile of 'trait' at probability point 'p' for age
# 'age'.
function marg_qnt(
    trait::Symbol,
    age::Float64,
    sex::String,
    la::Float64 = 1.0,
    bw::Float64 = 1.0,
)

    dx = df[:, [:ID, :Sex, :Age_Yrs, trait]]
    dx = dx[completecases(dx), :]
    dx = filter(r -> r.Sex == sex, dx)

    y = Array{Float64,1}(dx[:, trait])
    x = Array{Float64,1}(dx[:, :Age_Yrs])[:, :]
    qr = qreg_nn(y, x)

    f = function (p::Float64)
        _ = fit(qr, p, la)
        return predict_smooth(qr, [age], [bw])
    end

    return f
end
