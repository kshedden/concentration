#=
Use GEE to model Dogon SBP in terms of current height and BMI, 
early childhood growth (PC-scores), and control variables.
=#

using CSV, DataFrames, CodecZlib, Printf, Statistics, UnicodePlots
using StatsBase, StatsModels, Distributions, GLM, GEE

df = open("/home/kshedden/data/Beverly_Strassmann/Cohort_2021.csv.gz") do io
    CSV.read(GzipDecompressorStream(io), DataFrame)
end

function merge_pc_scores(df)
    pcs = open("dogon_pc_scores.csv") do io
        CSV.read(io, DataFrame)
    end
    pcs = rename(pcs, :id => :ID)
    na = names(pcs)[2:end]
    for a in na
        pcs = rename(pcs, a => Symbol(@sprintf("%s_pcscore", string(a))))
    end
    df = outerjoin(df, pcs, on = :ID)
    return df
end

df = merge_pc_scores(df)

# We want the first village where a person lived
df = sort(df, :datecomb)
vi = combine(groupby(df, :ID), :Village => first)
vi = rename(vi, :Village_first => :Village1)
df = leftjoin(df, vi, on = :ID)

# Drop people who are neither in the village nor in Bamako
df = filter(row -> !ismissing(row.Bamako) && row.Bamako in [0, 1], df)

# Logarithm of the number of prior BP measurements
df[:, :lognummeas] = df[:, :lognummeas_BASE10]

function zscore(x::AbstractVector)::AbstractVector
    x .-= median(x)
    x ./= median(abs.(x))
    x ./= 1.48
    return x
end

function check_regression(sex, cbs, adj, df, out)
    outcome = "SBP_MEAN"
    cbsv = string(cbs) * "_pcscore"
    xvars = ["BMI", "Ht_Ave_Use", cbsv, "Age_Yrs", "Village1", "lognummeas"]

    dr = filter(row -> row.Sex == sex, df)
    dr = dr[:, vcat([outcome, "ID"], xvars)]
    dr = dr[completecases(dr), :]
    for j = 1:size(dr, 2)
        dr[!, j] = Vector{Float64}(dr[:, j])
    end

    # Z-score non-categorical variables
    for x in xvars
        if !(x in ["Village1", "Bamako"])
            dr[:, x] = zscore(dr[:, x])
        end
    end

    # Polynomial terms in age
    for k = 2:3
        dr[:, Symbol("Age_Yrs$(k)")] = dr[:, :Age_Yrs] .^ k
    end

    fnames = [:BMI, :Ht_Ave_Use, Symbol(cbsv), :Age_Yrs, :Age_Yrs2, :Age_Yrs3]
    if adj == 1
        push!(fnames, :Village1, :lognummeas)
    end
    fml = Term(:SBP_MEAN) ~ sum(Term.(fnames))
    dr = sort(dr, :ID)
    contrasts = Dict(:Village1 => DummyCoding())
    mm = gee(
        fml,
        dr,
        dr[:, :ID],
        Normal(),
        ExchangeableCor(),
        IdentityLink(),
        contrasts = contrasts,
    )

    s = split(string(mm), "\n")
    s = join(s[6:end], "\n")
    write(out, @sprintf("%s %s\n", sex, string(cbs)))
    write(out, @sprintf("n=%d\n", nobs(mm)))
    write(out, s)
    write(out, "\n\n")
end

out = open("writing/check_regression.txt", "w")

for cbs in [:HT, :WT, :BMI, :HAZ, :WAZ, :BAZ]
    for adj in [0, 1]
        check_regression("Female", cbs, adj, df, out)
        check_regression("Male", cbs, adj, df, out)
    end
end

close(out)
