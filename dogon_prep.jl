using DataFrames, CodecZlib, CSV, Printf, Statistics, Distributions
using UnicodePlots, Interpolations

df = open(
    GzipDecompressorStream,
    "/home/kshedden/data/Beverly_Strassmann/Cohort_2021.csv.gz",
) do io
    CSV.read(io, DataFrame)
end

# We want the first village where a person lived
df = sort(df, :datecomb)
vi = combine(groupby(df, :ID), :Village => first)
vi = rename(vi, :Village_first => :Village1)
df = leftjoin(df, vi, on = :ID)

# Drop people who are neither in the village nor in Bamako
df = filter(row -> !ismissing(row.Bamako) && row.Bamako in [0, 1], df)

# Indicator that a female is pregnant
df[!, :Preg] = df[:, :PregMo_Use] .> 0
df[!, :Preg] = replace(df[!, :Preg], true => 1, false => 0)

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

function make_dogon_dataframe(
    sex::String,
    cbs::Symbol,
    extra_vars::AbstractVector{Symbol},
    df::AbstractDataFrame,
)
    dx = filter(row -> row.Sex == sex, df)
    outcome = :SBP_MEAN
    vl = [:Age_Yrs, :BMI, :Ht_Ave_Use, Symbol(string(cbs) * "_pcscore"), extra_vars...]
    if sex == "Female"
        push!(vl, :Preg)
    end
    dr = dx[:, vcat([outcome], vl)]
    dr = dr[completecases(dr), :]
    for j = 1:size(dr, 2)
        dr[!, j] = Vector{Float64}(dr[:, j])
    end
    return dr, vl
end
