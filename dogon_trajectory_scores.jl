using CSV, DataFrames, Printf, Statistics, LinearAlgebra

pa = "/home/kshedden/Projects/Beverly_Strassmann/Stature-BP/imputed_data"

vnames = ["BAZ", "HAZ", "WAZ", "BMI", "HT", "WT"]

function readimp(vname)
    ids = []
    xm = nothing
    for k = 0:19

        fn = joinpath(pa, @sprintf("%s_imp_%d.csv", vname, k))
        df = open(fn) do io
            CSV.read(io, DataFrame)
        end

        vx = ["$(vname)$k" for k = 1:10]
        push!(ids, df[:, :ID])
        x = Matrix{Float64}(df[:, vx])
        if isnothing(xm)
            xm = x
        else
            xm .+= x
        end
    end
    xm ./= 20

    for j in eachindex(ids)
        @assert all(ids[j] .== ids[1])
    end

    return (first(ids), xm)
end

function main()
    ids0 = nothing

    scores, loadings = [], []
    for vname in vnames
        ids, xm = readimp(vname)
        for c in eachcol(xm)
            c .-= mean(c)
        end

        u, s, v = svd(xm)
        if sum(v[:, 1] .> 0) < sum(v[:, 1] .< 0)
            v[:, 1] .*= -1
            u[:, 1] .*= -1
        end
        sc = DataFrame(:id => ids, Symbol(vname) => u[:, 1])
        push!(scores, sc)
        println(size(xm))
        push!(loadings, v[:, 1])
    end

    sc = first(scores)
    for sq in scores[2:end]
        sc = outerjoin(sc, sq, on = :id)
    end

    load = DataFrame(vnames[1] => loadings[1])
    for j = 2:length(loadings)
        load[:, vnames[j]] = loadings[j]
    end

    return sc, load
end

sc, loadings = main()

CSV.write("dogon_pc_scores.csv", sc)
CSV.write("dogon_pc_loadings.csv", loadings)
