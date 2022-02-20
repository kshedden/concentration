using DataFrames, CSV, Printf, Statistics, UnicodePlots
using CodecZlib, PyPlot

rm("plots", recursive = true, force = true)
mkdir("plots")

ifig = 0

include("qreg_nn.jl")
include("flr_reg.jl")
include("dogon_utils.jl")
include("basis.jl")
include("plot_utils.jl")

# Analyze one sex, at specific childhood and adult ages
age1 = 5.0
age2 = 21.0

df = open("/home/kshedden/data/Beverly_Strassmann/Cohort_2021.csv.gz") do io
    CSV.read(GzipDecompressorStream(io), DataFrame)
end

# Need to confirm that this is OK
df[!, :Bamako] = replace(df[!, :Bamako], 2 => 1)

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

function make_dataframe(cbs, age1, age2, sex, df)
    # Child variables
    vl1 = vspec[vspec(:HT_pcscore, 0, Inf),]

    # Adult variables
    vl2 = [
        vspec(:BMI, 0, Inf),
        vspec(:Ht_Ave_Use, 0, Inf),
        vspec(:F0_Mom_SBP_Z_Res_USE, 0, Inf),
        vspec(:Bamako, 0, Inf),
    ]

    dr = gendat(df, :SBP_MEAN, sex, age1, age2, vl1, vl2; adult_age_caliper = 2)

    return (dr, vl1, vl2)
end

function do_all(dr, vl1, vl2, sex, ifig)
    yv, xm, xmn, xsd, xna = regmat(:SBP_MEAN, dr, vl1, vl2)

    # The quantile regression model.
    qr = qreg_nn(yv, xm)

    # Probability points
    pp = collect(range(0.1, 0.9, length = 9))

    # Fill in the quantiles
    qh = zeros(length(yv), length(pp))
    for (j, p) in enumerate(pp)
        _ = fit(qr, p, 0.1)
        qh[:, j] = qr.fit
    end

    # Remove the central axis
    qhc = copy(qh)
    qhm = zeros(length(pp))
    for j = 1:size(qh, 2)
        qhm[j] = median(qh[:, j])
        qhc[:, j] .-= qhm[j]
    end

    # Build basis matrices for the low rank model
    X, Xp = Vector{Matrix{Float64}}(), Vector{Matrix{Float64}}()
    gr = collect(range(-2, 2, length = 101))
    grx = []
    for x in eachcol(xm)
        if length(unique(x)) > 2
            # Not a binary variable
            x = x .- median(x)
            x = x / median(abs.(x))
            x /= 0.67
            B, g = genbasis(x, 5, std(x) / 2, linear = true)
            push!(X, B)
            push!(Xp, g(gr))
        else
            # A binary variable
            push!(X, x[:, :])
            bb = zeros(101)
            bb[51:end] .= 1
            push!(Xp, bb[:, :])
        end
        push!(grx, gr)
    end

    # Fit the low-rank model
    cu = 100 * ones(length(X))
    cu[end] = 0
    cv = 100 * ones(9)
    fr = fitlr(X, Xp, qhc, cu, cv)

    vnames = [string(x) for x in xna]
    ifig = plots_flr(sex, X, Xp, pp, fr, grx, vnames, ifig)
    return ifig
end

function main(ifig)
    for sex in ["Female", "Male"]
        dr, vl1, vl2 = make_dataframe(cbs, age1, age2, sex, df)
        ifig = do_all(dr, vl1, vl2, sex, ifig)
    end
    return ifig
end

ifig = main(ifig)

f = [@sprintf("plots/%03d.pdf", j) for j = 0:ifig-1]
c =
    `gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=writing/dogon_flr_reg.pdf $f`
run(c)
