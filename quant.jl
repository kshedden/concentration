using CSV, GZip, DataFrames, NamedArrays, UnicodePlots, LinearAlgebra, Printf, Statistics

include("functional_lr.jl")
include("qreg_nn.jl")

pa = "/afs/umich.edu/user/k/s/kshedden/Projects/Beverly_Strassmann/ursps21/cohort.csv.gz"

da = GZip.open(pa) do io
    CSV.read(io, DataFrame)
end

function prep(sex)
    dx = filter(x -> !ismissing(x.Sex) && x.Sex == sex, da)
    dx = dx[:, [:Age_Yrs, :BMI, :WT, :Ht_Ave_Use, :SBP_MEAN]]
    dx = dx[completecases(dx), :]
    dx = NamedArray(Array{Float64}(dx))
    setnames!(dx, ["Age", "BMI", "WT", "HT", "SBP"], 2)
    dx
end

# Separate datasets for females and males.
df = prep("Female")
dm = prep("Male")

# Returns an estimate of the quantiles of out given v is fixed at each
# value in x, at the given age, using the data dx.  h is a grid
# parameter that controls how finely the quantile function is
# calculated.
function quant(out, v, x, age, dx; h = 0.02, lam = 0.1)

    pl = 0.02:h:0.98 # probability points
    ql = zeros(length(x), length(pl))

    dxx = copy(Array(dx[:, ["Age", v]]))
    mn, sd = zeros(2), zeros(2)
    for j = 1:size(dxx, 2)
        mn[j] = mean(dxx[:, j])
        sd[j] = std(dxx[:, j])
        dxx[:, j] = dxx[:, j] .- mn[j]
        dxx[:, j] = dxx[:, j] ./ sd[j]
    end

    for (k, p) in enumerate(pl)
        qr = qreg_nn(Array(dx[:, out]), dxx, p, lam)
        for (i, xv) in enumerate(x)
            z = [(age - mn[1]) / sd[1], (xv - mn[2]) / sd[2]]
            ql[i, k] = predict_smooth(qr, z, [1.0, 1.0])
        end
    end

    tuple(pl, ql)

end

# Returns an estimate of the quantiles of 'v' given age, using the
# data dx.  h is a grid parameter that controls how finely the
# quantile function is calculated.
function quant0(v, age, dx; h = 0.02, lam = 0.1)

    pl = 0.02:h:0.98 # probability points
    ql = zeros(length(pl))

    dxx = copy(Array(dx[:, "Age"]))
    dxx = dxx[:, :]
    mn = mean(dxx[:, 1])
    sd = std(dxx[:, 1])
    dxx[:, 1] = dxx[:, 1] .- mn
    dxx[:, 1] = dxx[:, 1] ./ sd

    z = [age - mn] / sd
    for (k, p) in enumerate(pl)
        qr = qreg_nn(Array(dx[:, v]), dxx, p, lam)
        ql[k] = predict_smooth(qr, z, [1.0])
    end

    tuple(pl, ql)

end

h = 0.1
age = 18
out = "SBP"
ctrl = "HT"

# Quantiles of the control variable
p0, q0 = quant0(ctrl, age, df, h = h)

plt = lineplot(
    p0,
    q0,
    xlabel = @sprintf("%s probability", ctrl),
    ylabel = @sprintf("%s quantile", ctrl),
    title = @sprintf("Quantiles of %s at age %.0f", ctrl, age),
)
println(plt)

pf, qf = quant(out, ctrl, q0, age, df, h = h)

plt = lineplot(
    pf,
    qf[1, :],
    xlabel = @sprintf("%s probability", out),
    ylabel = @sprintf("%s quantile", out),
    name = @sprintf("%s=%.1f", ctrl, q0[1]),
    title = @sprintf("Quantiles of %s at age %.0f given %s", out, age, ctrl),
)
lineplot!(plt, qf[end, :], name = @sprintf("%s=%.1f", ctrl, q0[end]))
println(plt)

# Center the quantiles around the median value of the control variable
ii = div(size(qf, 2) + 1, 2)
qfc = copy(qf)
for j = 1:size(qfc, 2)
    qfc[:, j] = qfc[:, j] .- qfc[ii, j]
end

# Fit a smooth low-rank model to the residuals
u, v = fit_flr(qfc, 5.0, 5.0)

plt = lineplot(
    u,
    xlabel = @sprintf("%s probability", ctrl),
    ylabel = @sprintf("%s coefficient", ctrl),
)
println(plt)

plt = lineplot(
    v,
    xlabel = @sprintf("%s probability", out),
    ylabel = @sprintf("%s coefficient", out),
)
println(plt)
