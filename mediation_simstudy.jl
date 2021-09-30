using Distributions, PyPlot, Printf, Statistics, UnicodePlots

rm("plots", force = true, recursive = true)
mkdir("plots")

include("mediation.jl")
include("functional_lr.jl")

n = 2000
m = 11

pg = collect(range(1 / m, 1 - 1 / m, length = m))

age1 = randn(n)
age2 = randn(n)

x = randn(n)
m1 = x + randn(n)
m2 = x + randn(n)
y = x + m1 + m2 + randn(n)
xm = hcat(age1, age2, x, m1, m2)

exq = quantile(Normal(0, 1), pg)
m1q = quantile(Normal(0, sqrt(2)), pg)
m2q = quantile(Normal(0, sqrt(2)), pg)

bw = [1.0, 1, 1, 1, 1]

xr = mediation_quantiles(y, xm, pg, exq, m1q, m2q, bw)

function make_plots(ifig::Int)::Int
    for j = 1:m

        # What we want
        x = exq .+ quantile(Normal(), pg[j])

        # What we have
        y = xr[:, 6, 6, j]

        PyPlot.clf()
        PyPlot.grid(true)
        PyPlot.plot(x, y, "-", color = "orange")
        PyPlot.plot([-3, 3], [-3, 3], "-", color = "black")
        PyPlot.xlabel("Actual quantiles", size = 15)
        PyPlot.ylabel("Estimated quantiles", size = 15)
        PyPlot.title(@sprintf("%.2f quantile of y given x, M1=0, M2=0", pg[j]))
        PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
        ifig += 1

    end

    return ifig
end

ifig = make_plots(0)

# Fit a low rank model to the estimated quantiles
u, v = fit_flr_tensor(vec(xr), m, 4, fill(1.0, 3), fill(1.0, 3))

# Put the fitted low rank model into interpretable coordinates
mn, uu, vv = reparameterize(u, v)

# Fitted values
xrf = zeros(m, m, m, m)
for i1 = 1:m
    for i2 = 1:m
        for i3 = 1:m
            for j = 1:m
                f = mn[j]
                f += uu[i1, 1] * vv[j, 1]
                f += uu[i2, 2] * vv[j, 2]
                f += uu[i3, 3] * vv[j, 3]
                xrf[i1, i2, i3, j] = f
            end
        end
    end
end

println(cor(vec(xr), vec(xrf)))
println(scatterplot(vec(xr), vec(xrf)))

f = [@sprintf("plots/%03d.pdf", j) for j = 0:ifig-1]
c = `gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=mediation_simstudy.pdf $f`
run(c)
