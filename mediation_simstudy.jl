using Distributions, PyPlot, Printf

rm("plots", force = true, recursive = true)
mkdir("plots")

include("mediation.jl")

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

f = [@sprintf("plots/%03d.pdf", j) for j = 0:ifig-1]
c = `gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=mediation_simstudy.pdf $f`
run(c)
