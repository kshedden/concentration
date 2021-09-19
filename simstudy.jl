using Distributions, PyPlot, Printf

include("qreg_nn.jl")
include("functional_lr.jl")

function sim(n, d, bw, qf, sf, la)

    meanfunc = x -> 1
    sdfunc = x -> sf * sqrt(1 + x^2)

    x = randn(n, d)
    ey = meanfunc.(x[:, 1])
    sdy = sdfunc.(x[:, 1])
    y = ey + sdy .* randn(n)

    # Number of grid points
    m = 20
    xg = range(-2, 2, length = m)
    pg = range(0.1, 0.9, length = m)

    # True quantiles
    tq = zeros(m, m)

    # Estimated quantiles
    eq = zeros(m, m)

    qr = qreg_nn(y, x)

    zz = zeros(d)
    for (j, p) in enumerate(pg)

        fit(qr, p, la)

        for (i, z) in enumerate(xg)
            zz[1] = z
            eq[i, j] = predict_smooth(qr, zz, [bw, bw])
            tq[i, j] = quantile(Normal(meanfunc(z[1]), sdfunc(z[1])), p)
        end
    end

    m1 = div(m, 2)
    eq = eq - ones(20) * eq[m1, :]'
    tq = tq - ones(20) * tq[m1, :]'

    eqr = fit_flr(eq, qf, qf)
    tqr = fit_flr(tq, 0, 0)

    return (eq, tq, eqr, tqr)

end

function flip(v)
    if sum(v[1] .< 0) > sum(v[1] .> 0)
        return (-v[1], -v[2])
    end
    return v
end

n = 1000
k = 2
nrep = 5

ixp = 0

for bw in [0.1, 0.2, 0.4]
    for qf in [50, 100, 200]
        for la in [0.1, 0.2]
            for sf in [0.5, 1]

                eqra, tqra = [], []
                for j = 1:nrep
                    eq, tq, eqr, tqr = sim(n, k, bw, qf, sf, la)
                    eqr = flip(eqr)
                    tqr = flip(tqr)
                    push!(eqra, eqr)
                    push!(tqra, tqr)
                end

                for j in [1, 2]

                    PyPlot.clf()
                    fig = PyPlot.figure()
                    PyPlot.axes([0.1, 0.1, 0.7, 0.8])
                    PyPlot.grid(true)
                    for i = 1:nrep
                        PyPlot.plot(eqra[i][j], color = "orange", label = "Estimate")
                    end
                    PyPlot.plot(tqra[1][j], label = "True")

                    PyPlot.title(
                        @sprintf(
                            "Bandwidth=%.2f, Penalty=%.0f, sigma=%.1f, la=%.1f",
                            bw,
                            qf,
                            sf,
                            la
                        )
                    )
                    PyPlot.ylabel(["u", "v"][j])
                    ha, lb = PyPlot.gca().get_legend_handles_labels()
                    leg = PyPlot.figlegend(ha[[1, end]], lb[[1, end]], "center right")
                    leg.draw_frame(false)

                    global ixp
                    PyPlot.savefig(@sprintf("plots/%03d.pdf", ixp))
                    ixp += 1

                end
            end
        end
    end
end

f = [@sprintf("plots/%03d.pdf", j) for j = 0:ixp-1]
c = `gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=simstudy.pdf $f`
run(c)
