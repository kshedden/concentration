using PyPlot

# Plot two X-side scores against each other as a biplot.
function plotxx(sex, vname, xp, spt, sptl, beta, j1, j2, ifig)

    sexs = sex == 1 ? "Male" : "Female"
    PyPlot.clf()
    PyPlot.title(@sprintf("%s %s contrasts", sexs, vname))
    PyPlot.grid(true)
    PyPlot.plot(xp[:, j1], xp[:, j2], "o", alpha = 0.2, rasterized = true)
    for (k, z) in enumerate(spt)
        PyPlot.text(z[j1], z[j2], sptl[k], size = 14, ha = "center", va = "center")
    end

    # Make it a biplot
    bs = 2 * beta[:, [j1, j2]]
    for j = 1:3
        # Move the text so that the arrow ends at the loadings.
        bs[j, :] .+= 0.3 * bs[j, :] / norm(bs[j, :])
    end
    PyPlot.gca().annotate(
        "Age",
        xytext = (bs[1, 1], bs[1, 2]),
        xy = (0, 0),
        arrowprops = Dict(:arrowstyle => "<-", :shrinkA => 0, :shrinkB => 0),
    )
    PyPlot.gca().annotate(
        "BMI",
        xytext = (bs[2, 1], bs[2, 2]),
        xy = (0, 0),
        arrowprops = Dict(:arrowstyle => "<-", :shrinkA => 0, :shrinkB => 0),
    )
    PyPlot.gca().annotate(
        "Ht",
        xytext = (bs[3, 1], bs[3, 2]),
        xy = (0, 0),
        arrowprops = Dict(:arrowstyle => "<-", :shrinkA => 0, :shrinkB => 0),
    )

    PyPlot.ylabel(@sprintf("Covariate score %d", j2), size = 15)
    PyPlot.xlabel(@sprintf("Covariate score %d", j1), size = 15)
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    ifig += 1
end

function plot_qtraj(sex, npc, vname, spt, sptl, kt, xmat, qhc, nnb, ifig)

    sexs = sex == 1 ? "Male" : "Female"
    rsltp = []
    PyPlot.clf()
    PyPlot.axes([0.12, 0.12, 0.75, 0.8])
    PyPlot.title(@sprintf("%s %s contrasts", sexs, vname))
    for (j, z) in enumerate(spt)

        # Nearest neighbors of the support point in the projected
        # X-space.
        ii, _ = knn(kt, z, nnb)

        # Store the x-variable means corresponding to
        # each support point.
        row = [sex == 1 ? "Male" : "Female", npc, sptl[j], mean(xmat[ii, :], dims = 1)...]
        push!(rsltp, row)

        qp = mean(qhc[ii, :], dims = 1)
        PyPlot.plot(pp, vec(qp), "-", label = sptl[j])
    end
    ha, lb = PyPlot.gca().get_legend_handles_labels()
    leg = PyPlot.figlegend(ha, lb, "center right")
    leg.draw_frame(false)
    PyPlot.grid(true)
    PyPlot.xlabel("Probability", size = 15)
    PyPlot.ylabel("SBP quantile deviation", size = 15)
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))

    return rsltp, ifig + 1
end

function plot_qtraj_diff(sex, npc, vname, spt, sptl, kt, xmat, qhc, nnb, ifig)

    sexs = sex == 1 ? "Male" : "Female"
    rsltp = []
    PyPlot.clf()
    PyPlot.axes([0.12, 0.12, 0.75, 0.8])
    PyPlot.title(@sprintf("%s %s contrasts", sexs, vname))
    for j = 1:div(length(spt), 2)

        z1 = spt[2*(j-1)+1]
        z2 = spt[2*(j-1)+2]

        # Nearest neighbors of the support point in the projected
        # X-space.
        ii1, _ = knn(kt, z1, nnb)
        ii2, _ = knn(kt, z2, nnb)

        qp1 = mean(qhc[ii1, :], dims = 1)
        qp2 = mean(qhc[ii2, :], dims = 1)
        qp = vec(qp1) - vec(qp2)

        PyPlot.plot(pp, qp, "-", label = sptl[2*(j-1)+1])
    end
    ha, lb = PyPlot.gca().get_legend_handles_labels()
    leg = PyPlot.figlegend(ha, lb, "center right")
    leg.draw_frame(false)
    PyPlot.grid(true)
    PyPlot.xlabel("Probability", size = 15)
    PyPlot.ylabel("SBP quantile deviation", size = 15)
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))

    return ifig + 1
end

function plotxx_all(sex, vname, xp, spt, sptl, beta, ifig)
    for j2 = 1:size(beta, 2)
        for j1 = 1:j2-1
            ifig = plotxx(sex, vname, xp, spt, sptl, beta, j1, j2, ifig)
        end
    end
    return ifig
end

function make_support(xp, beta, vx, npt = 6)

    # Get the support points
    sp = support([copy(r) for r in eachrow(xp)], npt)

    # Sort the support points so the first dimension is increasing
    ii = sortperm([v[1] for v in sp])
    sp = [sp[i] for i in ii]

    # Move each support point 1 unit in the positive or negative
    # direction for a single covariate.
    ee = zeros(size(beta, 1))
    ee[vx] = 1
    spt, sptl = [], []
    for (j, s) in enumerate(sp)
        a = string("ABCDEFGH"[j])
        push!(spt, s + beta' * ee)
        push!(sptl, a)
        push!(spt, s - beta' * ee)
        push!(sptl, "$(a)'")
    end

    return spt, sptl
end

function plots_flr(sex, X, Xp, ppy, fr, grx, vnames, ifig)

    sexs = sex == 2 || sex == "Female" ? "Female" : "Male"

    # Loop over the factors
    for k in eachindex(X)

        u = Xp[k] * fr.beta[k]
        is_binary = length(unique(Xp[k])) == 2

        # Plot the X-side factor loadings
        if !is_binary
            PyPlot.clf()
            PyPlot.title(sexs)
            PyPlot.grid(true)
            PyPlot.plot(grx[k], u, "-")
            PyPlot.xlabel(@sprintf("%s Z-score", vnames[k]), size = 15)
            PyPlot.ylabel(@sprintf("%s PC score", vnames[k]), size = 15)
            PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
            ifig += 1
        end

        # Plot the Q-side factor loadings
        vv = fr.v[:, k]
        if is_binary
            uu = sort(unique(X[k]))
            vv = fr.beta[k][1] * (uu[2] - uu[1]) * fr.v[:, k]
        end
        PyPlot.clf()
        PyPlot.title(sexs)
        PyPlot.grid(true)
        PyPlot.plot(ppy, vv, "-")
        if minimum(vv) > 0
            PyPlot.ylim(bottom = 0)
        end
        if maximum(vv) < 0
            PyPlot.ylim(top = 0)
        end
        PyPlot.xlabel("SBP probability points", size = 15)
        PyPlot.ylabel(@sprintf("%s loading", vnames[k]), size = 15)
        PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
        ifig += 1

        # Plot the rank-1 matrix for factor k.
        if !is_binary
            mm = u * fr.v[:, k]'
            mx = maximum(abs, mm)
            PyPlot.clf()
            PyPlot.title(sexs)
            xx = ones(length(grx[k])) * ppy'
            yy = grx[k] * ones(length(ppy))'
            a = PyPlot.contourf(xx, yy, mm, cmap = "bwr", vmin = -mx, vmax = mx)
            PyPlot.contour(xx, yy, mm, colors = "grey", vmin = -mx, vmax = mx)
            PyPlot.colorbar(a)
            PyPlot.xlabel("SBP quantiles", size = 15)
            PyPlot.ylabel(@sprintf("%s Z-score", vnames[k]), size = 15)
            PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
            ifig += 1
        end
    end

    return ifig
end
