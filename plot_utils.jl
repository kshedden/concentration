using PyPlot, SupportPoints, NearestNeighbors, QuantileNN

# Plot two X-side scores against each other as a biplot.
function plotxx(sex, vnames, vx, xp, sp, spl, spt, sptl, beta, j1, j2, ifig)

    PyPlot.clf()
    PyPlot.title(@sprintf("%s %s contrasts", sex, vnames[vx]))
    PyPlot.grid(true)
    PyPlot.plot(xp[:, j1], xp[:, j2], "o", alpha = 0.2, rasterized = true)

	# Draw a grey line connecting each pair of contrasting points
	cpa = PyPlot.matplotlib.patches.ConnectionPatch
	ax = PyPlot.gca()
	for k in 1:size(sp, 2)
		j = 2*(k-1) + 1
		xs = [spt[j1, j], spt[j1, j+1]]
		ys = [spt[j2, j], spt[j2, j+1]]
		rr = sqrt(40)
		PyPlot.plot(xs[1:1], ys[1:1], "+", color="grey", mfc="none", ms=rr, zorder=10)
		PyPlot.plot(xs[1:1], ys[1:1], "o", color="grey", mfc="none", ms=rr, zorder=10)
		PyPlot.plot(xs[2:2], ys[2:2], "_", color="grey", mfc="none", ms=rr, zorder=10)
		PyPlot.plot(xs[2:2], ys[2:2], "o", color="grey", mfc="none", ms=rr, zorder=10)

		# https://stackoverflow.com/questions/58490203/matplotlib-plot-a-line-with-open-markers-where-the-line-is-not-visible-within
    	cp = cpa((xs[1], ys[1]), (xs[2], ys[2]), 
                 coordsA="data", coordsB="data", axesA=ax, axesB=ax,
                 shrinkA=rr/2, shrinkB=rr/2,
                 color="grey", zorder=10)
        ax.add_patch(cp)
	end

	# Draw a label for each contrasting pair
    for (k, z) in enumerate(eachcol(sp))
        PyPlot.text(z[j1], z[j2], spl[k], size = 14, ha = "center", va = "center", zorder=11)
    end

    # Make it a biplot
    bs = 2 * beta[:, [j1, j2]]
    for j in eachindex(vnames)
        PyPlot.gca().annotate(
            vnames[j],
            xytext = (bs[j, 1], bs[j, 2]),
            xy = (0, 0),
            arrowprops = Dict(:arrowstyle => "<-", :shrinkA => 0, :shrinkB => 0),
            ha = "center",
            va = "center",
        )
    end

    PyPlot.ylabel(@sprintf("Covariate factor %d score", j2), size = 15)
    PyPlot.xlabel(@sprintf("Covariate factor %d score", j1), size = 15)
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    ifig += 1
end

function plot_qtraj(sex, npc, vname, spt, sptl, kt, xmat, qhc, nnb, ifig)

    rsltp = []
    PyPlot.clf()
    PyPlot.axes([0.12, 0.12, 0.75, 0.8])
    PyPlot.title(@sprintf("%s %s contrasts", sex, vname))
    for (j, z) in enumerate(eachcol(spt))

        # Nearest neighbors of the support point in the projected
        # X-space.
        ii, _ = knn(kt, z, nnb)

        # Store the x-variable means corresponding to
        # each support point.
        row = [sex, npc, sptl[j], mean(xmat[ii, :], dims = 1)...]
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

function plot_qtraj_diff(sex, npc, vname, sp, spl, spt, sptl, kt, xmat, qhc, nnb, ifig)

    rsltp = []
    PyPlot.clf()
    PyPlot.axes([0.12, 0.12, 0.75, 0.8])
    PyPlot.title(@sprintf("%s %s contrasts", sex, vname))
    for j = 1:size(sp, 2)

        z1 = spt[:, 2*(j-1)+1]
        z2 = spt[:, 2*(j-1)+2]

        # Nearest neighbors of the support point in the projected
        # X-space.
        ii1, _ = knn(kt, z1, nnb)
        ii2, _ = knn(kt, z2, nnb)

        qp1 = mean(qhc[ii1, :], dims = 1)
        qp2 = mean(qhc[ii2, :], dims = 1)
        qp = vec(qp1) - vec(qp2)

        PyPlot.plot(pp, qp, "-", label = spl[j])
    end
    ha, lb = PyPlot.gca().get_legend_handles_labels()
    leg = PyPlot.figlegend(ha, lb, "center right")
    leg.draw_frame(false)
    PyPlot.grid(true)
    PyPlot.xlabel("Probability", size = 15)
    PyPlot.ylabel("SBP quantile difference", size = 15)
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))

    return ifig + 1
end

function plotxx_all(sex, vnames, vx, xp, sp, spl, spt, sptl, beta, ifig)
    for j2 = 1:size(beta, 2)
        for j1 = 1:j2-1
            ifig = plotxx(sex, vnames, vx, xp, sp, spl, spt, sptl, beta, j1, j2, ifig)
        end
    end
    return ifig
end

function make_support(xp, beta, vx, npt = 6)

    # Get the support points
    sp = supportpoints(copy(xp'), npt)

    # Sort the support points so the first dimension is increasing
    ii = sortperm(sp[1, :])
    sp = sp[:, ii]

    # Move each support point 1 unit in the positive or negative
    # direction for a single covariate.
    spt, sptl, spl = zeros(size(sp, 1), 2 * npt), String[], String[]
    for (j, s) in enumerate(eachcol(sp))
        # Only 10 distinct labels
        a = string("ABCDEFGHIJ"[1+((j-1)%10)])
        spt[:, 2*j-1] = s + beta[vx, :]
		push!(spl, a)
        push!(sptl, "$(a)+")
        spt[:, 2*j] = s - beta[vx, :]
        push!(sptl, "$(a)−")
    end

    return sp, spl, spt, sptl
end

function loclin2d(
    y::AbstractVector,
    X::AbstractMatrix,
    xlim::AbstractVector,
    ylim::AbstractVector,
    nx::Int,
    ny::Int,
    bw::AbstractVector,
)
    @assert length(y) == size(X, 1)
    xs = collect(range(xlim[1], xlim[2], length = nx))
    ys = collect(range(ylim[1], ylim[2], length = ny))

    n = length(y)
    xx = ones(n, 3)
    ww = zeros(n)
    dx2 = zeros(n)
    dy2 = zeros(n)

    z = zeros(ny, nx)
    for i in eachindex(xs)
        xx[:, 2] = (X[:, 1] .- xs[i]) / bw[1]
        dx2 .= xx[:, 2] .^ 2
        for j in eachindex(ys)
            xx[:, 3] = (X[:, 2] .- ys[j]) / bw[2]
            dy2 .= xx[:, 3] .^ 2
            ww .= exp.(-(dx2 + dy2) / 2)
            ww ./= sum(ww)
            b = (xx' * diagm(ww) * xx) \ (xx' * y)
            z[j, i] = b[1]
        end
    end
    return z
end

function diff_support(
    sp::AbstractMatrix,
    spt::AbstractMatrix,
    kt::KDTree,
    qhc::AbstractMatrix;
    nnb = 100,
)
    qpl = []
    for j = 1:size(sp, 2)
        # Paired points
        z1 = spt[:, 2*(j-1)+1]
        z2 = spt[:, 2*(j-1)+2]

        # Nearest neighbors of the support point in the projected
        # X-space.
        ii1, _ = knn(kt, z1, nnb)
        ii2, _ = knn(kt, z2, nnb)

        # Average the corresponding points in the Q-space
        qp1 = mean(qhc[ii1, :], dims = 1)[:]
        qp2 = mean(qhc[ii2, :], dims = 1)[:]
        push!(qpl, qp1 - qp2)
    end

    # The rows contain all of the difference profiles
    qm = hcat(qpl...)'
    qmm = mean(qm, dims = 1)[:]
    for j = 1:size(qm, 2)
        qm[:, j] .-= qmm[j]
    end

    u, s, v = svd(qm)

    # Flip the dominant factors for interpretability
    for j = 1:2
        if sum(v[:, j] .> 0) < sum(v[:, j] .< 0)
            v[:, j] .*= -1
            u[:, j] .*= -1
        end
    end

    return u[:, 1], qmm, v[:, 1]
end

function plot_qtraj_diffmap(
    sp::AbstractMatrix,
    spt::AbstractMatrix,
    kt::KDTree,
    qhc::AbstractMatrix,
    pp::AbstractVector,
    vname::String,
    sex::String,
    ifig::Int;
    nnb = 100,
)
    yy, ym, pcl = diff_support(sp, spt, kt, qhc; nnb = nnb)

    # Plot the mean
    PyPlot.clf()
    PyPlot.title(@sprintf("Manipulate %s %s", sex, vname))
    PyPlot.grid(true)
    PyPlot.plot(pp, ym)
    if minimum(ym) > 0
        PyPlot.ylim(ymin = 0)
    elseif maximum(ym) < 0
        PyPlot.ylim(ymax = 0)
    end
    PyPlot.xlabel("SBP quantiles", size = 15)
    PyPlot.ylabel("Mean", size = 15)
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    ifig += 1

    # Plot the loadings
    PyPlot.clf()
    PyPlot.title(@sprintf("Manipulate %s %s", sex, vname))
    PyPlot.grid(true)
    PyPlot.plot(pp, pcl)
    if minimum(pcl) > 0
        PyPlot.ylim(ymin = 0)
    elseif maximum(pcl) < 0
        PyPlot.ylim(ymax = 0)
    end
    PyPlot.xlabel("SBP quantiles", size = 15)
    PyPlot.ylabel("Loading", size = 15)
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    ifig += 1

    for j1 = 1:size(sp, 1)
        for j2 = j1+1:size(sp, 1)

            # Select the dominant components
            xx = sp[[j1, j2], :]'
            xlim = [minimum(xx[:, 1]), maximum(xx[:, 1])]
            ylim = [minimum(xx[:, 2]), maximum(xx[:, 2])]

            # Create a 2d map corresponding to two of the x-variates.
            nn = 20
            bw = Float64[2, 2]
            z = loclin2d(yy, xx, xlim, ylim, nn, nn, bw)

            # Plot the scores
            PyPlot.clf()
            PyPlot.title(@sprintf("Manipulate %s %s", sex, vname))
            mx = maximum(abs, z)
            x0 = ones(size(z, 1)) * range(xlim[1], xlim[2], length = nn)'
            y0 = range(ylim[1], ylim[2], length = nn) * ones(size(z, 2))'
            a = PyPlot.contourf(x0, y0, z, cmap = "bwr", vmin = -mx, vmax = mx)
            PyPlot.contour(x0, y0, z, colors = "grey", vmin = -mx, vmax = mx)
            PyPlot.colorbar(a)

            PyPlot.xlabel(@sprintf("Covariate factor %d", j1), size = 15)
            PyPlot.ylabel(@sprintf("Covariate factor %d", j2), size = 15)
            PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
            ifig += 1
        end
    end

    return ifig
end

function plots_flr(sex, X, Xp, ppy, fr, grx, vnames, ifig)

    # Loop over the factors
    for k in eachindex(X)

        u = Xp[k] * fr.beta[k]
        is_binary = length(unique(Xp[k])) == 2

        # Plot the X-side factor loadings
        if !is_binary
            PyPlot.clf()
            PyPlot.title(sex)
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
        PyPlot.title(sex)
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
            PyPlot.title(sex)
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

function plot_tensor(
    mm::AbstractMatrix,
    pp::AbstractVector,
    xname::String,
    title::String,
    ifig::Int,
)

    # In the provided matrix, the origin is in the upper left corner.
    # We want to plot as if in quadrant 1.
    mm = reverse(mm, dims = 1)

    mx = maximum(abs, mm)
    PyPlot.clf()
    PyPlot.imshow(
        mm,
        interpolation = "nearest",
        cmap = "bwr",
        origin = "upper",
        vmin = -mx,
        vmax = mx,
        extent = [0, 1, 0, 1],
    )
    PyPlot.colorbar()
    PyPlot.title(title)
    PyPlot.xlabel("SBP quantiles", size = 15)
    PyPlot.ylabel(@sprintf("%s quantiles", xname), size = 15)
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    return ifig + 1
end
