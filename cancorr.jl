using LinearAlgebra, Statistics, Dimred

include("qreg_nn.jl")

function cancorr(X, Y)
    n = size(X, 1)

    X = copy(X)
    for x in eachcol(X)
        x .-= mean(x)
    end

    Y = copy(Y)
    for y in eachcol(Y)
        y .-= mean(y)
    end

    cx = cov(X)
    Ux = cholesky(cx).U

    cy = cov(Y)
    Uy = cholesky(cy).U

    Sxy = X' * Y / n
    M = Ux' \ Sxy / Uy

    u, s, v = svd(M)
    beta = Ux \ u
    eta = Uy \ v

    for b in eachcol(beta)
        b ./= norm(b)
    end

    for e in eachcol(eta)
        e ./= norm(e)
    end

    return (beta, eta, s)
end

function qnn_cca(y, xmat, npc, nperm)

    xmat = copy(xmat)
    center!(xmat)

    sp = []
    eta1, beta1, s1 = nothing, nothing, nothing
    qhc1, xmat1 = nothing, nothing
    for k = 1:nperm+1

        # On the first iteration, analyze the actual data.
        # Subsequent iterations work with permuted data.
        if k > 1
            ii = collect(1:size(xmat, 1))
            shuffle!(ii)
            xmat = xmat[ii, :]
        end

        # Get the fitted quantiles
        qr = qreg_nn(y, xmat)
        qh = zeros(length(y), length(pp))
        for (j, p) in enumerate(pp)
            _ = fit(qr, p, 0.1)
            qh[:, j] = qr.fit
        end

        # Center the fitted quantiles and calculate their dominant factors.
        qhc = copy(qh)
        center!(qhc)
        upc, spc, vpc = svd(qhc)

        # Reduce the quantiles to PC's
        qhcpc = qhc * vpc[:, 1:npc]

        # Do PCA->CCA
        eta, beta, s = cancorr(qhcpc, xmat)
        eta = vpc[:, 1:npc] * eta
        betar1, etar1 = rotate_hom(beta, eta, xmat, qhc)

        if k == 1
            # Actual result
            eta1, beta1, s1 = etar1, betar1, s
            qhc1, xmat1 = qhc, xmat
        else
            # Permuted result
            push!(sp, s)
        end
    end

    sp = hcat(sp...)

    return (eta1, beta1, qhc1, xmat1, s1, sp)
end

function qnn_mpsir(y, xmat, npc, nmp, nperm)

    xmat = copy(xmat)
    center!(xmat)

    sp = []
    eta1, beta1, s1 = nothing, nothing, nothing
    qhc1, xmat1 = nothing, nothing
    eigx, eigy = nothing, nothing
    for k = 1:nperm+1

        # On the first iteration, analyze the actual data.
        # Subsequent iterations work with permuted data.
        if k > 1
            ii = collect(1:size(xmat, 1))
            shuffle!(ii)
            xmat = xmat[ii, :]
        end

        # Get the fitted quantiles
        qr = qreg_nn(y, xmat)
        qh = zeros(length(y), length(pp))
        for (j, p) in enumerate(pp)
            _ = fit(qr, p, 0.1)
            qh[:, j] = qr.fit
        end
        qhc = copy(qh)
        center!(qhc)

        # Center the fitted quantiles and calculate their dominant factors.
        qhc = copy(qh)
        center!(qhc)
        upc, spc, vpc = svd(qhc)

        # Reduce the quantiles to PC's
        qhcpc = qhc * vpc[:, 1:npc]

        mp = MPSIR(qhcpc, xmat)
        fit!(mp, 2, 2; nslicex = 10, nslicey = 10)
        etar1, betar1 = coef(mp)
        etar1 = vpc[:, 1:npc] * etar1
        eigy1, eigx1 = eig(mp)
        betar1, etar1 = rotate_hom(betar1, etar1, xmat, qhc)

        if k == 1
            # Actual result
            eta1, beta1 = etar1, betar1
            qhc1, xmat1 = qhc, xmat
            eigx, eigy = eigx1, eigy1
        else
            # Permuted result
            #push!(sp, s)
        end
    end

    #sp = hcat(sp...)

    return (eta1, beta1, qhc1, xmat1, eigx, eigy)
end

function center!(x)
    for y in eachcol(x)
        y .-= mean(y)
    end
end

function canonical_angles(A, B)
    A, _, _ = svd(A)
    B, _, _ = svd(B)
    _, s, _ = svd(A' * B)
    return s
end

#=
Rotate a canonical correlation solutio so that the first
factor approximately represents homoscedastic changes in
the quantiles.

'X' and 'Y' are data matrices.  A' contains coefficients 
for 'X' and 'B' contains coefficients for 'Y', so that 
'XA' and 'YB' are linearly transformed versions of 'X' and 
'Y'.

We wish to find new coefficient matrices A1 and B1
that span the same subspaces as A and B, so that span(X*A) = 
span(X*A1) and span(Y*B) = span(Y*B1).  Also we want
the first column of Y*B1 to be approximately
constant.  In addition, we want the scores in 
X*A1 to be standardized and uncorrelated.
=#
function rotate_hom(A, B, X, Y)

    n = size(X, 1)
    X = copy(X)
    center!(X)
    Sxx = X' * X / n
    Asy = A' * Sxx * A

    Y = copy(Y)
    center!(Y)
    Syy = Y' * Y / n

    q1, q2 = size(B)
    u, s, v = svd(B)
    oo = ones(q1)
    c = v * diagm(1 ./ s) * u' * oo
    C = zeros(q2, q2)
    C[:, 1] = c ./ sqrt(c' * Asy * c)
    for j = 2:q2
        b = zeros(q2)
        b[j] = 1
        for k = 1:j-1
            b .-= (b' * Asy * C[:, k]) * C[:, k]
        end
        b ./= sqrt(b' * Asy * b)
        C[:, j] = b
    end

    return (A * C, B * C)
end

function check_rotate_hom()

    X = randn(200, 5)
    Y = randn(200, 8)
    A = randn(5, 3)
    B = randn(8, 3)
    B[:, 2] = 1 .+ 0.1 * randn(8)

    A1, B1 = rotate_hom(A, B, X, Y)

    # The rotated scores should have unit variance
    # and zero pairwise correlation.
    C = cov(X * A1)
    C *= 199 / 200
    @assert maximum(abs, C - I(3)) < 1e-8

    # The first right loading vector should be approximately
    # constant.
    @assert std(B1[:, 1]) / mean(B1[:, 1]) < 0.2
end

function check_cancorr()
    n = 300
    X = randn(n, 3)
    Y = randn(n, 3)
    Y[:, 2] = X[:, 1] - X[:, 3] + randn(n)
    Y[:, 3] = X[:, 2] + randn(n)
    beta, eta, s = cancorr(X, Y)
end
