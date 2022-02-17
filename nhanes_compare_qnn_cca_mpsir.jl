using LinearAlgebra, CSV, DataFrames

include("nhanes_prep.jl")
include("cancorr.jl")

# Probability points for SBP quantiles
pp = range(0.1, 0.9, length = 9)

eta_cca = open("eta_cca.csv") do io
    CSV.read(io, DataFrame)
end

beta_cca = open("beta_cca.csv") do io
    CSV.read(io, DataFrame)
end

eta_mpsir = open("eta_mpsir.csv") do io
    CSV.read(io, DataFrame)
end

beta_mpsir = open("beta_mpsir.csv") do io
    CSV.read(io, DataFrame)
end

function get_xy(sex)
    dx = select_sex(sex)
    y = dx[:, :BPXSY1]
    y = Vector{Float64}(y)
    dx = select_sex(sex)
    xmat = dx[:, [:RIDAGEYR_z, :BMXBMI_z, :BMXHT_z]]
    xmat = Matrix{Float64}(xmat)
    return (y, xmat)
end

for sex in [2, 1]

    y, xmat = get_xy(sex)

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

    sexs = sex == 2 ? "Female" : "Male"

    bcca = filter(row -> row.Sex == sexs, beta_cca)
    bcca = bcca[:, 3:end]
    bcca = Array{Float64}(bcca)'
    bmps = filter(row -> row.Sex == sexs, beta_mpsir)
    bmps = bmps[:, 3:end]
    bmps = Array{Float64}(bmps)'
    a = canonical_angles(xmat * bcca, xmat * bmps)
    println(a)

    ecca = filter(row -> row.Sex == sexs, eta_cca)
    ecca = ecca[:, 3:end]
    ecca = Array{Float64}(ecca)'
    emps = filter(row -> row.Sex == sexs, eta_mpsir)
    emps = emps[:, 3:end]
    emps = Array{Float64}(emps)'
    a = canonical_angles(qhc * ecca, qhc * emps)
    println(a)

end
