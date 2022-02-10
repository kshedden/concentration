using DataFrames, CSV, CodecZlib, Printf, UnicodePlots, LinearAlgebra
using Distributions

include("qreg_nn.jl")
include("flr_reg.jl")

dx = []
for x in ["DEMO", "BPX", "BMX"]
    d = open(@sprintf("%s_I.csv.gz", x)) do io
        CSV.read(GzipDecompressorStream(io), DataFrame)
    end
    push!(dx, d)
end

df = outerjoin(dx[1], dx[2], on = :SEQN)
df = outerjoin(df, dx[3], on = :SEQN)

da = df[:, [:RIAGENDR, :RIDAGEYR, :BPXSY1, :BMXBMI]]
da = da[completecases(da), :]

sex = 2
da = da[da.RIAGENDR .== sex, :]
da = select(da, Not(:RIAGENDR))

for a in [:RIDAGEYR, :BMXBMI]
	da[:, Symbol(string(a) * "_z")] = (da[:, a] .- mean(da[:, a])) ./ std(da[:, a])
end

y = Vector{Float64}(da[:, :BPXSY1])
X0 = Matrix{Float64}(da[:, [:RIDAGEYR, :BMXBMI]])
nn = qreg_nn(y, X0)
yq = zeros(length(y), 9)
yqm = zeros(9)
pp = collect(range(0.1, 0.9, length=9))
for j in 1:9
	yq[:, j] = fit(nn, pp[j], 0.1)
	yqm[j] = mean(yq[:, j])
	yq[:, j] .-= yqm[j]
end

function genbasis(x, b, s)

    # Basis centers
    c = if b == 1
        [mean(x)]
    else
        range(minimum(x), maximum(x), length = b)
    end

    B = zeros(length(x), b)
    fl = []
    for j = 1:b
        f = function (x)
            y = (x .- c[j]) / s
            return exp.(-y .^ 2 / 2)
        end
        B[:, j] = f(x)
        push!(fl, f)
    end

    function g(z)
        x = zeros(length(z), length(fl))
        for j in eachindex(fl)
            x[:, j] = fl[j](z)
        end
        return x
    end

    return B, g
end

X, Xp = Vector{Matrix{Float64}}(), Vector{Matrix{Float64}}()
for x0 in eachcol(X0)
	B, g = genbasis(x0, 5, std(x0))
	push!(X, B)
	pp = collect(range(0.01, 0.99, length=100))
	qq = quantile(Normal(mean(x0), std(x0)), pp)
	push!(Xp, g(qq))
end

fr = fitlr(X, Xp, yq, 0*ones(2), 0*ones(2))
