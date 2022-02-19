using DataFrames, CSV, CodecZlib, Printf, Statistics

vnames = ["Age", "BMI", "Height"]
dnames = [:RIDAGEYR, :BMXBMI, :BMXHT]

# Load the NHANES data files
dx = []
for x in ["DEMO", "BPX", "BMX"]
    d = open(@sprintf("%s_I.csv.gz", x)) do io
        CSV.read(GzipDecompressorStream(io), DataFrame)
    end
    push!(dx, d)
end

# Merge the NHANES data files
df = outerjoin(dx[1], dx[2], on = :SEQN)
df = outerjoin(df, dx[3], on = :SEQN)

# Restrict the the variables of interest and complete cases
vn = vcat([:RIAGENDR, :BPXSY1], dnames)
da = df[:, vn]
da = da[completecases(da), :]

# Restrict to people aged 18 and over.
da = da[da.RIDAGEYR.>=18, :]

# Attach a Z-scored version of the variable named 'vn' to
# the dataframe df.
function zscore!(df, vn)
    vz = Symbol(string(vn) * "_z")
    md = median(df[:, vn])
    mad = median(abs.(df[:, vn] .- md))
    df[!, vz] = (df[:, vn] .- md) ./ (0.67 * mad)
end

function select_nhanes(sex, age1, age2)
    dx = da[da.RIAGENDR.==sex, :]
    dx = dx[:, [:BPXSY1, :BMXBMI, :RIDAGEYR, :BMXHT]]
    dx = dx[completecases(dx), :]
    dx = filter(row -> row.RIDAGEYR >= age1 && row.RIDAGEYR <= age2, dx)
    zscore!(dx, :RIDAGEYR)
    zscore!(dx, :BMXBMI)
    zscore!(dx, :BMXHT)
    return dx
end
