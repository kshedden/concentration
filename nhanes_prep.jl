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
