using Statistics

# Attach a Z-scored version of the variable named 'vn' to
# the dataframe df.
function zscore!(df::AbstractDataFrame, vn::Symbol)
    vz = Symbol(string(vn) * "_z")
    md = median(df[:, vn])
    mad = median(abs.(df[:, vn] .- md))
    df[!, vz] = (df[:, vn] .- md) ./ (1.48 * mad)
end
