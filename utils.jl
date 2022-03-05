using Statistics

# Attach a Z-scored version of the variable named 'vn' to
# the dataframe df.
function zscore!(df::AbstractDataFrame, vn::Symbol; mode = "quantile")
    vz = Symbol(string(vn) * "_z")

    if mode == "moment"
        mn = mean(df[:, vn])
        df[!, vz] = (df[:, vn] .- mn) ./ std(df[:, vn])
    elseif mode == "quantile"
        md = median(df[:, vn])
        mad = median(abs.(df[:, vn] .- md))
        df[!, vz] = (df[:, vn] .- md) ./ (1.48 * mad)
    else
        error("!!")
    end
end
