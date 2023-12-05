import dask.dataframe as dd

# Specify the columns that are required
required_cols = [
    "BUYER_STATE",
    "BUYER_ZIP",
    "BUYER_COUNTY",
    "TRANSACTION_DATE",
    "DOSAGE_UNIT",
    "CALC_BASE_WT_IN_GM",
    "MME",
]

# Read the data into Dask DataFrame with the required columns
consumption_df_dask = dd.read_csv(
    "/Users/aaryadesai/Downloads/arcos_all_washpost.tsv",
    sep="\t",
    usecols=required_cols,
)

# Add the year and month columns from the TRANSACTION_DATE column using the .dt() function
consumption_df_dask["Year"] = dd.to_datetime(
    consumption_df_dask["TRANSACTION_DATE"], format="%Y-%m-%d"
).dt.year
consumption_df_dask["Month"] = dd.to_datetime(
    consumption_df_dask["TRANSACTION_DATE"], format="%Y-%m-%d"
).dt.month

# Group the data by the required columns and sum the MME column for consumption calculations
consumption_df_dask = (
    consumption_df_dask.groupby(["BUYER_COUNTY", "BUYER_STATE", "Year"])["MME"]
    .sum()
    .reset_index()
)

# Trigger the computation by calling the .compute() method
final_df_dask = consumption_df_dask.compute()

# To check if the df is correct and the code is working
final_df_dask.head()
