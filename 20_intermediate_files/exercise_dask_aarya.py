import dask.dataframe as dd

required_cols = [
    "BUYER_STATE",
    "BUYER_ZIP",
    "BUYER_COUNTY",
    "TRANSACTION_DATE",
    "DOSAGE_UNIT",
    "CALC_BASE_WT_IN_GM",
    "MME",
]

consumption_df_dask = dd.read_csv(
    "/Users/aaryadesai/Downloads/arcos_all_washpost.tsv",
    sep="\t",
    usecols=required_cols,
)

consumption_df_dask["Year"] = dd.to_datetime(
    consumption_df_dask["TRANSACTION_DATE"], format="%Y-%m-%d"
).dt.year
consumption_df_dask["Month"] = dd.to_datetime(
    consumption_df_dask["TRANSACTION_DATE"], format="%Y-%m-%d"
).dt.month

consumption_df_dask = (
    consumption_df_dask.groupby(["BUYER_COUNTY", "BUYER_STATE", "Year"])["MME"]
    .sum()
    .reset_index()
)

final_df_dask = consumption_df_dask.compute()

print("The head of the df is: ")
final_df_dask.head()
