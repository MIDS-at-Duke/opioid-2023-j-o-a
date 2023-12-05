import pandas as pd
import numpy as np
import warnings

pd.set_option("mode.copy_on_write", True)
warnings.simplefilter(action="ignore", category=FutureWarning)
import dask.dataframe as dd
from dask.distributed import Client

client = Client()

# used in group project
cols_needed = [
    "BUYER_STATE",
    "BUYER_ZIP",
    "BUYER_COUNTY",
    "TRANSACTION_DATE",
    "DOSAGE_UNIT",
    "CALC_BASE_WT_IN_GM",
    "MME",
]

opiod_project_df = dd.read_csv(
    "../00_data/arcos_all_washpost.tsv",
    sep="\t",
    usecols=cols_needed,
)

opiod_project_df["Year"] = dd.to_datetime(
    opiod_project_df["TRANSACTION_DATE"], format="%Y-%m-%d"
).dt.year
opiod_project_df["Month"] = dd.to_datetime(
    opiod_project_df["TRANSACTION_DATE"], format="%Y-%m-%d"
).dt.month

opiod_project_df = (
    opiod_project_df.groupby(["BUYER_COUNTY", "BUYER_STATE", "Year"])["MME"]
    .sum()
    .reset_index()
)

output_df = opiod_project_df.compute()

print("The head of the df is: ")
output_df.head()
# here I would save the file as a parquet file for the project but as this is a toy problem there is no need
