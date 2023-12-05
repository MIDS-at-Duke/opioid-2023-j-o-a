import pandas as pd
import numpy as np
import warnings
import dask.dataframe as dd
from dask.distributed import Client

warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize Dask client
dask_client = Client()

# Define columns to be used
required_columns = [
    "BUYER_STATE",
    "BUYER_ZIP",
    "BUYER_COUNTY",
    "TRANSACTION_DATE",
    "DOSAGE_UNIT",
    "CALC_BASE_WT_IN_GM",
    "MME",
]

# Load data using Dask
data_df = dd.read_csv(
    "../00_data/arcos_all_washpost.tsv",
    delimiter="\t",
    usecols=required_columns,
)

# Extract Year and Month from TRANSACTION_DATE
data_df["Year"] = dd.to_datetime(data_df["TRANSACTION_DATE"], format="%Y-%m-%d").dt.year
data_df["Month"] = dd.to_datetime(data_df["TRANSACTION_DATE"], format="%Y-%m-%d").dt.month

# Group by BUYER_COUNTY, BUYER_STATE, Year and sum MME
grouped_df = data_df.groupby(["BUYER_COUNTY", "BUYER_STATE", "Year"])["MME"].sum().reset_index()

# Compute the Dask dataframe to get the result in pandas dataframe
opioid_county = grouped_df.compute()

opioid_county.sample(10)

# We could now save the resulting file, but we've already done that manually
