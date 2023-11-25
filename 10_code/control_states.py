"""find the possible control states for each treatment state"""
import pyarrow.parquet as pq
import pandas as pd
import os

TREATMENT_STATES = ["TX", "FL", "WA"]
# control states must have years such that there are before the intervention and after
# Texas BEFORE AND AFTER 2007
# Florida BEFORE AND AFTER 2010
# Washington BEFORE AND AFTER 2012

# Naive solution would be to check all the years for each state
# What about the other requirements for it to be a control state?
# check how many counties it has?
# need to merge data to find similar trend of opiod deaths so for now just check number of years
# it has for each state and number of counties

PRE_DEFINED_STATES = [
    "AL",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "DC",
]

# Specify the path to the folder containing Parquet files
folder_path = "../00_data/states/"


def find_control_states_prep(pre_defined_states):
    """
    Prep to find the control states for each treatment state
    """
    for state in pre_defined_states:
        year = set()
        counties = set()
        print(state)
        parquet_file = os.path.join(folder_path + str(state) + ".parquet")
        # Load the Parquet file
        table = pq.read_table(parquet_file)

        # Convert the table to a pandas DataFrame
        df = table.to_pandas()
        # find years
        year.update(df["YEAR"])
        print(year)
        # find counties
        counties.update(df["BUYER_COUNTY"])
        print(counties)


if __name__ == "__main__":
    find_control_states_prep(PRE_DEFINED_STATES)
