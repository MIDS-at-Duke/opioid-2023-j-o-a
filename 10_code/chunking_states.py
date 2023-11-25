"""grabs all the states and pareses them into individual parquet files"""
import pandas as pd

input_file = "../00_data/arcos_all_washpost.tsv"
output_dir = "../00_data/states/"

"""pre defined states list"""
STATES = [
    "AK",
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


# List of columns to keep
columns_to_keep = [
    "BUYER_STATE",
    "BUYER_ZIP",
    "BUYER_COUNTY",
    "DRUG_NAME",
    "YEAR",
    "MONTH",
    "CALC_BASE_WT_IN_GM",
    "DOSAGE_UNIT",
    "MME",
    "MME_Conversion_Factor",
]


def process_state(state, chunks):
    """saves state as parquet file"""
    state_df = []
    for i, chunk in enumerate(chunks):
        # converts columns to datetime for later conversion
        chunk["TRANSACTION_DATE"] = pd.to_datetime(
            chunk["TRANSACTION_DATE"], errors="coerce"
        )
        # splits TRANSACTION_DATE to two columns year and month
        chunk["YEAR"] = chunk["TRANSACTION_DATE"].dt.year
        chunk["MONTH"] = chunk["TRANSACTION_DATE"].dt.month
        # keeps only the columns we discussed
        state_chunk = chunk[columns_to_keep].loc[chunk["BUYER_STATE"] == state]
        if not state_chunk.empty:
            state_df.append(state_chunk)
            print(i)
            # print(state_df)
    # grabs all relevant rows related to the searched state
    pd.concat(state_df, ignore_index=True).to_parquet(
        output_dir + state + ".parquet", index=False
    )


def grab_states(chunks):
    """adds to a set of states to check"""
    states = set()
    for i, chunk in enumerate(chunks):
        states.update(chunk["BUYER_STATE"])
        print(states)
    return states


if __name__ == "__main__":
    path = "../00_data/arcos_all_washpost.tsv"

    for i in STATES:
        print(i)
        # need to read this again to remake iterator
        chunks = pd.read_table(path, chunksize=10000000, low_memory=False)
        process_state(i, chunks)
