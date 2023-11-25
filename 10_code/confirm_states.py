"""checks states for sanity check"""
import pandas as pd

csv_url = "https://raw.githubusercontent.com/wpinvestigative/arcos-api/master/data/county_fips.csv?raw=true"

df = pd.read_csv(csv_url)

# Get the unique states in the "BUYER_STATE" column
unique_states = df["BUYER_STATE"].unique()

# Self-made list of states
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


# Print the unique states
print(unique_states)

# Find missing states
missing_states = set(unique_states) - set(STATES)

# Print the missing states
print("Missing States:", missing_states)

check2 = {
    "ID",
    "PR",
    "NC",
    "AL",
    "MT",
    "OR",
    "MN",
    "FL",
    "KY",
    "MI",
    "SC",
    "WI",
    "WA",
    "AE",
    "NE",
    "MP",
    "WY",
    "AR",
    "DE",
    "AK",
    "LA",
    "IA",
    "TX",
    "KS",
    "NJ",
    "ME",
    "NY",
    "WV",
    "IL",
    "MD",
    "MO",
    "SD",
    "OH",
    "AZ",
    "NH",
    "CT",
    "ND",
    "PA",
    "CA",
    "MA",
    "RI",
    "NV",
    "CO",
    "VI",
    "HI",
    "PW",
    "OK",
    "DC",
    "TN",
    "MS",
    "VT",
    "NM",
    "VA",
    "IN",
    "GA",
    "GU",
    "UT",
}

missing_states = set(check2) - set(STATES)
print("Missing States:", missing_states)
