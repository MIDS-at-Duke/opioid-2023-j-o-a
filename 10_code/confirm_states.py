import pandas as pd

csv_url = "https://raw.githubusercontent.com/wpinvestigative/arcos-api/master/data/county_fips.csv?raw=true"

df = pd.read_csv(csv_url)

# Get the unique states in the "BUYER_STATE" column
unique_states = df["BUYER_STATE"].unique()

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
