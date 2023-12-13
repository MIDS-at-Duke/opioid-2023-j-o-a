import pandas as pd
import os

# List of state abbreviations
states = [
    "AL",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "DC",
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
]

# Initialize an empty DataFrame to hold the aggregated data
aggregated_data_years = pd.DataFrame()
aggregated_data_months = pd.DataFrame()

missing_states = []

# Loop over the state abbreviations
for state in states:
    # Construct the filename and read the file into a DataFrame
    filename = f"../00_data/states/{state}.parquet"
    if os.path.exists(filename):
        df = pd.read_parquet(filename)

        # Group the DataFrame and sum 'MME' and 'CALC_BASE_WT_IN_GM'
        grouped_year = (
            df.groupby(["BUYER_STATE", "BUYER_COUNTY", "YEAR"])[
                ["MME", "CALC_BASE_WT_IN_GM"]
            ]
            .sum()
            .reset_index()
        )
        grouped_month = (
            df.groupby(["BUYER_STATE", "BUYER_COUNTY", "YEAR", "MONTH"])[
                ["MME", "CALC_BASE_WT_IN_GM"]
            ]
            .sum()
            .reset_index()
        )

        # Append the grouped data to the aggregated DataFrame
        aggregated_data_years = pd.concat([aggregated_data_years, grouped_year])

        aggregated_data_months = pd.concat([aggregated_data_months, grouped_month])
    else:
        missing_states.append(state)

# Write the aggregated DataFrame to a parquet file
aggregated_data_years.to_parquet("../00_data/aggregated_data_years.parquet")
aggregated_data_months.to_parquet("../00_data/aggregated_data_months.parquet")

aggregated_data_years = aggregated_data_years.rename(
    columns={
        "BUYER_STATE": "State",
        "BUYER_COUNTY": "County",
        "YEAR": "Year",
        "MONTH": "Month",
        "MME": "MME",
        "CALC_BASE_WT_IN_GM": "CALC_BASE_WT_IN_GM",
    }
)

aggregated_data_years = aggregated_data_years[
    (aggregated_data_years["Year"] >= 2002) & (aggregated_data_years["Year"] <= 2018)
]


population = pd.read_csv("../00_data/PopFips.csv")

population = population.rename(
    columns={"CTYNAME": "County", "Year": "Year", "STATE": "State"}
)

# Rename 'Dona Ana' to 'Doña ana'
aggregated_data_years.loc[
    aggregated_data_years["County"] == "DONA ANA", "County"
] = "Do̱a Ana"

# Rename 'Radford' to 'Radford City'
aggregated_data_years.loc[
    aggregated_data_years["County"] == "RADFORD", "County"
] = "Radford City"


# Remove 'District of Columbia'
aggregated_data_years = aggregated_data_years[
    aggregated_data_years["County"] != "DISTRICT OF COLUMBIA"
]


aggregated_data_years = aggregated_data_years[
    ~(
        (aggregated_data_years["County"] == "LA SALLE")
        & (aggregated_data_years["State"] == "LA")
    )
]


population.loc[
    (population["County"] == "Salem city") & (population["State"] == "VA"), "County"
] = "salem"


def clean_county(df):
    df["County"] = df["County"].str.lower()
    df["County"] = df["County"].str.replace(" parish", "")
    df["County"] = df["County"].str.replace("parish", "")
    df["County"] = df["County"].str.replace(" county", "")
    df["County"] = df["County"].str.replace("st.", "saint")
    df["County"] = df["County"].str.replace("st ", "saint")
    df["County"] = df["County"].str.replace("ste.", "sainte")
    df["County"] = df["County"].str.replace(" ", "")
    df["County"] = df["County"].str.replace("-", "")
    df["County"] = df["County"].str.replace("'", "")
    return df


population_clean = clean_county(population)
agg_years_clean = clean_county(aggregated_data_years)

# Rename 'saintjohnthebaptisaint' to 'saintjohnthebaptist'
population_clean.loc[
    population_clean["County"] == "saintjohnthebaptisaint", "County"
] = "saintjohnthebaptist"

population_counties = set(population_clean["County"])
aggregated_data_years_counties = set(agg_years_clean["County"])

only_in_population = population_counties - aggregated_data_years_counties
only_in_aggregated_data_years = aggregated_data_years_counties - population_counties


opioid_pop = agg_years_clean.merge(
    population_clean, how="left", on=["State", "County", "Year"], indicator=True
)

assert (opioid_pop["_merge"] == "both").all()

# write the merged file to parquet
opioid_pop.to_parquet("../00_data/opioid_pop_clean.parquet")


#####################################################################
#       We repeat the same process for months data for Texas        #
#####################################################################


aggregated_data_months = aggregated_data_months.rename(
    columns={
        "BUYER_STATE": "State",
        "BUYER_COUNTY": "County",
        "YEAR": "Year",
        "MONTH": "Month",
        "MME": "MME",
        "CALC_BASE_WT_IN_GM": "CALC_BASE_WT_IN_GM",
    }
)

aggregated_data_months = aggregated_data_months[
    (aggregated_data_months["Year"] >= 2002) & (aggregated_data_months["Year"] <= 2018)
]

# Rename 'Dona Ana' to 'Doña ana'
aggregated_data_months.loc[
    aggregated_data_months["County"] == "DONA ANA", "County"
] = "Do̱a Ana"

# Rename 'Radford' to 'Radford City'
aggregated_data_months.loc[
    aggregated_data_months["County"] == "RADFORD", "County"
] = "Radford City"


# Remove 'District of Columbia'
aggregated_data_months = aggregated_data_months[
    aggregated_data_months["County"] != "DISTRICT OF COLUMBIA"
]


aggregated_data_months = aggregated_data_months[
    ~(
        (aggregated_data_months["County"] == "LA SALLE")
        & (aggregated_data_months["State"] == "LA")
    )
]

agg_months_clean = clean_county(aggregated_data_months)

# population_counties = set(population_clean["County"])
# aggregated_data_years_counties = set(agg_years_clean["County"])

# only_in_population = population_counties - aggregated_data_years_counties
# only_in_aggregated_data_years = aggregated_data_years_counties - population_counties


opioid_pop_months = agg_months_clean.merge(
    population_clean, how="left", on=["State", "County", "Year"], indicator=True
)

assert (opioid_pop_months["_merge"] == "both").all()

# write the merged file to parquet
opioid_pop_months.to_parquet("../00_data/opioid_pop_months_clean.parquet")
