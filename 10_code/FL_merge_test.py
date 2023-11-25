"""
    This script merges the Florida parquet file with the raw mortality data and population data.
    If this works, we can automate it in the same way for all other states.
"""

import pandas as pd

# Loading Florida parquet file from 00_data/states
florida_data = pd.read_parquet("../00_data/states/FL.parquet")
florida_data.rename(columns={"BUYER_COUNTY": "County"}, inplace=True)
florida_data["County"] = florida_data["County"].str.lower()

# Loading raw mortality data from 00_data
mortality_data = pd.read_csv("../00_data/mortality_final.csv")

# Loading population data from 00_data
population_data = pd.read_csv("../00_data/PopFips.csv")
population_data.rename(columns={"CTYNAME": "County"}, inplace=True)
population_data["County"] = population_data["County"].str.lower()
population_data["County"] = population_data["County"].str.replace(" county", "")

# Subset population data to just FL
florida_pop_data = population_data[population_data["STATE"] == "FL"]
florida_mortality_data = mortality_data[mortality_data["State"] == "FL"]
florida_mortality_data["County"] = florida_mortality_data["County"].str.lower()
florida_mortality_data["County"] = florida_mortality_data["County"].str.replace(
    " county", ""
)

pop_mortality_data = florida_data.merge(
    florida_mortality_data, how="left", on=["County"], indicator=True
)

fl_finaldata = pop_mortality_data.merge(
    florida_pop_data, how="left", on=["County", "Year"]
)

"""
Filtering based on country population because we don't want counties with v small populations (will ruin comparison)
For missing data in mortality: calculate rate and fill in [only necessry if valid after above]
Opioid calculation: MME x calc_base_weight

Limitations:
only used big counties
"""
