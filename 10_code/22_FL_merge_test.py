"""
    This script merges the Florida parquet file with the raw mortality data and population data.
    If this works, we can automate it in the same way for all other states.
"""

import pandas as pd

# Loading Florida parquet file from 00_data/states
florida_data = pd.read_parquet("../00_data/states/FL.parquet")
florida_data.rename(columns={"BUYER_COUNTY": "County"}, inplace=True)
# florida_data["County"] = florida_data["County"].str.lower()
florida_data = florida_data[florida_data["County"].notnull()]
florida_data = florida_data[florida_data["YEAR"].between(2002, 2018)]
unique_counties_per_year = florida_data.groupby("YEAR")["County"].nunique()

# Loading raw mortality data from 00_data
mortality_data = pd.read_csv("../00_data/mortality_final.csv")

# Loading population data from 00_data
population_data = pd.read_csv("../00_data/PopFips.csv")
population_data.rename(columns={"CTYNAME": "County", "Year": "YEAR"}, inplace=True)
population_data = population_data[population_data["STATE"] == "FL"]
population_data["County"] = population_data["County"].str.upper()
population_data["County"] = population_data["County"].str.replace(" COUNTY", "")
population_data["County"] = population_data["County"].replace(
    {"ST. JOHNS": "SAINT JOHNS", "ST. LUCIE": "SAINT LUCIE", "DESOTO": "DE SOTO"}
)


# Subset population data to just FL
florida_pop_data = population_data[population_data["STATE"] == "FL"]
florida_mortality_data = mortality_data[mortality_data["State"] == "FL"]
florida_mortality_data["County"] = florida_mortality_data["County"].str.upper()
florida_mortality_data["County"] = florida_mortality_data["County"].str.replace(
    " COUNTY", ""
)
florida_mortality_data.rename(columns={"Year": "YEAR"}, inplace=True)
florida_mortality_data["County"] = florida_mortality_data["County"].replace(
    {"ST. JOHNS": "SAINT JOHNS", "ST. LUCIE": "SAINT LUCIE"}
)
observations_per_county_year = florida_mortality_data.groupby(["County", "YEAR"]).size()
florida_mortality_data = florida_mortality_data[
    florida_mortality_data["YEAR"].between(2002, 2018)
]
observations_per_county_year = florida_mortality_data.groupby(["County", "YEAR"]).size()


# Merge population data with opioid data
pop_opioid_fl = florida_data.merge(
    population_data, how="left", on=["County", "YEAR"], indicator=True
)

assert (pop_opioid_fl["_merge"] == "both").all(), "Not all merges were successful"

pop_opioid_fl.to_parquet("../00_data/FL_Opioid_Pop.parquet")

# Remove the _merge column
pop_opioid_fl = pop_opioid_fl.drop(columns=["_merge"])


# Merge mortality data with population data
pop_mort_fl = florida_mortality_data.merge(
    population_data, how="left", on=["County", "YEAR"], indicator=True
)

assert (pop_mort_fl["_merge"] == "both").all(), "Not all merges were successful"

pop_mort_fl.to_parquet("../00_data/FL_Mortality_Pop.parquet")

# Merge opioid and population data with mortality data
# This one has alot of missing data
florida_mortality_data_op = florida_mortality_data[
    florida_mortality_data["YEAR"].between(2006, 2018)
]
pop_opioid_mort_fl = pop_opioid_fl.merge(
    florida_mortality_data_op, how="outer", on=["County", "YEAR"], indicator=True
)

assert not (
    pop_opioid_mort_fl["_merge"] == "right_only"
).all(), "All left and both merges were successful"

pop_opioid_mort_fl.to_parquet("../00_data/FL_Opioid_Mortality_Pop.parquet")

"""
Filtering based on country population because we don't want counties with v small populations (will ruin comparison)
For missing data in mortality: calculate rate and fill in [only necessry if valid after above]
Opioid calculation: MME x calc_base_weight

Limitations:
only used big counties
"""
# Exploring the intersections and differences of the dataframes for merges

# Get unique County values from both dataframes
florida_counties = set(florida_data["County"].unique())
mortality_counties = set(florida_mortality_data["County"].unique())

# Find shared County values
shared_counties = florida_counties.intersection(mortality_counties)

# Find County values that are in florida_data but not in mortality_data
florida_diff = florida_counties.difference(mortality_counties)

# Find County values that are in mortality_data but not in florida_data
mortality_diff = mortality_counties.difference(florida_counties)

print("Shared Counties:", shared_counties)
print("Counties in Florida data but not in Mortality data:", florida_diff)
print("Counties in Mortality data but not in Florida data:", mortality_diff)


# Get unique County values from both dataframes
population_counties = set(population_data["County"].unique())
florida_counties = set(florida_data["County"].unique())

# Find shared County values
shared_counties = florida_counties.intersection(population_counties)

# Find County values that are in florida_data but not in population_data
florida_pop_diff = florida_counties.difference(population_counties)

# Find County values that are in population_data but not in florida_data
population_florida_diff = population_counties.difference(florida_counties)

print("Shared Counties:", shared_counties)
print("Counties in Florida data but not in Population data:", florida_pop_diff)
print("Counties in Population data but not in Florida data:", population_florida_diff)
