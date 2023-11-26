"""
    This script merges the Florida parquet file with the raw mortality data and population data.
    If this works, we can automate it in the same way for all other states.
"""

import pandas as pd
import dask.dataframe as dd


def clean_data(state_parquet, state_mortality, state_population):
    state_data = dd.read_parquet(f"../00_data/states/{state_parquet}.parquet")
    state_data = state_data.rename(columns={"BUYER_COUNTY": "County"})

    mortality_data = dd.read_csv(f"../00_data/{state_mortality}.csv")
    state_mortality_data = mortality_data[
        mortality_data["State"] == state_parquet
    ].copy()
    state_mortality_data["County"] = (
        state_mortality_data["County"].str.upper().str.replace(" COUNTY", "")
    )

    population_data = dd.read_csv(f"../00_data/{state_population}.csv")
    population_data = population_data.rename(columns={"CTYNAME": "County"})
    state_pop_data = population_data[population_data["STATE"] == state_parquet].copy()
    state_pop_data["County"] = (
        state_pop_data["County"].str.upper().str.replace(" COUNTY", "")
    )

    return state_data, state_mortality_data, state_pop_data


# Clean the data using dask dataframes
FL_consume, FL_mortality, FL_pop = clean_data("FL", "mortality_final", "PopFips")

# Merge using dask dataframes
pop_mortality_data = dd.merge(
    FL_consume, FL_mortality, how="left", on=["County"], indicator=True
)
fl_finaldata = dd.merge(pop_mortality_data, FL_pop, how="left", on=["County", "Year"])
fl_finaldata = fl_finaldata.compute()  # Convert back to pandas dataframe


"""
    Filtering based on country population because we don't want counties with v small populations (will ruin comparison)
    For missing data in mortality: calculate rate and fill in [only necessry if valid after above]
    Opioid calculation: MME x calc_base_weight

    Limitations:
    only used big counties
"""
