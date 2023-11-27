import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np


def process_state_consumption(parquet_path):
    consumption_data = pd.read_parquet(parquet_path)
    consumption_data.rename(columns={"BUYER_COUNTY": "County"}, inplace=True)
    consumption_data = consumption_data[consumption_data["County"].notnull()]
    unique_counties_per_year = consumption_data.groupby("YEAR")["County"].nunique()
    return consumption_data, unique_counties_per_year


def process_mortality_data(csv_path, state_codes):
    mortality_data = pd.read_csv(csv_path)
    if isinstance(state_codes, list):
        mortality_data = mortality_data[mortality_data["State"].isin(state_codes)]
    else:
        mortality_data = mortality_data[mortality_data["State"] == state_codes]

    mortality_data["County"] = mortality_data["County"].str.upper()
    mortality_data["County"] = mortality_data["County"].str.replace(" COUNTY", "")
    mortality_data.rename(columns={"Year": "YEAR"}, inplace=True)
    mortality_data["Deaths"] = pd.to_numeric(mortality_data["Deaths"], errors="coerce")

    # Assuming 'Deaths' represent total drug-related deaths, calculate 75% of them as opioid-related
    mortality_data["Deaths"] = mortality_data["Deaths"] * 0.75
    observations_per_county_year = mortality_data.groupby(["County", "YEAR"]).size()
    return mortality_data, observations_per_county_year


def process_population_data(csv_path, state_codes):
    population_data = pd.read_csv(csv_path)
    population_data.rename(columns={"CTYNAME": "County", "Year": "YEAR"}, inplace=True)
    population_data["County"] = population_data["County"].str.upper()
    population_data["County"] = population_data["County"].str.replace(" COUNTY", "")

    if isinstance(state_codes, list):
        state_pop_data = population_data[population_data["STATE"].isin(state_codes)]
    else:
        state_pop_data = population_data[population_data["STATE"] == state_codes]

    # Calculate mean and median population per county
    mean_population = state_pop_data.groupby("County")["Population"].mean()
    median_population = state_pop_data.groupby("County")["Population"].median()

    # Calculate proportion of counties above 65,000
    prop_above_threshold = (
        mean_population > 65000
    ).mean()  # Proportion of counties above 65,000

    threshold = 65000 if prop_above_threshold > 0.5 else median_population.median()

    # Filter population data based on the determined threshold
    filtered_pop_data = state_pop_data[
        state_pop_data.groupby("County")["Population"].transform("mean") > threshold
    ]
    return filtered_pop_data


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


def merge_mortality_population(population_data, mortality_data, start_year, end_year):
    # Merge mortality data with population data
    merged_data = population_data.merge(
        mortality_data, how="left", on=["County", "YEAR"], indicator=True
    )
    # Filtering for specific years related to mortality data
    filtered_data = merged_data[merged_data["YEAR"].between(start_year, end_year)]
    return filtered_data


def find_mismatched_counties(merged_data):
    # Filter for rows that exist only in the left dataframe (population_data)
    left_only = merged_data.loc[merged_data["_merge"] == "left_only", "County"].unique()

    # Filter for rows that exist only in the right dataframe (mortality_data)
    right_only = merged_data.loc[
        merged_data["_merge"] == "right_only", "County"
    ].unique()
    print(f"Counties only in the pop data set: {left_only}")
    print(f"Counties only in the mortality dataset: {right_only}")


def check_unique_counties_per_year(merged_data, population_data):
    unique_counties_per_year = merged_data.groupby("YEAR")["County"].nunique()
    if unique_counties_per_year.nunique() == 1 and unique_counties_per_year.unique()[
        0
    ] == len(population_data["County"].unique()):
        return "All years have all unique counties."
    else:
        years_with_missing_counties = unique_counties_per_year[
            unique_counties_per_year != len(population_data["County"].unique())
        ].index.tolist()
        missing_counties_per_year = {}
        for year in years_with_missing_counties:
            missing_counties = (
                population_data[
                    ~population_data["County"].isin(
                        merged_data[merged_data["YEAR"] == year]["County"]
                    )
                ]["County"]
                .unique()
                .tolist()
            )
            missing_counties_per_year[year] = missing_counties
        return missing_counties_per_year


def get_death_null_counties(pop_mort_wa):
    nan_counts = pop_mort_wa.groupby("County")["Deaths"].apply(
        lambda x: x.isnull().sum()
    )
    no_null_counties = nan_counts[nan_counts == 0].index.tolist()
    some_null_counties = nan_counts[
        (nan_counts > 0) & (nan_counts < len(pop_mort_wa["YEAR"].unique()))
    ].index.tolist()
    all_null_counties = nan_counts[
        nan_counts == len(pop_mort_wa["YEAR"].unique())
    ].index.tolist()
    return some_null_counties, all_null_counties, no_null_counties


# To check the populations for the lists retrieved from the previous function
def print_county_population(county_list, population_data):
    for county in county_list:
        county_population = population_data.loc[
            population_data["County"] == county, "Population"
        ].values[0]
        print(f"Population of {county}: {county_population}")


def impute_for_mortality(pop_mortality_data):
    # Calculate the death rate (i.e., Deaths/Population) for each county-year with data
    pop_mortality_data["Death_Rate"] = (
        pop_mortality_data["Deaths"] / pop_mortality_data["Population"]
    )
    # Calculate mean Death Rate for each county
    county_means = pop_mortality_data.groupby("County")["Death_Rate"].mean()
    # Fill missing Death Rate values using each county's mean
    for county, mean_rate in county_means.items():
        mask = (pop_mortality_data["County"] == county) & (
            pop_mortality_data["Death_Rate"].isnull()
        )
        pop_mortality_data.loc[mask, "Death_Rate"] = mean_rate
    # Drop all counties that have ALL null values
    all_null_counties = pop_mortality_data.groupby("County")["Death_Rate"].apply(
        lambda x: x.isnull().all()
    )
    dropped_counties = all_null_counties[all_null_counties].index.tolist()
    print(f"Counties dropped due to all null Death Rates: {dropped_counties}")
    pop_mortality_data = pop_mortality_data[
        ~pop_mortality_data["County"].isin(dropped_counties)
    ]
    return pop_mortality_data


def plot_pre_post_ols(df, state_code, policy_year):
    pre_policy = df[df["YEAR"] < (policy_year - 1)]
    post_policy = df[df["YEAR"] > (policy_year - 1)]

    X_pre = sm.add_constant(pre_policy["YEAR"])
    y_pre = pre_policy["Death_Rate"]
    model_pre = sm.OLS(y_pre, X_pre)
    results_pre = model_pre.fit()

    X_post = sm.add_constant(post_policy["YEAR"])
    y_post = post_policy["Death_Rate"]
    model_post = sm.OLS(y_post, X_post)
    results_post = model_post.fit()

    plt.figure(figsize=(8, 6))
    # Confidence interval bands for Pre-2010
    pred_pre = results_pre.get_prediction(X_pre)
    pred_pre_ci = pred_pre.conf_int()
    plt.plot(
        pre_policy["YEAR"],
        results_pre.predict(X_pre),
        color="blue",
        label=(f"Pre-{policy_year} OLS"),
    )
    plt.fill_between(
        pre_policy["YEAR"],
        pred_pre_ci[:, 0],
        pred_pre_ci[:, 1],
        color="blue",
        alpha=0.3,
    )

    # Confidence interval bands for Post-2010
    pred_post = results_post.get_prediction(X_post)
    pred_post_ci = pred_post.conf_int()
    plt.plot(
        post_policy["YEAR"],
        results_post.predict(X_post),
        color="green",
        label=(f"Post-{policy_year} OLS"),
    )
    plt.fill_between(
        post_policy["YEAR"],
        pred_post_ci[:, 0],
        pred_post_ci[:, 1],
        color="green",
        alpha=0.3,
    )
    plt.axvline(
        x=policy_year, color="black", linestyle="--", label=(f"Year {policy_year}")
    )

    plt.xlabel("Year")
    plt.ylabel("Death Rate")
    plt.title(f"Death Rate in {state_code} (Pre and Post {policy_year})")
    plt.legend()
    plt.grid(True)
    plt.show()


# For Washington
wa_mort, o = process_mortality_data("../00_data/mortality_final.csv", "FL")
wa_pop = process_population_data("../00_data/PopFips.csv", "FL")
clean_county(wa_mort)
clean_county(wa_pop)
wa_pop_mort = merge_mortality_population(wa_pop, wa_mort, 2003, 2015)
print(check_unique_counties_per_year(wa_pop_mort, wa_pop))
some_null_counties, all_null_counties, no_null_counties = get_death_null_counties(
    wa_pop_mort
)
find_mismatched_counties(wa_pop_mort)
imputed_wa_pop_mort = impute_for_mortality(wa_pop_mort)

print(wa_pop_mort["Deaths"].isnull().sum())
print(imputed_wa_pop_mort["Death_Rate"].isnull().sum())

plot_pre_post_ols(imputed_wa_pop_mort, "WA", 2010)
