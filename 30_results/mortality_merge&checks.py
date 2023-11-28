import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import statsmodels.api as sm
import numpy as np


def process_mortality_data(csv_path, state_codes=None):
    mortality_data = pd.read_csv(csv_path)
    if state_codes is None:
        mortality_data = mortality_data
    else:
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


def process_population_data(csv_path, state_codes=None):
    population_data = pd.read_csv(csv_path)
    population_data.rename(columns={"CTYNAME": "County", "Year": "YEAR"}, inplace=True)
    population_data["County"] = population_data["County"].str.upper()
    population_data["County"] = population_data["County"].str.replace(" COUNTY", "")
    if state_codes is None:
        state_pop_data = population_data
    else:
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
    # pop_mortality_data["Deaths_Revised"] = (
    #     pop_mortality_data["Death_Rate"] * pop_mortality_data["Population"]
    # )
    return pop_mortality_data


def format_percent(x, pos):
    return f"{x * 100:.3f}%"


def plot_pre_post_ols(df, state_code, policy_year):
    years_before = policy_year - 2003
    years_after = 2015 - policy_year

    limit = min(
        years_before, years_after
    )  # Determine the limit based on the lesser value

    pre_policy = df[(df["YEAR"] >= policy_year - limit) & (df["YEAR"] < policy_year)]
    post_policy = df[(df["YEAR"] >= policy_year) & (df["YEAR"] <= policy_year + limit)]

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
        color="orange",
        label=(f"Post-{policy_year} OLS"),
    )
    plt.fill_between(
        post_policy["YEAR"],
        pred_post_ci[:, 0],
        pred_post_ci[:, 1],
        color="orange",
        alpha=0.3,
    )
    plt.axvline(
        x=policy_year, color="black", linestyle="--", label=(f"Year {policy_year}")
    )

    plt.xlabel("Year")
    plt.ylabel("Percentage (%) of Population that died by opioids")
    plt.title(
        f"Percentage (%) of Population that died by opioids in Washington (Pre and Post {policy_year})"
    )
    formatter = mtick.FuncFormatter(format_percent)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.legend()
    plt.tight_layout()
    plt.show()
    pre_policy_mean = pre_policy["Death_Rate"].mean()
    post_policy_mean = post_policy["Death_Rate"].mean()
    print(f"Pre-{policy_year} mean: {pre_policy_mean}")
    print(f"Post-{policy_year} mean: {post_policy_mean}")


def plot_diff_in_diff(treatment_df, control_df, policy_year):
    years_before = policy_year - 2003
    years_after = 2015 - policy_year
    limit = min(
        years_before, years_after
    )  # Determine the limit based on the lesser value
    treatment_pre_policy = treatment_df[
        (treatment_df["YEAR"] >= policy_year - limit)
        & (treatment_df["YEAR"] < policy_year)
    ]
    treatment_post_policy = treatment_df[
        (treatment_df["YEAR"] >= policy_year)
        & (treatment_df["YEAR"] <= policy_year + limit)
    ]
    control_pre_policy = control_df[
        (control_df["YEAR"] >= policy_year - limit) & (control_df["YEAR"] < policy_year)
    ]
    control_post_policy = control_df[
        (control_df["YEAR"] >= policy_year)
        & (control_df["YEAR"] <= policy_year + limit)
    ]

    X_treatment_pre = sm.add_constant(treatment_pre_policy["YEAR"])
    y_treatment_pre = treatment_pre_policy["Death_Rate"]
    model_treatment_pre = sm.OLS(y_treatment_pre, X_treatment_pre)
    results_treatment_pre = model_treatment_pre.fit()

    X_treatment_post = sm.add_constant(treatment_post_policy["YEAR"])
    y_treatment_post = treatment_post_policy["Death_Rate"]
    model_treatment_post = sm.OLS(y_treatment_post, X_treatment_post)
    results_treatment_post = model_treatment_post.fit()

    X_control_pre = sm.add_constant(control_pre_policy["YEAR"])
    y_control_pre = control_pre_policy["Death_Rate"]
    model_control_pre = sm.OLS(y_control_pre, X_control_pre)
    results_control_pre = model_control_pre.fit()

    X_control_post = sm.add_constant(control_post_policy["YEAR"])
    y_control_post = control_post_policy["Death_Rate"]
    model_control_post = sm.OLS(y_control_post, X_control_post)
    results_control_post = model_control_post.fit()

    plt.figure(figsize=(8, 6))
    # Confidence interval bands for Pre-2010
    pred_treatment_pre = results_treatment_pre.get_prediction(X_treatment_pre)
    pred_treatment_pre_ci = pred_treatment_pre.conf_int()
    plt.plot(
        treatment_pre_policy["YEAR"],
        results_treatment_pre.predict(X_treatment_pre),
        color="blue",
        label=(f"Treatment State: WA"),
    )
    plt.fill_between(
        treatment_pre_policy["YEAR"],
        pred_treatment_pre_ci[:, 0],
        pred_treatment_pre_ci[:, 1],
        color="blue",
        alpha=0.3,
    )

    pred_control_pre = results_control_pre.get_prediction(X_control_pre)
    pred_control_pre_ci = pred_control_pre.conf_int()
    plt.plot(
        control_pre_policy["YEAR"],
        results_control_pre.predict(X_control_pre),
        color="orange",
        label=(f"Control States: OH, MI, ME, HI"),
    )
    plt.fill_between(
        control_pre_policy["YEAR"],
        pred_control_pre_ci[:, 0],
        pred_control_pre_ci[:, 1],
        color="orange",
        alpha=0.3,
    )

    # Confidence interval bands for Post-2010
    pred_treatment_post = results_treatment_post.get_prediction(X_treatment_post)
    pred_treatment_post_ci = pred_treatment_post.conf_int()
    plt.plot(
        treatment_post_policy["YEAR"],
        results_treatment_post.predict(X_treatment_post),
        color="blue",
    )
    plt.fill_between(
        treatment_post_policy["YEAR"],
        pred_treatment_post_ci[:, 0],
        pred_treatment_post_ci[:, 1],
        color="blue",
        alpha=0.3,
    )

    pred_control_post = results_control_post.get_prediction(X_control_post)
    pred_control_post_ci = pred_control_post.conf_int()
    plt.plot(
        control_post_policy["YEAR"],
        results_control_post.predict(X_control_post),
        color="orange",
    )
    plt.fill_between(
        control_post_policy["YEAR"],
        pred_control_post_ci[:, 0],
        pred_control_post_ci[:, 1],
        color="orange",
        alpha=0.3,
    )
    plt.axvline(
        x=policy_year, color="black", linestyle="--", label=(f"Year {policy_year}")
    )

    plt.xlabel("Year")
    plt.ylabel("Percentage (%) of Population that died by opioids")
    plt.title(
        f"Percentage (%) of Population that died by opioids (Pre and Post {policy_year})"
    )
    formatter = mtick.FuncFormatter(format_percent)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.legend()
    plt.tight_layout()
    plt.show()
    treatment_pre_change = treatment_pre_policy["Death_Rate"].mean()
    treatment_post_change = treatment_post_policy["Death_Rate"].mean()
    control_pre_change = control_pre_policy["Death_Rate"].mean()
    control_post_change = control_post_policy["Death_Rate"].mean()
    did_treatment = (
        (treatment_post_change - treatment_pre_change) / treatment_pre_change
    ) * 100
    did_control = (
        (control_post_change - control_pre_change) / control_pre_change
    ) * 100
    # Calculate the Difference in Differences
    diff_in_diff = did_treatment - did_control

    # Calculate standard errors or conduct statistical tests if needed
    # You can use statistical methods like t-tests or regression models for significance testing

    # Print or use summary statistics as needed
    print(f"Change in Death Rate for Treatment Group (Pre): {treatment_pre_change}")
    print(f"Change in Death Rate for Treatment Group (Post): {treatment_post_change}")
    print(f"Change in Death Rate for Control Group (Pre): {control_pre_change}")
    print(f"Change in Death Rate for Control Group (Post): {control_post_change}")
    print(f"Difference in Differences: {diff_in_diff}")


# For Washington
wa_mort, o = process_mortality_data("../00_data/mortality_final.csv", "FL")
wa_pop = process_population_data("../00_data/PopFips.csv", "FL")
clean_county(wa_mort)
clean_county(wa_pop)
wa_pop_mort = merge_mortality_population(wa_pop, wa_mort, 2003, 2015)
# print(check_unique_counties_per_year(wa_pop_mort, wa_pop))
# some_null_counties, all_null_counties, no_null_counties = get_death_null_counties(
#     wa_pop_mort
# )
# find_mismatched_counties(wa_pop_mort)
imputed_wa_pop_mort = impute_for_mortality(wa_pop_mort)

# print(wa_pop_mort["Deaths"].isnull().sum())
# print(imputed_wa_pop_mort["Death_Rate"].isnull().sum())

wa_controls_mort, o = process_mortality_data(
    "../00_data/mortality_final.csv", ["OH", "MI", "ME", "HI"]
)
wa_controls_pop = process_population_data(
    "../00_data/PopFips.csv", ["OH", "MI", "ME", "HI"]
)
clean_county(wa_controls_mort)
clean_county(wa_controls_pop)
wa_controls_pop_mort = merge_mortality_population(
    wa_controls_pop, wa_controls_mort, 2003, 2015
)
imputed_wa_controls = impute_for_mortality(wa_controls_pop_mort)
print(wa_controls_pop_mort["Deaths"].isnull().sum())
print(imputed_wa_controls["Death_Rate"].isnull().sum())

plot_pre_post_ols(imputed_wa_pop_mort, "FL", 2010)
# plot_diff_in_diff(imputed_wa_pop_mort, imputed_wa_controls, 2012)


def calculate_slope(df):
    X = sm.add_constant(df["YEAR"])
    y = df["Death_Rate"]
    model = sm.OLS(y, X)
    results = model.fit()
    return results.params[1]  # Return the slope coefficient


# Assuming your mortality dataset is stored in mortality_data variable
# 'treatment_state_code' is the code of your treatment state

# mort, o = process_mortality_data("../00_data/mortality_final.csv")
# pop = process_population_data("../00_data/PopFips.csv")
# clean_county(mort)
# clean_county(pop)
# mortality_data_1 = merge_mortality_population(pop, mort, 2003, 2015)
# mortality_data = impute_for_mortality(mortality_data_1)
# treatment_state_data = mortality_data[mortality_data["STATE"] == "WA"]

# # Calculate slope for treatment state
# slope_treatment_state = calculate_slope(treatment_state_data)

# similar_states = []
# for state_code, state_df in mortality_data.groupby("STATE"):
#     if state_code != "WA":  # Skip the treatment state
#         pre_policy_data = state_df[state_df["YEAR"] < 2012]
#         if len(pre_policy_data) > 1:  # At least two years needed for slope calculation
#             slope = calculate_slope(pre_policy_data)
#             similar_states.append((state_code, slope))

# # Sort the states based on the absolute difference in slope
# similar_states.sort(key=lambda x: abs(x[1] - slope_treatment_state))

# # Select the states with slopes most similar to the treatment state
# num_states_to_select = 10  # Change this number as needed
# selected_states = similar_states[:num_states_to_select]

# print("States with slopes most similar to WA:")
# for state_code, slope in selected_states:
#     print(
#         f"State Code: {state_code}, Slope Difference: {slope - slope_treatment_state}"
#     )
