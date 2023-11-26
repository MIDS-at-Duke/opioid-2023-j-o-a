"""find the possible control states for each treatment state"""
import pyarrow.parquet as pq
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


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
# Specify the path to the population csv
population_path = "../00_data/PopFips.csv"


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


def example_pre_post_fl():
    ## change this later
    parquet_file = os.path.join(folder_path + "FL.parquet")
    # Load the Parquet file
    table = pq.read_table(parquet_file)
    # Convert the table to a pandas DataFrame
    df = table.to_pandas()
    # drop the random None county not sure why its there
    df = df[df["YEAR"] != 2019]
    df = df.dropna(subset=["BUYER_COUNTY"])
    print(df["BUYER_COUNTY"].unique())
    print(len(df["BUYER_COUNTY"].unique()))
    df["BUYER_COUNTY"] = df["BUYER_COUNTY"].str.lower()
    df = (
        df.groupby(["BUYER_COUNTY", "BUYER_STATE", "YEAR"])[
            ["MME", "CALC_BASE_WT_IN_GM"]
        ]
        .sum()
        .reset_index()
    )

    # print(df.head())
    # Load the population data
    pop = pd.read_csv(population_path)
    pop["CTYNAME"] = pop["CTYNAME"].str.lower().str.replace(" county", "")
    # print(pop.head())
    pop_subset = pop[
        (pop["STATE"] == "FL") & ((pop["Year"] > 2005) & (pop["Year"] < 2020))
    ]
    print(pop_subset["Year"].unique())
    rename_mapping = {
        "desoto": "de soto",
        "st. lucie": "saint lucie",
        "st. johns": "saint johns",
    }
    pop_subset["CTYNAME"] = pop_subset["CTYNAME"].replace(rename_mapping)
    print(pop_subset["CTYNAME"].unique())
    print(df["BUYER_COUNTY"].unique())
    print(pop_subset.shape)
    merged_df = pd.merge(
        df,
        pop_subset,
        left_on=["BUYER_COUNTY", "YEAR"],
        right_on=["CTYNAME", "Year"],
        how="left",
        indicator=True,
    )
    print(merged_df.head())
    successful_merge = merged_df["_merge"] == "both"

    # Subset the rows where the merge was successful
    successful_rows = merged_df[successful_merge]
    print("Successful:", len(successful_rows))

    unsuccessful_merge = merged_df["_merge"] != "both"

    # Subset the rows where the merge was unsuccessful
    unsuccessful_rows = merged_df[unsuccessful_merge]
    print(unsuccessful_rows["BUYER_COUNTY"].unique())
    print("Unsuccessful:", len(unsuccessful_rows))

    # Print the successful rows
    print(successful_rows)
    print(df.shape)
    # merged_df["MME_Conversion_Factor"] = pd.to_numeric(
    #     merged_df["MME_Conversion_Factor"], errors="coerce"
    # )
    # merged_df["CALC_BASE_WT_IN_GM"] = pd.to_numeric(
    #     merged_df["CALC_BASE_WT_IN_GM"], errors="coerce"
    # )
    print(merged_df.shape)
    # Make an estimate of total morphine equivalent shipments
    # merged_df["morphine_equivalent_g"] = (
    #     merged_df["CALC_BASE_WT_IN_GM"]
    #     * merged_df["MME_Conversion_Factor"]
    #     * merged_df["DOSAGE_UNIT"]
    #     * 1000
    # )
    print("here")
    # print(min(merged_df["MME_Conversion_Factor"].tolist()))
    # print(merged_df["MME_Conversion_Factor"].head())
    # print(merged_df["MME"].head())
    # print(merged_df["Population"].head())

    merged_df["Opioid per capita"] = merged_df["MME"] / merged_df["Population"]
    print(merged_df["BUYER_COUNTY"].unique())
    print(len(merged_df["BUYER_COUNTY"].unique()))
    print(df["BUYER_COUNTY"].unique())
    print(len(df["BUYER_COUNTY"].unique()))
    print(pop_subset["CTYNAME"].unique())
    print(len(pop_subset["CTYNAME"].unique()))
    print(merged_df["Opioid per capita"].head())

    # # Get all columns of the DataFrame
    all_columns = merged_df.columns.tolist()

    # # Print the list of columns
    print(all_columns)

    florida_pre = merged_df[(merged_df["YEAR"] < 2010) & (merged_df["YEAR"] > 2006)]
    florida_post = merged_df[(merged_df["YEAR"] >= 2010) & (merged_df["YEAR"] < 2013)]
    print(florida_pre.head())

    # Create a design matrix
    X = sm.add_constant(florida_pre["Year"])

    # # Fit OLS model
    model = sm.OLS(florida_pre["Opioid per capita"], X)
    results = model.fit()
    # # Get predictions
    model_predict = results.get_prediction(X)
    # this should be how you get the control states
    print("The slope is: ")
    print(results.params["Year"])
    # # Print mean predictions
    # print(model_predict.summary_frame()["mean"])
    # print(model_predict.conf_int(alpha=0.05))
    mean_predictions = model_predict.summary_frame()["mean"]
    ci = model_predict.conf_int(alpha=0.05)
    print(ci)
    plt.plot(
        florida_pre["YEAR"],
        mean_predictions,
        color="blue",
        label="Mean Predictions (Pre-2010)",
    )
    plt.fill_between(
        florida_pre["YEAR"],
        ci[:, 0],
        ci[:, 1],
        color="blue",
        alpha=0.2,
    )

    # Fit OLS model for florida_post
    X_post = sm.add_constant(florida_post["YEAR"])
    model_post = sm.OLS(florida_post["Opioid per capita"], X_post)
    results_post = model_post.fit()

    # Get predictions for florida_post
    model_predict_post = results_post.get_prediction(X_post)
    mean_predictions_post = model_predict_post.summary_frame()["mean"]
    ci = model_predict_post.conf_int(alpha=0.05)

    # Plot the mean predictions for florida_post
    plt.plot(
        florida_post["YEAR"],
        mean_predictions_post,
        color="green",
    )
    plt.fill_between(
        florida_post["YEAR"],
        ci[:, 0],
        ci[:, 1],
        color="blue",
        alpha=0.2,
    )
    plt.axvline(x=2010, color="red", linestyle="--", label="Year 2010")

    plt.xlabel("Year")
    plt.ylabel("Opioid per capita")
    plt.savefig("../30_results/pre_post_test.png")


def find_control_states(state, pre_defined_states):
    ## change this later
    parquet_file = os.path.join(folder_path + "FL.parquet")
    # Load the Parquet file
    table = pq.read_table(parquet_file)
    # Convert the table to a pandas DataFrame
    df = table.to_pandas()
    # drop the random None county not sure why its there
    df = df[df["YEAR"] != 2019]
    df = df.dropna(subset=["BUYER_COUNTY"])
    print(df["BUYER_COUNTY"].unique())
    print(len(df["BUYER_COUNTY"].unique()))
    df["BUYER_COUNTY"] = df["BUYER_COUNTY"].str.lower()
    df = (
        df.groupby(["BUYER_COUNTY", "BUYER_STATE", "YEAR"])[
            ["MME", "CALC_BASE_WT_IN_GM"]
        ]
        .sum()
        .reset_index()
    )

    # print(df.head())
    # Load the population data
    pop = pd.read_csv(population_path)
    pop["CTYNAME"] = pop["CTYNAME"].str.lower().str.replace(" county", "")
    # print(pop.head())
    pop_subset = pop[
        (pop["STATE"] == "FL") & ((pop["Year"] > 2005) & (pop["Year"] < 2020))
    ]
    print(pop_subset["Year"].unique())
    rename_mapping = {
        "desoto": "de soto",
        "st. lucie": "saint lucie",
        "st. johns": "saint johns",
    }
    pop_subset["CTYNAME"] = pop_subset["CTYNAME"].replace(rename_mapping)
    print(pop_subset["CTYNAME"].unique())
    print(df["BUYER_COUNTY"].unique())
    print(pop_subset.shape)
    merged_df = pd.merge(
        df,
        pop_subset,
        left_on=["BUYER_COUNTY", "YEAR"],
        right_on=["CTYNAME", "Year"],
        how="left",
        indicator=True,
    )
    print(merged_df.head())
    successful_merge = merged_df["_merge"] == "both"

    # Subset the rows where the merge was successful
    successful_rows = merged_df[successful_merge]
    print("Successful:", len(successful_rows))

    unsuccessful_merge = merged_df["_merge"] != "both"

    # Subset the rows where the merge was unsuccessful
    unsuccessful_rows = merged_df[unsuccessful_merge]
    print(unsuccessful_rows["BUYER_COUNTY"].unique())
    print("Unsuccessful:", len(unsuccessful_rows))

    # Print the successful rows
    print(successful_rows)
    print(df.shape)
    # merged_df["MME_Conversion_Factor"] = pd.to_numeric(
    #     merged_df["MME_Conversion_Factor"], errors="coerce"
    # )
    # merged_df["CALC_BASE_WT_IN_GM"] = pd.to_numeric(
    #     merged_df["CALC_BASE_WT_IN_GM"], errors="coerce"
    # )
    print(merged_df.shape)
    # Make an estimate of total morphine equivalent shipments
    # merged_df["morphine_equivalent_g"] = (
    #     merged_df["CALC_BASE_WT_IN_GM"]
    #     * merged_df["MME_Conversion_Factor"]
    #     * merged_df["DOSAGE_UNIT"]
    #     * 1000
    # )
    print("here")
    # print(min(merged_df["MME_Conversion_Factor"].tolist()))
    # print(merged_df["MME_Conversion_Factor"].head())
    # print(merged_df["MME"].head())
    # print(merged_df["Population"].head())

    merged_df["Opioid per capita"] = merged_df["MME"] / merged_df["Population"]
    print(merged_df["BUYER_COUNTY"].unique())
    print(len(merged_df["BUYER_COUNTY"].unique()))
    print(df["BUYER_COUNTY"].unique())
    print(len(df["BUYER_COUNTY"].unique()))
    print(pop_subset["CTYNAME"].unique())
    print(len(pop_subset["CTYNAME"].unique()))
    print(merged_df["Opioid per capita"].head())

    # # Get all columns of the DataFrame
    all_columns = merged_df.columns.tolist()

    # # Print the list of columns
    print(all_columns)

    florida_pre = merged_df[(merged_df["YEAR"] < 2010) & (merged_df["YEAR"] > 2006)]
    florida_post = merged_df[(merged_df["YEAR"] >= 2010) & (merged_df["YEAR"] < 2013)]
    print(florida_pre.head())

    # Create a design matrix
    X = sm.add_constant(florida_pre["Year"])

    # # Fit OLS model
    model = sm.OLS(florida_pre["Opioid per capita"], X)
    results = model.fit()
    # # Get predictions
    model_predict = results.get_prediction(X)
    # this should be how you get the control states
    print("The slope is: ")
    print(results.params["Year"])
    slope = results.params["Year"]
    potential_control = {}
    for i in pre_defined_states:
        print(i)
        parquet_file = os.path.join(folder_path + i + ".parquet")
        table = pq.read_table(parquet_file)
        # Convert the table to a pandas DataFrame
        df = table.to_pandas()
        df = df.dropna(subset=["BUYER_COUNTY"])
        df = df[df["YEAR"] != 2019]

        print(df["BUYER_COUNTY"].unique())
        print(len(df["BUYER_COUNTY"].unique()))
        df["BUYER_COUNTY"] = df["BUYER_COUNTY"].str.lower()
        df = (
            df.groupby(["BUYER_COUNTY", "BUYER_STATE", "YEAR"])[
                ["MME", "CALC_BASE_WT_IN_GM"]
            ]
            .sum()
            .reset_index()
        )
        pop = pd.read_csv(population_path)
        pop["CTYNAME"] = pop["CTYNAME"].str.lower().str.replace(" county", "")
        # print(pop.head())
        pop_subset = pop[
            (pop["STATE"] == i) & ((pop["Year"] > 2005) & (pop["Year"] < 2020))
        ]
        print(pop_subset.head())
        merged_df = pd.merge(
            df,
            pop_subset,
            left_on=["BUYER_COUNTY", "YEAR"],
            right_on=["CTYNAME", "Year"],
            how="left",
            indicator=True,
        )
        successful_merge = merged_df["_merge"] == "both"

        # Subset the rows where the merge was successful
        successful_rows = merged_df[successful_merge]
        print("Successful:", len(successful_rows))

        unsuccessful_merge = merged_df["_merge"] != "both"

        # Subset the rows where the merge was unsuccessful
        unsuccessful_rows = merged_df[unsuccessful_merge]
        print("Unsuccessful:", len(unsuccessful_rows))
        print(unsuccessful_rows.head())
        if len(successful_rows) < 5:
            continue

        # change this later
        merged_df["Opioid per capita"] = merged_df["MME"] / merged_df["Population"]

        state_pre = merged_df[(merged_df["YEAR"] < 2010) & (merged_df["YEAR"] > 2006)]
        print(state_pre.head())
        # Create a design matrix
        # change this later, this shouldn't happen if we merge propely
        state_pre = state_pre[
            pd.notnull(state_pre["Opioid per capita"]) & pd.notnull(state_pre["Year"])
        ].reset_index()

        X = sm.add_constant(state_pre["Year"])

        # # Fit OLS model
        model = sm.OLS(state_pre["Opioid per capita"], X)
        results = model.fit()
        # # Get predictions
        model_predict = results.get_prediction(X)
        # this should be how you get the control states
        print("The slope is: ")
        print(results.params["Year"])
        slope_state = results.params["Year"]
        # Print the correlation
        print(slope, slope_state)
        # arbritrary number
        if abs(slope - slope_state) < 15:
            potential_control[i] = [slope_state]
    return potential_control


if __name__ == "__main__":
    # find_control_states_prep(PRE_DEFINED_STATES)
    # example_pre_post_fl()
    controls = find_control_states("FL", PRE_DEFINED_STATES)
    print(controls)
