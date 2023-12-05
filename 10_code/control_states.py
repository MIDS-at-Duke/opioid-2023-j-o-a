"""find the possible control states for each treatment state"""
import pyarrow.parquet as pq
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import operator


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
    #  "DC",
]

# Specify the path to the folder containing Parquet files
folder_path = "../00_data/states/"
# Specify the path to the population csv
population_path = "../00_data/PopFips.csv"
# Specify the path to the opiod_pop csv
opiod_pop_path = "../00_data/opioid_pop_clean.parquet"
opiod_pop_month_path = "../00_data/opioid_pop_months_clean.parquet"


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
    print(results.params)
    # # Print mean predictions
    # print(model_predict.summary_frame()["mean"])
    # print(model_predict.conf_int(alpha=0.05))
    mean_predictions = model_predict.summary_frame()["mean"]
    ci = model_predict.conf_int(alpha=0.05)
    ci_l = model_predict.summary_frame()["mean_ci_lower"]
    ci_u = model_predict.summary_frame()["mean_ci_upper"]
    print("here......")
    print(model_predict.summary_frame())
    print(ci)
    plt.plot(
        florida_pre["YEAR"],
        mean_predictions,
        color="blue",
        label="Mean Predictions (Pre-2010)",
    )
    plt.fill_between(
        florida_pre["YEAR"],
        ci_l,
        ci_u,
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
    # ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
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


def example_pre_post_fl_cleaned():
    table = pq.read_table(opiod_pop_path)

    df = table.to_pandas()
    print(df[df["State"] == "FL"].head())
    df_fl = df[df["State"] == "FL"]
    df_fl["Opioid per capita"] = df_fl["MME"] / df_fl["Population"]
    print(df_fl.head())

    # grab pre and post intervention
    florida_pre = df_fl[(df_fl["Year"] < 2010) & (df_fl["Year"] > 2006)]
    florida_post = df_fl[(df_fl["Year"] >= 2010) & (df_fl["Year"] < 2013)]

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
    print(results.params)
    # # Print mean predictions
    # print(model_predict.summary_frame()["mean"])
    # print(model_predict.conf_int(alpha=0.05))
    mean_predictions = model_predict.summary_frame()["mean"]
    pred_post_ci = model_predict.conf_int()

    ci = model_predict.conf_int(alpha=0.05)
    ci_l = model_predict.summary_frame()["mean_ci_lower"]
    ci_u = model_predict.summary_frame()["mean_ci_upper"]
    print("here......")
    print(model_predict.summary_frame())
    print(ci)
    plt.plot(
        florida_pre["Year"],
        mean_predictions,
        color="blue",
        label="Mean Predictions (Pre-2010)",
    )
    plt.fill_between(
        florida_pre["Year"],
        pred_post_ci[:, 0],
        pred_post_ci[:, 1],
        color="blue",
        alpha=0.2,
    )

    # Fit OLS model for florida_post
    X_post = sm.add_constant(florida_post["Year"])
    model_post = sm.OLS(florida_post["Opioid per capita"], X_post)
    results_post = model_post.fit()

    # Get predictions for florida_post
    model_predict_post = results_post.get_prediction(X_post)
    mean_predictions_post = model_predict_post.summary_frame()["mean"]
    ci = model_predict_post.conf_int(alpha=0.05)
    # ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
    # Plot the mean predictions for florida_post
    plt.plot(
        florida_post["Year"],
        mean_predictions_post,
        color="blue",
    )
    plt.fill_between(
        florida_post["Year"],
        ci[:, 0],
        ci[:, 1],
        color="blue",
        alpha=0.2,
    )
    plt.axvline(x=2010, color="red", linestyle="--", label="Year 2010")
    # "WV", "KY", "ME"
    df = table.to_pandas()

    # Filter data for the specified states
    # states_to_process = ["WV", "KY", "ME"]
    states_to_process = ["DE", "NV", "TN"]
    df_selected_states = df[df["State"].isin(states_to_process)]

    # Calculate Opioid per capita
    df_selected_states["Opioid per capita"] = (
        df_selected_states["MME"] / df_selected_states["Population"]
    )

    # Print the processed data for the selected states
    print(df_selected_states.head())

    # Grab pre and post intervention for the selected states
    selected_states_pre = df_selected_states[
        (df_selected_states["Year"] < 2010) & (df_selected_states["Year"] > 2006)
    ]
    selected_states_post = df_selected_states[
        (df_selected_states["Year"] >= 2010) & (df_selected_states["Year"] < 2013)
    ]

    # Create a design matrix
    X = sm.add_constant(selected_states_pre["Year"])

    # # Fit OLS model
    model = sm.OLS(selected_states_pre["Opioid per capita"], X)
    results = model.fit()
    # # Get predictions
    model_predict = results.get_prediction(X)
    print("The slope is for the three states is: ")
    print(results.params["Year"])
    # # Print mean predictions
    # print(model_predict.summary_frame()["mean"])
    # print(model_predict.conf_int(alpha=0.05))
    mean_predictions = model_predict.summary_frame()["mean"]
    ci = model_predict.conf_int(alpha=0.05)
    ci_l = model_predict.summary_frame()["mean_ci_lower"]
    ci_u = model_predict.summary_frame()["mean_ci_upper"]
    print("here......")
    print(model_predict.summary_frame())
    print(ci)
    plt.plot(
        selected_states_pre["Year"],
        mean_predictions,
        color="green",
        label="Mean Predictions (Pre-2010)",
    )
    plt.fill_between(
        selected_states_pre["Year"],
        ci_l,
        ci_u,
        color="green",
        alpha=0.2,
    )
    X_post = sm.add_constant(selected_states_post["Year"])
    model_post = sm.OLS(selected_states_post["Opioid per capita"], X_post)
    results_post = model_post.fit()

    # Get predictions for florida_post
    model_predict_post = results_post.get_prediction(X_post)
    mean_predictions_post = model_predict_post.summary_frame()["mean"]
    ci = model_predict_post.conf_int(alpha=0.05)
    # ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
    # Plot the mean predictions for florida_post
    plt.plot(
        selected_states_post["Year"],
        mean_predictions_post,
        color="green",
    )
    plt.fill_between(
        selected_states_post["Year"],
        ci[:, 0],
        ci[:, 1],
        color="green",
        alpha=0.2,
    )

    # Print pre and post intervention data for the selected states

    plt.xlabel("Year")
    plt.ylabel("Opioid per capita")
    plt.savefig("../30_results/pre_post_test.png")


def find_control_states_cleaned_fl(state, pre_defined_states):
    table = pq.read_table(opiod_pop_path)

    df = table.to_pandas()
    # print(df[df["State"] == "FL"].head())
    df_fl = df[df["State"] == "FL"]
    df_fl["Opioid per capita"] = df_fl["MME"] / df_fl["Population"]
    # print(df_fl.head())

    # grab pre and post intervention
    florida_pre = df_fl[(df_fl["Year"] < 2010) & (df_fl["Year"] > 2006)]
    X = sm.add_constant(florida_pre["Year"])

    # # Fit OLS model
    model = sm.OLS(florida_pre["Opioid per capita"], X)
    results = model.fit()
    slope = results.params["Year"]
    potential_control = {}
    mean_opiod = {}
    for i in pre_defined_states:
        print(i)
        table = pq.read_table(opiod_pop_path)
        df = table.to_pandas()
        df_state = df[df["State"] == i]
        df_state["Opioid per capita"] = df_state["MME"] / df_state["Population"]
        state_pre = df_state[(df_state["Year"] < 2010) & (df_state["Year"] > 2006)]
        X = sm.add_constant(state_pre["Year"])

        # # Fit OLS model
        model = sm.OLS(state_pre["Opioid per capita"], X)
        results = model.fit()
        model_predict = results.get_prediction(X)
        print(model_predict.summary_frame())
        state_slope = results.params["Year"]
        print(slope, state_slope)
        potential_control[i] = [state_slope]
        mean_opiod[i] = np.mean(state_pre["Opioid per capita"])
    # print(mean_opiod)
    florida_mean = mean_opiod["FL"]

    differences = {
        state: abs(mean - florida_mean) for state, mean in mean_opiod.items()
    }
    # print(differences)
    top_three_states = sorted(differences, key=differences.get)[:11]
    print(top_three_states)

    return dict(sorted(potential_control.items(), key=lambda item: item[1]))


def find_control_states_cleaned_wa(state, pre_defined_states):
    table = pq.read_table(opiod_pop_path)

    df = table.to_pandas()
    # print(df[df["State"] == "FL"].head())
    df_fl = df[df["State"] == "WA"]
    df_fl["Opioid per capita"] = df_fl["MME"] / df_fl["Population"]
    # print(df_fl.head())

    # grab pre and post intervention
    florida_pre = df_fl[(df_fl["Year"] < 2012) & (df_fl["Year"] > 2008)]
    X = sm.add_constant(florida_pre["Year"])

    # # Fit OLS model
    model = sm.OLS(florida_pre["Opioid per capita"], X)
    results = model.fit()
    slope = results.params["Year"]
    potential_control = {}
    mean_opiod = {}
    for i in pre_defined_states:
        print(i)
        table = pq.read_table(opiod_pop_path)
        df = table.to_pandas()
        df_state = df[df["State"] == i]
        df_state["Opioid per capita"] = df_state["MME"] / df_state["Population"]
        state_pre = df_state[(df_state["Year"] < 2012) & (df_state["Year"] > 2008)]
        X = sm.add_constant(state_pre["Year"])

        # # Fit OLS model
        model = sm.OLS(state_pre["Opioid per capita"], X)
        results = model.fit()
        model_predict = results.get_prediction(X)
        print(model_predict.summary_frame())
        state_slope = results.params["Year"]
        print(slope, state_slope)
        potential_control[i] = [state_slope]
        mean_opiod[i] = np.mean(state_pre["Opioid per capita"])

    return dict(sorted(potential_control.items(), key=lambda item: item[1]))


def example_pre_post_wa_cleaned():
    table = pq.read_table(opiod_pop_path)

    df = table.to_pandas()
    df_fl = df[df["State"] == "WA"]
    df_fl["Opioid per capita"] = df_fl["MME"] / df_fl["Population"]
    print(df_fl.head())

    # grab pre and post intervention
    florida_pre = df_fl[(df_fl["Year"] < 2012) & (df_fl["Year"] > 2002)]
    florida_post = df_fl[(df_fl["Year"] >= 2012) & (df_fl["Year"] < 2015)]

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
    print(results.params)
    # # Print mean predictions
    # print(model_predict.summary_frame()["mean"])
    # print(model_predict.conf_int(alpha=0.05))
    mean_predictions = model_predict.summary_frame()["mean"]
    ci = model_predict.conf_int(alpha=0.05)
    ci_l = model_predict.summary_frame()["mean_ci_lower"]
    ci_u = model_predict.summary_frame()["mean_ci_upper"]
    print("here......")
    print(model_predict.summary_frame())
    print(ci)
    plt.plot(
        florida_pre["Year"],
        mean_predictions,
        color="blue",
        label="Mean Predictions (Pre-2010)",
    )
    # plt.fill_between(
    #     florida_pre["Year"],
    #     ci_l,
    #     ci_u,
    #     color="blue",
    #     alpha=0.2,
    # )

    # Fit OLS model for florida_post
    X_post = sm.add_constant(florida_post["Year"])
    model_post = sm.OLS(florida_post["Opioid per capita"], X_post)
    results_post = model_post.fit()

    # Get predictions for florida_post
    model_predict_post = results_post.get_prediction(X_post)
    mean_predictions_post = model_predict_post.summary_frame()["mean"]
    ci = model_predict_post.conf_int(alpha=0.05)
    # ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
    # Plot the mean predictions for florida_post
    plt.plot(
        florida_post["Year"],
        mean_predictions_post,
        color="blue",
    )
    # plt.fill_between(
    #     florida_post["Year"],
    #     ci[:, 0],
    #     ci[:, 1],
    #     color="blue",
    #     alpha=0.2,
    # )
    plt.axvline(x=2012, color="red", linestyle="--", label="Year 2010")
    # "WV", "KY", "ME"
    df = table.to_pandas()

    # Filter data for the specified states
    # states_to_process = ["WV", "KY", "ME"]
    states_to_process = ["HI", "OH", "MA"]
    df_selected_states = df[df["State"].isin(states_to_process)]

    # Calculate Opioid per capita
    df_selected_states["Opioid per capita"] = (
        df_selected_states["MME"] / df_selected_states["Population"]
    )

    # Print the processed data for the selected states
    print(df_selected_states.head())

    # Grab pre and post intervention for the selected states
    selected_states_pre = df_selected_states[
        (df_selected_states["Year"] < 2012) & (df_selected_states["Year"] > 2008)
    ]
    selected_states_post = df_selected_states[
        (df_selected_states["Year"] >= 2012) & (df_selected_states["Year"] < 2015)
    ]
    # Create a design matrix
    X = sm.add_constant(selected_states_pre["Year"])

    # # Fit OLS model
    model = sm.OLS(selected_states_pre["Opioid per capita"], X)
    results = model.fit()
    # # Get predictions
    model_predict = results.get_prediction(X)
    print("The slope is for the three states is: ")
    print(results.params["Year"])
    # # Print mean predictions
    # print(model_predict.summary_frame()["mean"])
    # print(model_predict.conf_int(alpha=0.05))
    mean_predictions = model_predict.summary_frame()["mean"]
    ci = model_predict.conf_int(alpha=0.05)
    ci_l = model_predict.summary_frame()["mean_ci_lower"]
    ci_u = model_predict.summary_frame()["mean_ci_upper"]
    print("here......")
    print(model_predict.summary_frame())
    print(ci)
    plt.plot(
        selected_states_pre["Year"],
        mean_predictions,
        color="green",
        label="Mean Predictions (Pre-2010)",
    )
    # plt.fill_between(
    #     selected_states_pre["Year"],
    #     ci_l,
    #     ci_u,
    #     color="green",
    #     alpha=0.2,
    # )
    X_post = sm.add_constant(selected_states_post["Year"])
    model_post = sm.OLS(selected_states_post["Opioid per capita"], X_post)
    results_post = model_post.fit()

    # Get predictions for florida_post
    model_predict_post = results_post.get_prediction(X_post)
    mean_predictions_post = model_predict_post.summary_frame()["mean"]
    ci = model_predict_post.conf_int(alpha=0.05)
    # ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
    # Plot the mean predictions for florida_post
    plt.plot(
        selected_states_post["Year"],
        mean_predictions_post,
        color="green",
    )
    # plt.fill_between(
    #     selected_states_post["Year"],
    #     ci[:, 0],
    #     ci[:, 1],
    #     color="green",
    #     alpha=0.2,
    # )

    # Print pre and post intervention data for the selected states

    plt.xlabel("Year")
    plt.ylabel("Opioid per capita")
    plt.savefig("../30_results/pre_post_test.png")


def example_pre_post_tx():
    # MO, MN, AR
    """example pre post tx"""
    table = pq.read_table(opiod_pop_month_path)
    df = table.to_pandas()
    df_tx = df[df["State"] == "TX"]
    df_tx["Opioid per capita"] = df_tx["MME"] / df_tx["Population"]
    print(df_tx.head())
    # Texas BEFORE AND AFTER 2007
    # data has years by month YEAR 2006 Month 1, Year 2006 Month 2, etc.
    # 12 months for 2006, and then 12 months after 2007 or 24 months
    # how do I do add the constant when the years are different
    # grab pre and post intervention
    # tx_pre = df_tx[(df_tx["Year"] < 2007) & (df_tx["Year"] > 2005)]
    # tx_post = df_tx[(df_tx["Year"] >= 2007) & (df_tx["Year"] < 2009)]
    # tx_pre.sort_values(by="Year", inplace=True)
    # tx_post.sort_values(by="Year", inplace=True)
    # -12, -11, -10, -9, -8, -7, -6, -5. -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6 to -12

    # df_tx["YearMonth"] = pd.to_datetime(
    #     df_tx["Year"].astype(str) + "-" + df_tx["Month"].astype(str), format="%Y-%m"
    # )

    # Calculate the difference in months between "YearMonth" and 2007-01-01
    df_tx["Months since Policy Implmentation"] = (
        (df_tx["Year"] - 2007) * 12 + df_tx["Month"] - 1
    )
    # print(df_tx["Months since Policy Implmentation"].head())
    tx_pre = df_tx[(df_tx["Months since Policy Implmentation"] < 0)]
    tx_post = df_tx[
        (df_tx["Months since Policy Implmentation"] >= 0)
        & (df_tx["Months since Policy Implmentation"] < 13)
    ]

    tx_pre.sort_values(by="Months since Policy Implmentation", inplace=True)
    tx_post.sort_values(by="Months since Policy Implmentation", inplace=True)

    # tx_post.to_csv("../20_intermediate_files/tx_stuff.csv", index=False)

    X = sm.add_constant(tx_pre["Months since Policy Implmentation"])
    model = sm.OLS(tx_pre["Opioid per capita"], X)
    results = model.fit()
    model_predict = results.get_prediction(X)

    tx_pre["predicted_opiod_per_cap"] = model_predict.summary_frame()["mean"]
    tx_pre[["ci_low", "ci_high"]] = model_predict.conf_int(alpha=0.05)

    X_post = sm.add_constant(tx_post["Months since Policy Implmentation"])
    model_post = sm.OLS(tx_post["Opioid per capita"], X_post)
    results_post = model_post.fit()
    model_predict_post = results_post.get_prediction(X_post)
    tx_post["predicted_opiod_per_cap"] = model_predict_post.summary_frame()["mean"]
    tx_post[["ci_low", "ci_high"]] = model_predict_post.conf_int(alpha=0.05)

    fig, ax = plt.subplots()
    ax.plot(
        tx_pre["Months since Policy Implmentation"],
        tx_pre["predicted_opiod_per_cap"],
        label="Pre-Policy",
    )
    ax.plot(
        tx_post["Months since Policy Implmentation"],
        tx_post["predicted_opiod_per_cap"],
        label="Post-Policy",
    )

    ax.fill_between(
        tx_pre["Months since Policy Implmentation"],
        tx_pre["ci_low"],
        tx_pre["ci_high"],
        alpha=0.2,
    )

    ax.fill_between(
        tx_post["Months since Policy Implmentation"],
        tx_post["ci_low"],
        tx_post["ci_high"],
        alpha=0.2,
    )

    df = table.to_pandas()

    # Filter data for the specified states
    states_to_process = ["MO", "MN", "AR"]
    df_selected_states = df[df["State"].isin(states_to_process)]

    # Calculate Opioid per capita
    df_selected_states["Opioid per capita"] = (
        df_selected_states["MME"] / df_selected_states["Population"]
    )

    df_selected_states["Months since Policy Implmentation"] = (
        (df_selected_states["Year"] - 2007) * 12 + df_selected_states["Month"] - 1
    )

    # df_selected_states.to_csv("../20_intermediate_files/tx_stuff.csv", index=False)

    # grab pre and post intervention
    control_pre = df_selected_states[
        (df_selected_states["Months since Policy Implmentation"] < 0)
    ]
    control_post = df_selected_states[
        (df_selected_states["Months since Policy Implmentation"] >= 0)
        & (df_selected_states["Months since Policy Implmentation"] < 13)
    ]

    control_post.to_csv("../20_intermediate_files/tx_stuff.csv", index=False)

    control_pre.sort_values(by="Months since Policy Implmentation", inplace=True)
    control_post.sort_values(by="Months since Policy Implmentation", inplace=True)

    X_control = sm.add_constant(control_pre["Months since Policy Implmentation"])
    # print(X_control)
    # Fit OLS model
    # print(control_pre["Opioid per capita"])
    model_control = sm.OLS(control_pre["Opioid per capita"], X_control)
    results_control = model_control.fit()
    # Get predictions
    model_predict_control = results_control.get_prediction(X_control)

    control_pre["predicted_opiod_per_cap"] = model_predict_control.summary_frame()[
        "mean"
    ]
    # print(control_pre["predicted_opiod_per_cap"])
    control_pre[["ci_low", "ci_high"]] = model_predict_control.conf_int(alpha=0.05)

    X_control_post = sm.add_constant(control_post["Months since Policy Implmentation"])
    print(X_control_post)
    model_control_post = sm.OLS(control_post["Opioid per capita"], X_control_post)
    results_control_post = model_control_post.fit()
    model_predict_control_post = results_control_post.get_prediction(X_control_post)

    control_post[
        "predicted_opiod_per_cap"
    ] = model_predict_control_post.summary_frame()["mean"]
    control_post[["ci_low", "ci_high"]] = model_predict_control_post.conf_int(
        alpha=0.05
    )

    ax.plot(
        control_pre["Months since Policy Implmentation"],
        control_pre["predicted_opiod_per_cap"],
        label="Control Pre-Policy",
        color="orange",
    )
    ax.plot(
        control_post["Months since Policy Implmentation"],
        control_post["predicted_opiod_per_cap"],
        label="Control Post-Policy",
        color="orange",
    )
    ax.fill_between(
        control_pre["Months since Policy Implmentation"],
        control_pre["ci_low"],
        control_pre["ci_high"],
        alpha=0.2,
        color="orange",
    )

    ax.fill_between(
        control_post["Months since Policy Implmentation"],
        control_post["ci_low"],
        control_post["ci_high"],
        alpha=0.2,
        color="orange",
    )

    ax.axvline(x=0, color="black", linestyle="--")

    ax.set_xlabel("Month")
    ax.set_ylabel("Opioid (MME) per Capita")
    ax.set_title("Texas Average Opioid per Capita Shipments")
    # ax.legend(loc="upper left")
    # ax.set_xlim(
    #     min(tx_pre["Month"].min(), tx_post["Month"].min()),
    #     max(tx_pre["Month"].max(), tx_post["Month"].max()),
    # )
    plt.show()


def find_control_states_cleaned_tx(PRE_DEFINED_STATES):
    """example pre post tx"""
    table = pq.read_table(opiod_pop_month_path)
    df = table.to_pandas()
    df_tx = df[df["State"] == "TX"]
    df_tx["Opioid per capita"] = df_tx["MME"] / df_tx["Population"]
    print(df_tx.head())
    # Texas BEFORE AND AFTER 2007
    # data has years by month YEAR 2006 Month 1, Year 2006 Month 2, etc.
    # 12 months for 2006, and then 12 months after 2007 or 24 months
    # how do I do add the constant when the years are different
    # grab pre and post intervention
    # tx_pre = df_tx[(df_tx["Year"] < 2007) & (df_tx["Year"] > 2005)]
    # tx_post = df_tx[(df_tx["Year"] >= 2007) & (df_tx["Year"] < 2009)]
    # tx_pre.sort_values(by="Year", inplace=True)
    # tx_post.sort_values(by="Year", inplace=True)
    # -12, -11, -10, -9, -8, -7, -6, -5. -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6 to -12

    # df_tx["YearMonth"] = pd.to_datetime(
    #     df_tx["Year"].astype(str) + "-" + df_tx["Month"].astype(str), format="%Y-%m"
    # )

    # Calculate the difference in months between "YearMonth" and 2007-01-01
    df_tx["Months since Policy Implmentation"] = (
        (df_tx["Year"] - 2007) * 12 + df_tx["Month"] - 1
    )
    # print(df_tx["Months since Policy Implmentation"].head())
    tx_pre = df_tx[(df_tx["Months since Policy Implmentation"] < 0)]
    tx_post = df_tx[
        (df_tx["Months since Policy Implmentation"] >= 0)
        & (df_tx["Months since Policy Implmentation"] < 13)
    ]

    tx_pre.sort_values(by="Months since Policy Implmentation", inplace=True)
    tx_post.sort_values(by="Months since Policy Implmentation", inplace=True)

    # tx_post.to_csv("../20_intermediate_files/tx_stuff.csv", index=False)

    X = sm.add_constant(tx_pre["Months since Policy Implmentation"])
    model = sm.OLS(tx_pre["Opioid per capita"], X)
    results = model.fit()
    model_predict = results.get_prediction(X)
    pass




if __name__ == "__main__":
    # find_control_states_prep(PRE_DEFINED_STATES)
    # example_pre_post_fl()
    # controls = find_control_states("FL", PRE_DEFINED_STATES)
    # print(controls)
    # example_pre_post_fl_cleaned()
    # controls = find_control_states_cleaned_fl("FL", PRE_DEFINED_STATES)

    # # print(controls)
    # # Extract states and slopes
    # # states = list(controls.keys())
    # # slopes_values = [slope[0] for slope in controls.values()]
    # # print(states)
    # # print(slopes_values)
    # # Extract the slope for Florida
    # florida_slope = controls["FL"][0]

    # # Calculate absolute differences and store in a new dictionary
    # differences = {
    #     state: abs(slope[0] - florida_slope)
    #     for state, slope in controls.items()
    #     if state != "FL"
    # }

    # # Sort the dictionary by absolute differences in ascending order
    # sorted_differences = dict(sorted(differences.items(), key=operator.itemgetter(1)))

    # # Get the top 3 states with the closest slopes
    # closest_states = list(sorted_differences.keys())[:10]

    # print(f"The three states with the closest slopes to Florida are: {closest_states}")

    # controls = find_control_states_cleaned_wa("WA", PRE_DEFINED_STATES)
    # wa_slope = controls["WA"][0]
    # print(wa_slope)
    # # Calculate absolute differences and store in a new dictionary
    # differences = {
    #     state: abs(slope[0] - wa_slope)
    #     for state, slope in controls.items()
    #     if state != "WA"
    # }

    # # Sort the dictionary by absolute differences in ascending order
    # sorted_differences = dict(sorted(differences.items(), key=operator.itemgetter(1)))

    # # Get the top 4 states with the closest slopes
    # closest_states = list(sorted_differences.keys())[:10]

    # print(f"The four states with the closest slopes to WA are: {closest_states}")
    # print(controls)

    # example_pre_post_wa_cleaned()
    example_pre_post_tx()
