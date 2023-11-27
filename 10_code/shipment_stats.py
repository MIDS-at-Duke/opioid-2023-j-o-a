import pyarrow.parquet as pq
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

opiod_pop_path = "../00_data/opioid_pop_clean.parquet"


def fl_shipment_stats():
    """get summary stats of opiod shipments for Florida and its control states"""
    table = pq.read_table(opiod_pop_path)

    df = table.to_pandas()
    df_fl = df[df["State"] == "FL"]
    df_fl["Opioid per capita"] = df_fl["MME"] / df_fl["Population"]
    # grab pre and post intervention
    florida_pre = df_fl[(df_fl["Year"] < 2010) & (df_fl["Year"] > 2006)]
    florida_post = df_fl[(df_fl["Year"] >= 2010) & (df_fl["Year"] < 2013)]
    florida_pre.sort_values(by="Year", inplace=True)
    florida_post.sort_values(by="Year", inplace=True)

    # Calculate stats before florida policy
    mean_opioid_pre = florida_pre["Opioid per capita"].mean()
    median_opioid_pre = florida_pre["Opioid per capita"].median()
    min_opioid_pre = florida_pre["Opioid per capita"].min()
    max_opioid_pre = florida_pre["Opioid per capita"].max()

    # Calculate stats after florida policy
    mean_opioid_post = florida_post["Opioid per capita"].mean()
    median_opioid_post = florida_post["Opioid per capita"].median()
    min_opioid_post = florida_post["Opioid per capita"].min()
    max_opioid_post = florida_post["Opioid per capita"].max()

    # Create a DataFrame for results
    fl_results = pd.DataFrame(
        {
            "Period": ["Pre-Policy", "Post-Policy"],
            "Mean Opioid per capita": [mean_opioid_pre, mean_opioid_post],
            "Median Opioid per capita": [median_opioid_pre, median_opioid_post],
            "Minimum Opioid per capita": [min_opioid_pre, min_opioid_post],
            "Maximum Opioid per capita": [max_opioid_pre, max_opioid_post],
        }
    )

    # Now do the control states ["DE", "NV", "TN"]
    df = table.to_pandas()

    # Filter data for the specified states
    states_to_process = ["DE", "NV", "TN"]
    df_selected_states = df[df["State"].isin(states_to_process)]

    # Calculate Opioid per capita
    df_selected_states["Opioid per capita"] = (
        df_selected_states["MME"] / df_selected_states["Population"]
    )

    # grab pre and post intervention
    control_pre = df_selected_states[
        (df_selected_states["Year"] < 2010) & (df_selected_states["Year"] > 2006)
    ]
    control_post = df_selected_states[
        (df_selected_states["Year"] >= 2010) & (df_selected_states["Year"] < 2013)
    ]
    control_pre.sort_values(by="Year", inplace=True)
    control_post.sort_values(by="Year", inplace=True)
    # Calculate mean, median, min, and max for control_pre

    mean_opioid_pre_control = control_pre["Opioid per capita"].mean()
    median_opioid_pre_control = control_pre["Opioid per capita"].median()
    min_opioid_pre_control = control_pre["Opioid per capita"].min()
    max_opioid_pre_control = control_pre["Opioid per capita"].max()

    # Calculate mean, median, min, and max for control_post
    mean_opioid_post_control = control_post["Opioid per capita"].mean()
    median_opioid_post_control = control_post["Opioid per capita"].median()
    min_opioid_post_control = control_post["Opioid per capita"].min()
    max_opioid_post_control = control_post["Opioid per capita"].max()

    # Create a DataFrame
    controls_results = pd.DataFrame(
        {
            "Period": ["Pre-Policy", "Post-Policy"],
            "Mean Opioid per capita": [
                mean_opioid_pre_control,
                mean_opioid_post_control,
            ],
            "Median Opioid per capita": [
                median_opioid_pre_control,
                median_opioid_post_control,
            ],
            "Minimum Opioid per capita": [
                min_opioid_pre_control,
                min_opioid_post_control,
            ],
            "Maximum Opioid per capita": [
                max_opioid_pre_control,
                max_opioid_post_control,
            ],
        }
    )

    df_concatenated = pd.concat([fl_results, controls_results], ignore_index=True)

    df_concatenated.to_csv(
        "../20_intermediate_files/fl_shipment_stats.csv", index=False
    )


def wa_shipment_stats():
    """get summary stats of opiod shipments for Washington and its control states"""

    table = pq.read_table(opiod_pop_path)
    df = table.to_pandas()

    df_wa = df[df["State"] == "WA"]
    df_wa["Opioid per capita"] = df_wa["MME"] / df_wa["Population"]

    wa_pre = df_wa[(df_wa["Year"] < 2012) & (df_wa["Year"] > 2002)]
    wa_post = df_wa[(df_wa["Year"] >= 2012) & (df_wa["Year"] < 2015)]
    wa_pre.sort_values(by="Year", inplace=True)
    wa_post.sort_values(by="Year", inplace=True)

    states_to_process = ["HI", "OH", "MA"]
    df_selected_states = df[df["State"].isin(states_to_process)]

    df_selected_states["Opioid per capita"] = (
        df_selected_states["MME"] / df_selected_states["Population"]
    )

    control_pre = df_selected_states[
        (df_selected_states["Year"] < 2012) & (df_selected_states["Year"] > 2002)
    ]
    control_post = df_selected_states[
        (df_selected_states["Year"] >= 2012) & (df_selected_states["Year"] < 2015)
    ]
    control_pre.sort_values(by="Year", inplace=True)
    control_post.sort_values(by="Year", inplace=True)

    mean_opioid_pre_wa = wa_pre["Opioid per capita"].mean()
    median_opioid_pre_wa = wa_pre["Opioid per capita"].median()
    min_opioid_pre_wa = wa_pre["Opioid per capita"].min()
    max_opioid_pre_wa = wa_pre["Opioid per capita"].max()

    mean_opioid_post_wa = wa_post["Opioid per capita"].mean()
    median_opioid_post_wa = wa_post["Opioid per capita"].median()
    min_opioid_post_wa = wa_post["Opioid per capita"].min()
    max_opioid_post_wa = wa_post["Opioid per capita"].max()

    mean_opioid_pre_control = control_pre["Opioid per capita"].mean()
    median_opioid_pre_control = control_pre["Opioid per capita"].median()
    min_opioid_pre_control = control_pre["Opioid per capita"].min()
    max_opioid_pre_control = control_pre["Opioid per capita"].max()

    mean_opioid_post_control = control_post["Opioid per capita"].mean()
    median_opioid_post_control = control_post["Opioid per capita"].median()
    min_opioid_post_control = control_post["Opioid per capita"].min()
    max_opioid_post_control = control_post["Opioid per capita"].max()

    wa_results = pd.DataFrame(
        {
            "Group": ["Pre-Policy", "Post-Policy"],
            "Mean Opioid per capita": [mean_opioid_pre_wa, mean_opioid_post_wa],
            "Median Opioid per capita": [median_opioid_pre_wa, median_opioid_post_wa],
            "Minimum Opioid per capita": [min_opioid_pre_wa, min_opioid_post_wa],
            "Maximum Opioid per capita": [max_opioid_pre_wa, max_opioid_post_wa],
        }
    )

    control_results = pd.DataFrame(
        {
            "Group": ["Pre-Policy", "Post-Policy"],
            "Mean Opioid per capita": [
                mean_opioid_pre_control,
                mean_opioid_post_control,
            ],
            "Median Opioid per capita": [
                median_opioid_pre_control,
                median_opioid_post_control,
            ],
            "Minimum Opioid per capita": [
                min_opioid_pre_control,
                min_opioid_post_control,
            ],
            "Maximum Opioid per capita": [
                max_opioid_pre_control,
                max_opioid_post_control,
            ],
        }
    )

    df_concatenated = pd.concat([wa_results, control_results], ignore_index=True)

    df_concatenated.to_csv(
        "../20_intermediate_files/wa_shipment_stats.csv", index=False
    )


if __name__ == "__main__":
    fl_shipment_stats()
    wa_shipment_stats()
