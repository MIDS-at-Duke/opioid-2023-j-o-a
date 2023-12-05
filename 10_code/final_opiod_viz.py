import pyarrow.parquet as pq
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import operator
import seaborn.objects as so
import matplotlib.patches as mpatches

opiod_pop_path = "../00_data/opioid_pop_clean.parquet"
opiod_pop_month_path = "../00_data/opioid_pop_months_clean.parquet"


def pre_post_fl_cleaned():
    """make pre-post viz for Florida"""
    table = pq.read_table(opiod_pop_path)

    df = table.to_pandas()
    df_fl = df[df["State"] == "FL"]
    df_fl["Opioid per capita"] = df_fl["MME"] / df_fl["Population"]

    # grab pre and post intervention
    florida_pre = df_fl[(df_fl["Year"] < 2010) & (df_fl["Year"] > 2006)]
    florida_post = df_fl[(df_fl["Year"] >= 2010) & (df_fl["Year"] < 2013)]
    florida_pre.sort_values(by="Year", inplace=True)
    florida_post.sort_values(by="Year", inplace=True)

    # Create a design matrix
    X = sm.add_constant(florida_pre["Year"])

    # # Fit OLS model
    model = sm.OLS(florida_pre["Opioid per capita"], X)
    results = model.fit()
    # Get predictions
    model_predict = results.get_prediction(X)

    florida_pre["predicted_opiod_per_cap"] = model_predict.summary_frame()["mean"]
    florida_pre[["ci_low", "ci_high"]] = model_predict.conf_int(alpha=0.05)

    X_post = sm.add_constant(florida_post["Year"])
    model_post = sm.OLS(florida_post["Opioid per capita"], X_post)
    results_post = model_post.fit()
    model_predict_post = results_post.get_prediction(X_post)

    florida_post["predicted_opiod_per_cap"] = model_predict_post.summary_frame()["mean"]
    florida_post[["ci_low", "ci_high"]] = model_predict_post.conf_int(alpha=0.05)

    fig, ax = plt.subplots()
    ax.plot(
        florida_pre["Year"],
        florida_pre["predicted_opiod_per_cap"],
        label="Pre-Policy",
    )
    ax.plot(
        florida_post["Year"],
        florida_post["predicted_opiod_per_cap"],
        label="Post-Policy",
    )
    ax.fill_between(
        florida_pre["Year"],
        florida_pre["ci_low"],
        florida_pre["ci_high"],
        alpha=0.2,
    )

    ax.fill_between(
        florida_post["Year"],
        florida_post["ci_low"],
        florida_post["ci_high"],
        alpha=0.2,
    )

    ax.axvline(x=2010, color="black", linestyle="--")

    ax.set_xlabel("Year")
    ax.set_ylabel("Opioid (MME) per Capita")
    ax.set_title("Florida Average Opioid per Capita Shipments")
    ax.legend(loc="upper left")
    ax.set_xlim(
        min(florida_pre["Year"].min(), florida_post["Year"].min()),
        max(florida_pre["Year"].max(), florida_post["Year"].max()),
    )
    plt.savefig("../30_results/fl_pre_post.png")
    plt.show()


def pre_post_wa_cleaned():
    """make pre-post viz for Washington"""

    table = pq.read_table(opiod_pop_path)

    df = table.to_pandas()
    df_wa = df[df["State"] == "WA"]
    df_wa["Opioid per capita"] = df_wa["MME"] / df_wa["Population"]

    # grab pre and post intervention
    wa_pre = df_wa[(df_wa["Year"] < 2012) & (df_wa["Year"] > 2002)]
    wa_post = df_wa[(df_wa["Year"] >= 2012) & (df_wa["Year"] < 2015)]
    wa_pre.sort_values(by="Year", inplace=True)
    wa_post.sort_values(by="Year", inplace=True)

    # Create a design matrix
    X = sm.add_constant(wa_pre["Year"])

    # # Fit OLS model
    model = sm.OLS(wa_pre["Opioid per capita"], X)
    results = model.fit()
    # Get predictions
    model_predict = results.get_prediction(X)

    wa_pre["predicted_opiod_per_cap"] = model_predict.summary_frame()["mean"]
    wa_pre[["ci_low", "ci_high"]] = model_predict.conf_int(alpha=0.05)

    X_post = sm.add_constant(wa_post["Year"])
    model_post = sm.OLS(wa_post["Opioid per capita"], X_post)
    results_post = model_post.fit()
    model_predict_post = results_post.get_prediction(X_post)

    wa_post["predicted_opiod_per_cap"] = model_predict_post.summary_frame()["mean"]
    wa_post[["ci_low", "ci_high"]] = model_predict_post.conf_int(alpha=0.05)

    fig, ax = plt.subplots()
    ax.plot(
        wa_pre["Year"],
        wa_pre["predicted_opiod_per_cap"],
        label="Pre-Policy",
    )
    ax.plot(
        wa_post["Year"],
        wa_post["predicted_opiod_per_cap"],
        label="Post-Policy",
    )
    ax.fill_between(
        wa_pre["Year"],
        wa_pre["ci_low"],
        wa_pre["ci_high"],
        alpha=0.2,
    )

    ax.fill_between(
        wa_post["Year"],
        wa_post["ci_low"],
        wa_post["ci_high"],
        alpha=0.2,
    )

    ax.axvline(x=2012, color="black", linestyle="--")

    ax.set_xlabel("Year")
    ax.set_ylabel("Opioid (MME) per capita ")
    ax.set_title("Washington Average Opioid per Capita Shipments")
    ax.set_xlim(
        min(wa_pre["Year"].min(), wa_post["Year"].min()),
        max(wa_pre["Year"].max(), wa_post["Year"].max()),
    )
    ax.legend()
    plt.savefig("../30_results/wa_pre_post.png")
    plt.show()


def diff_diff_fl_cleaned():
    """Make diff-diff viz for Florida"""

    table = pq.read_table(opiod_pop_path)

    df = table.to_pandas()
    df_fl = df[df["State"] == "FL"]
    df_fl["Opioid per capita"] = df_fl["MME"] / df_fl["Population"]

    # grab pre and post intervention
    florida_pre = df_fl[(df_fl["Year"] < 2010) & (df_fl["Year"] > 2006)]
    florida_post = df_fl[(df_fl["Year"] >= 2010) & (df_fl["Year"] < 2013)]
    florida_pre.sort_values(by="Year", inplace=True)
    florida_post.sort_values(by="Year", inplace=True)

    # Create a design matrix
    X = sm.add_constant(florida_pre["Year"])

    # Fit OLS model
    model = sm.OLS(florida_pre["Opioid per capita"], X)
    results = model.fit()
    # Get predictions
    model_predict = results.get_prediction(X)

    florida_pre["predicted_opiod_per_cap"] = model_predict.summary_frame()["mean"]
    florida_pre[["ci_low", "ci_high"]] = model_predict.conf_int(alpha=0.05)

    X_post = sm.add_constant(florida_post["Year"])
    model_post = sm.OLS(florida_post["Opioid per capita"], X_post)
    results_post = model_post.fit()
    model_predict_post = results_post.get_prediction(X_post)

    florida_post["predicted_opiod_per_cap"] = model_predict_post.summary_frame()["mean"]
    florida_post[["ci_low", "ci_high"]] = model_predict_post.conf_int(alpha=0.05)

    fig, ax = plt.subplots()
    ax.plot(
        florida_pre["Year"],
        florida_pre["predicted_opiod_per_cap"],
        label="FL Pre-Policy",
        color="blue",
    )
    ax.plot(
        florida_post["Year"],
        florida_post["predicted_opiod_per_cap"],
        label="FL Post-Policy",
        color="blue",
    )
    ax.fill_between(
        florida_pre["Year"],
        florida_pre["ci_low"],
        florida_pre["ci_high"],
        alpha=0.2,
        color="blue",
    )

    ax.fill_between(
        florida_post["Year"],
        florida_post["ci_low"],
        florida_post["ci_high"],
        alpha=0.2,
        color="blue",
    )

    # Now do the control states ["DE", "NV", "TN"]
    df = table.to_pandas()

    # Filter data for the specified states
    states_to_process = ["KY", "WV", "TN", "NV", "OR"]
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

    X_control = sm.add_constant(control_pre["Year"])

    # Fit OLS model
    model_control = sm.OLS(control_pre["Opioid per capita"], X_control)
    results_control = model_control.fit()
    # Get predictions
    model_predict_control = results_control.get_prediction(X_control)

    control_pre["predicted_opiod_per_cap"] = model_predict_control.summary_frame()[
        "mean"
    ]
    control_pre[["ci_low", "ci_high"]] = model_predict_control.conf_int(alpha=0.05)

    X_control_post = sm.add_constant(control_post["Year"])
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
        control_pre["Year"],
        control_pre["predicted_opiod_per_cap"],
        label="Control Pre-Policy",
        color="orange",
    )
    ax.plot(
        control_post["Year"],
        control_post["predicted_opiod_per_cap"],
        label="Control Post-Policy",
        color="orange",
    )
    ax.fill_between(
        control_pre["Year"],
        control_pre["ci_low"],
        control_pre["ci_high"],
        alpha=0.2,
        color="orange",
    )

    ax.fill_between(
        control_post["Year"],
        control_post["ci_low"],
        control_post["ci_high"],
        alpha=0.2,
        color="orange",
    )

    ax.axvline(x=2010, color="black", linestyle="--")

    ax.set_xlabel("Year")
    ax.set_ylabel("Opioid (MME) per Capita")
    ax.set_title("Average Opioid per Capita Shipments")
    orange_patch = mpatches.Patch(color="orange", label="KY, WV, TN, NV, OR")
    blue_patch = mpatches.Patch(color="blue", label="FL")

    plt.legend(handles=[orange_patch, blue_patch], loc="upper left")

    ax.set_xlim(
        min(florida_pre["Year"].min(), florida_post["Year"].min()),
        max(florida_pre["Year"].max(), florida_post["Year"].max()),
    )
    plt.savefig("../30_results/fl_diff_diff.png")
    plt.show()


def diff_diff_wa_cleaned():
    """make diff-diff viz for Washington"""

    table = pq.read_table(opiod_pop_path)

    df = table.to_pandas()
    df_wa = df[df["State"] == "WA"]
    df_wa["Opioid per capita"] = df_wa["MME"] / df_wa["Population"]

    # grab pre and post intervention
    wa_pre = df_wa[(df_wa["Year"] < 2012) & (df_wa["Year"] > 2002)]
    wa_post = df_wa[(df_wa["Year"] >= 2012) & (df_wa["Year"] < 2015)]
    wa_pre.sort_values(by="Year", inplace=True)
    wa_post.sort_values(by="Year", inplace=True)

    # Create a design matrix
    X = sm.add_constant(wa_pre["Year"])

    # # Fit OLS model
    model = sm.OLS(wa_pre["Opioid per capita"], X)
    results = model.fit()
    # Get predictions
    model_predict = results.get_prediction(X)

    wa_pre["predicted_opiod_per_cap"] = model_predict.summary_frame()["mean"]
    wa_pre[["ci_low", "ci_high"]] = model_predict.conf_int(alpha=0.05)

    X_post = sm.add_constant(wa_post["Year"])
    model_post = sm.OLS(wa_post["Opioid per capita"], X_post)
    results_post = model_post.fit()
    model_predict_post = results_post.get_prediction(X_post)

    wa_post["predicted_opiod_per_cap"] = model_predict_post.summary_frame()["mean"]
    wa_post[["ci_low", "ci_high"]] = model_predict_post.conf_int(alpha=0.05)

    fig, ax = plt.subplots()
    ax.plot(
        wa_pre["Year"],
        wa_pre["predicted_opiod_per_cap"],
        label="Pre-Policy",
        color="blue",
    )
    ax.plot(
        wa_post["Year"],
        wa_post["predicted_opiod_per_cap"],
        label="Post-Policy",
        color="blue",
    )
    ax.fill_between(
        wa_pre["Year"],
        wa_pre["ci_low"],
        wa_pre["ci_high"],
        alpha=0.2,
        color="blue",
    )

    ax.fill_between(
        wa_post["Year"],
        wa_post["ci_low"],
        wa_post["ci_high"],
        alpha=0.2,
        color="blue",
    )

    df = table.to_pandas()

    # Filter data for the specified states
    states_to_process = ["OH", "MI", "ME", "HI"]
    df_selected_states = df[df["State"].isin(states_to_process)]

    # Calculate Opioid per capita
    df_selected_states["Opioid per capita"] = (
        df_selected_states["MME"] / df_selected_states["Population"]
    )

    # grab pre and post intervention
    control_pre = df_selected_states[
        (df_selected_states["Year"] < 2012) & (df_selected_states["Year"] > 2002)
    ]
    control_post = df_selected_states[
        (df_selected_states["Year"] >= 2012) & (df_selected_states["Year"] < 2015)
    ]

    control_pre.sort_values(by="Year", inplace=True)
    control_post.sort_values(by="Year", inplace=True)

    X_control = sm.add_constant(control_pre["Year"])

    # Fit OLS model
    model_control = sm.OLS(control_pre["Opioid per capita"], X_control)
    results_control = model_control.fit()
    # Get predictions
    model_predict_control = results_control.get_prediction(X_control)

    control_pre["predicted_opiod_per_cap"] = model_predict_control.summary_frame()[
        "mean"
    ]
    control_pre[["ci_low", "ci_high"]] = model_predict_control.conf_int(alpha=0.05)

    X_control_post = sm.add_constant(control_post["Year"])
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
        control_pre["Year"],
        control_pre["predicted_opiod_per_cap"],
        label="Control Pre-Policy",
        color="orange",
    )
    ax.plot(
        control_post["Year"],
        control_post["predicted_opiod_per_cap"],
        label="Control Post-Policy",
        color="orange",
    )
    ax.fill_between(
        control_pre["Year"],
        control_pre["ci_low"],
        control_pre["ci_high"],
        alpha=0.2,
        color="orange",
    )

    ax.fill_between(
        control_post["Year"],
        control_post["ci_low"],
        control_post["ci_high"],
        alpha=0.2,
        color="orange",
    )
    orange_patch = mpatches.Patch(color="orange", label="OH, MI, ME, HI")
    blue_patch = mpatches.Patch(color="blue", label="WA")

    plt.legend(handles=[orange_patch, blue_patch])

    ax.axvline(x=2012, color="black", linestyle="--")

    ax.set_xlabel("Year")
    ax.set_ylabel("Opioid (MME) per capita ")
    ax.set_title("Average Opioid per Capita Shipments")
    ax.set_xlim(
        min(wa_pre["Year"].min(), wa_post["Year"].min()),
        max(wa_pre["Year"].max(), wa_post["Year"].max()),
    )
    plt.savefig("../30_results/wa_diff_diff.png")
    plt.show()


def pre_post_tx_cleaned():
    """generate texas pre-post"""
    table = pq.read_table(opiod_pop_month_path)
    df = table.to_pandas()
    df_tx = df[df["State"] == "TX"]
    df_tx["Opioid per capita"] = df_tx["MME"] / df_tx["Population"]
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

    ax.axvline(x=0, color="black", linestyle="--")

    ax.set_xlabel("Months since Policy Implmentation")
    ax.set_ylabel("Opioid (MME) per Capita")
    ax.set_title("Texas Average Opioid per Capita Shipments")
    ax.legend(loc="upper left")

    ax.set_xlim(
        min(
            tx_pre["Months since Policy Implmentation"].min(),
            tx_post["Months since Policy Implmentation"].min(),
        ),
        max(
            tx_pre["Months since Policy Implmentation"].max(),
            tx_post["Months since Policy Implmentation"].max(),
        ),
    )
    plt.savefig("../30_results/tx_pre_post.png")
    plt.show()


def diff_diff_tx_cleaned():
    """diff diff plot for texas"""
    table = pq.read_table(opiod_pop_month_path)
    df = table.to_pandas()
    df_tx = df[df["State"] == "TX"]
    df_tx["Opioid per capita"] = df_tx["MME"] / df_tx["Population"]

    df_tx["Months since Policy Implmentation"] = (
        (df_tx["Year"] - 2007) * 12 + df_tx["Month"] - 1
    )
    tx_pre = df_tx[(df_tx["Months since Policy Implmentation"] < 0)]
    tx_post = df_tx[
        (df_tx["Months since Policy Implmentation"] >= 0)
        & (df_tx["Months since Policy Implmentation"] < 13)
    ]

    tx_pre.sort_values(by="Months since Policy Implmentation", inplace=True)
    tx_post.sort_values(by="Months since Policy Implmentation", inplace=True)

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
        color="blue",
    )
    ax.plot(
        tx_post["Months since Policy Implmentation"],
        tx_post["predicted_opiod_per_cap"],
        label="Post-Policy",
        color="blue",
    )

    ax.fill_between(
        tx_pre["Months since Policy Implmentation"],
        tx_pre["ci_low"],
        tx_pre["ci_high"],
        alpha=0.2,
        color="blue",
    )

    ax.fill_between(
        tx_post["Months since Policy Implmentation"],
        tx_post["ci_low"],
        tx_post["ci_high"],
        alpha=0.2,
        color="blue",
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
    # grab pre and post intervention
    control_pre = df_selected_states[
        (df_selected_states["Months since Policy Implmentation"] < 0)
    ]
    control_post = df_selected_states[
        (df_selected_states["Months since Policy Implmentation"] >= 0)
        & (df_selected_states["Months since Policy Implmentation"] < 13)
    ]

    control_pre.sort_values(by="Months since Policy Implmentation", inplace=True)
    control_post.sort_values(by="Months since Policy Implmentation", inplace=True)

    X_control = sm.add_constant(control_pre["Months since Policy Implmentation"])

    model_control = sm.OLS(control_pre["Opioid per capita"], X_control)
    results_control = model_control.fit()
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

    ax.set_xlabel("Months since Policy Implmentation")
    ax.set_ylabel("Opioid (MME) per Capita")
    ax.set_title("Average Opioid per Capita Shipments")
    ax.set_xlim(
        min(
            tx_pre["Months since Policy Implmentation"].min(),
            tx_post["Months since Policy Implmentation"].min(),
        ),
        max(
            tx_pre["Months since Policy Implmentation"].max(),
            tx_post["Months since Policy Implmentation"].max(),
        ),
    )
    orange_patch = mpatches.Patch(color="orange", label="MO, MN, AR")
    blue_patch = mpatches.Patch(color="blue", label="TX")

    plt.legend(handles=[orange_patch, blue_patch])
    plt.savefig("../30_results/tx_diff_diff.png")

    plt.show()


if __name__ == "__main__":
    # pre_post_fl_cleaned()
    # pre_post_wa_cleaned()
    # diff_diff_fl_cleaned()
    # diff_diff_wa_cleaned()
    # pre_post_tx_cleaned()
    diff_diff_tx_cleaned()