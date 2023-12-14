"""converts mortality files to csv files"""
import os
import pandas as pd


def txt_to_csv(input_folder, output_csv):
    # Initialize an empty DataFrame
    df = pd.DataFrame()

    # Loop through each .txt file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder, filename)

            # Read the .txt file into a DataFrame
            txt_df = pd.read_csv(
                file_path, delimiter="\t"
            )  # Change delimiter if necessary

            # Append the DataFrame to the main DataFrame
            df = pd.concat([df, txt_df], ignore_index=True)

    # Save the combined DataFrame to a CSV file
    df.to_csv(output_csv, index=False)


current_dir = "00_data/US_VitalStatistics"

# Specify the input folder containing .txt files and the output CSV file
input_folder = os.path.join(current_dir, "20_intermediate_files/US_VitalStatistics")
output_csv = os.path.join(current_dir, "20_intermediate_files/mortality_raw.csv")

# Convert .txt files to a single CSV file
txt_to_csv(input_folder, output_csv)


# Cleaning up raw mortality data
# Loading in raw mortality data
mortality_data = pd.read_csv(output_csv)

# Filling na values in "Drug/Alcohol Induced Cause" column with empty string to avoid errors
mortality_data["Drug/Alcohol Induced Cause"] = mortality_data[
    "Drug/Alcohol Induced Cause"
].fillna("")

# Creating a new column "Drug/Alcohol Induced" that is True if "Drug/Alcohol Induced Cause" column contains "Drug poisonings" or "drug-induced"
relevant_mortality_data = mortality_data[
    mortality_data["Drug/Alcohol Induced Cause"].str.contains(
        "Drug poisonings|drug-induced", case=False
    )
]

# Changing County Code and Year columns to int type to avoid errors
relevant_mortality_data.loc[:, "County Code"] = relevant_mortality_data[
    "County Code"
].astype(int)
relevant_mortality_data.loc[:, "Year"] = relevant_mortality_data["Year"].astype(int)


# Check if "County" column exists in the DataFrame
if "County" in relevant_mortality_data.columns:
    # Split "County" column into "County" and "State" columns
    relevant_mortality_data[["County", "State"]] = relevant_mortality_data[
        "County"
    ].str.split(", ", expand=True)


output_file_path = os.path.join(current_dir, "00_data", "mortality_final.csv")

# Save relevant_mortality_data to a CSV file
relevant_mortality_data.to_csv(output_file_path, index=False)
