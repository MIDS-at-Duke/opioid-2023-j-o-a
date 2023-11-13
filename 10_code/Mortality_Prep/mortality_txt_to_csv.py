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


# Specify the input folder containing .txt files and the output CSV file
input_folder = "00_data/US_VitalStatistics"
output_csv = "00_data/mortality_raw.csv"

# Convert .txt files to a single CSV file
txt_to_csv(input_folder, output_csv)
