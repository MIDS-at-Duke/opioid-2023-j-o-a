import os
import pyarrow.parquet as pq
import pandas as pd

# Specify the path to the folder containing Parquet files
folder_path = "../00_data/states/"

# Initialize a variable to store the total number of rows
total_rows = 0

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".parquet"):
        # Build the full path to the Parquet file
        parquet_file = os.path.join(folder_path, filename)

        # Load the Parquet file
        table = pq.read_table(parquet_file)

        # Convert the table to a pandas DataFrame
        df = table.to_pandas()

        # Add the number of rows in the current file to the total
        total_rows += df.shape[0]

# Display the total number of rows
print("Total Rows Across All Files:", total_rows)
# Total Rows Across All Files: 329463664

# wc -l arcos_all_washpost.tsv
# 329755895 arcos_all_washpost.tsv
