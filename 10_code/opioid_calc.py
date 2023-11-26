import pandas as pd

consumption = pd.read_parquet("../00_data/FL_Opioid_Pop.parquet")
consumption = consumption.rename(columns={"BUYER_COUNTY": "County"})

consumption["Opioid_Consumption_MME"] = (
    consumption["DOSAGE_UNIT"] * consumption["MME_Conversion_Factor"]
)

# If CALC_BASE_WT_IN_GM seems relevant, consider its impact on consumption
# Normalize CALC_BASE_WT_IN_GM if necessary
consumption["Normalized_CALC_BASE_WT_IN_GM"] = (
    consumption["CALC_BASE_WT_IN_GM"] - consumption["CALC_BASE_WT_IN_GM"].min()
) / (consumption["CALC_BASE_WT_IN_GM"].max() - consumption["CALC_BASE_WT_IN_GM"].min())

# Calculate total opioid consumption considering both Dosage Units and Weight if necessary
consumption["Total_Opioid_Consumption"] = (
    consumption["Normalized_CALC_BASE_WT_IN_GM"] + consumption["Opioid_Consumption_MME"]
)
