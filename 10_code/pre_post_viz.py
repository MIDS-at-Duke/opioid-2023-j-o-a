import pandas as pd
import matplotlib.pyplot as plt

# Loading Florida parquet file from 00_data/states
florida_data = pd.read_parquet("../00_data/FL_Opioid_Pop.parquet")

# Renaming these: "desoto": "de soto", "st. lucie": "saint lucie", "st. johns": "saint johns",
florida_data["County"] = florida_data["County"].replace(
    {"desoto": "de soto", "st. lucie": "saint lucie", "st. johns": "saint johns"}
)

# florida_data = florida_data[florida_data["Population"] > 250000]

florida_data["Opioid_Consumption_MME"] = florida_data["MME"]

florida_data["Opioid Per Capita"] = (
    florida_data["Opioid_Consumption_MME"] / florida_data["Population"]
) * 1000

# Visualizing pre-post data for Florida before 2010 and after 2010
florida_data_pre = florida_data[florida_data["YEAR"].between(2007, 2009)]
florida_data_post = florida_data[florida_data["YEAR"].between(2010, 2012)]

# Creating OLS models for pre and post 2010
import statsmodels.api as sm

# For Pre-2010
X_pre = sm.add_constant(florida_data_pre["YEAR"])
y_pre = florida_data_pre["Opioid Per Capita"]

model_pre = sm.OLS(y_pre, X_pre)
results_pre = model_pre.fit()

# For Post-2010
X_post = sm.add_constant(florida_data_post["YEAR"])
y_post = florida_data_post["Opioid Per Capita"]

model_post = sm.OLS(y_post, X_post)
results_post = model_post.fit()

plt.figure(figsize=(8, 6))


# Plotting regression lines
plt.plot(
    florida_data_pre["YEAR"],
    results_pre.predict(X_pre),
    color="red",
    label="Pre-2010 OLS",
)
plt.plot(
    florida_data_post["YEAR"],
    results_post.predict(X_post),
    color="blue",
    label="Post-2010 OLS",
)

plt.xlabel("Year")
plt.ylabel("Opioid Consumption per Capita")
plt.title("Opioid Consumption per Capita in Florida (Pre and Post 2010)")
plt.legend()
plt.grid(True)
plt.show()
