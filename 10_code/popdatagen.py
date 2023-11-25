import pandas as pd

# Define the required columns
cols1 = [
    "STNAME",
    "CTYNAME",
    "POPESTIMATE2002",
    "POPESTIMATE2003",
    "POPESTIMATE2004",
    "POPESTIMATE2005",
    "POPESTIMATE2006",
    "POPESTIMATE2007",
    "POPESTIMATE2008",
    "POPESTIMATE2009",
]

# Load only the required columns from the CSV file
pop1 = pd.read_csv(
    "../20_intermediate_files/2000_2009_NoFIPS.csv", encoding="latin1", usecols=cols1
)


# Define the required columns
cols2 = [
    "STNAME",
    "CTYNAME",
    "POPESTIMATE2010",
    "POPESTIMATE2011",
    "POPESTIMATE2012",
    "POPESTIMATE2013",
    "POPESTIMATE2014",
    "POPESTIMATE2015",
    "POPESTIMATE2016",
    "POPESTIMATE2017",
    "POPESTIMATE2018",
]

# Load only the required columns from the CSV file
pop2 = pd.read_csv(
    "../20_intermediate_files/2010_2018_NoFIPS.csv", encoding="latin1", usecols=cols2
)

# Merge the two DataFrames
poptot = pd.merge(pop1, pop2, on=["STNAME", "CTYNAME"])

# Define a dictionary of state abbreviations
state_abbreviations = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}


# Add a new column 'STATE' that maps the state names to their abbreviations
poptot["STATE"] = poptot["STNAME"].map(state_abbreviations)

# Define a new column order with 'STATE' as the first column
cols = ["STATE"] + [col for col in poptot.columns if col != "STATE"]

# Reorder the columns
poptot = poptot[cols]

# Remove rows with null values in the 'STATE' column (i.e. District of Columbia)
poptot = poptot[poptot["STATE"].notna()]

# Load the FIPS data
fips_data = pd.read_csv("../20_intermediate_files/fips.csv")

# Replace 'Doña Ana County' with 'Do̱a Ana County' in the 'CTYNAME' column to match the FIPS data
poptot["CTYNAME"] = poptot["CTYNAME"].replace("Doña Ana County", "Do̱a Ana County")

# Merge the population data with the FIPS data
popfips = pd.merge(
    poptot,
    fips_data,
    left_on=["STATE", "CTYNAME"],
    right_on=["state", "name"],
    how="left",
    indicator=True,
)

# Remove rows with null values in the 'fips' column, which are 50 values corresponding to total population for each state
popfips = popfips[popfips["_merge"] == "both"]

# popfips.to_csv("../00_data/PopFips.csv", index=False)

# Define the id_vars for the melt function
id_vars = ["STATE", "STNAME", "CTYNAME", "fips", "name", "state"]

# Define the value_vars for the melt function
value_vars = [
    "POPESTIMATE2002",
    "POPESTIMATE2003",
    "POPESTIMATE2004",
    "POPESTIMATE2005",
    "POPESTIMATE2006",
    "POPESTIMATE2007",
    "POPESTIMATE2008",
    "POPESTIMATE2009",
    "POPESTIMATE2010",
    "POPESTIMATE2011",
    "POPESTIMATE2012",
    "POPESTIMATE2013",
    "POPESTIMATE2014",
    "POPESTIMATE2015",
    "POPESTIMATE2016",
    "POPESTIMATE2017",
    "POPESTIMATE2018",
]

# Use the melt function to pivot the DataFrame
melted_popfips = popfips.melt(
    id_vars=id_vars, value_vars=value_vars, var_name="Year", value_name="Population"
)

# Extract the year from the Year column
melted_popfips["Year"] = melted_popfips["Year"].str.extract("(\d+)").astype(int)

# Drop the _merge column
melted_popfips = melted_popfips.drop(columns=["name"])

# Convert the 'Year' column to a datetime object
melted_popfips["Year_datetime"] = pd.to_datetime(melted_popfips["Year"], format="%Y")

melted_popfips.to_csv("../00_data/PopFips.csv", index=False)
