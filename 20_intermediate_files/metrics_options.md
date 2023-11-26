Some options for metrics to choose control states:

1. Checking the slope of the treatment state and then compare against slope of control states for necessary years. 

Sample code (we would need to change a lot, but this is the general idea):

```
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors

    # Assume 'data' is a dictionary where each key is a state code and each value is the corresponding dataset

    # Define predictors for propensity score model
    predictors = ['Population', 'Other_Variables']  # Define appropriate predictors

    matched_states = {}

    for state_code, state_data in data.items():
        # Create a DataFrame without the current state
        control_states = pd.concat([data[state] for s, data[state] in data.items() if s != state_code])

        # Fit logistic regression to predict the likelihood of being the current state
        lr = LogisticRegression()
        lr.fit(control_states[predictors], control_states['STATE'] == state_code)

        # Calculate propensity scores for the current state
        state_data['Propensity_Score'] = lr.predict_proba(state_data[predictors])[:, 1]

        # Match control states to the current state based on propensity scores
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(state_data['Propensity_Score'].values.reshape(-1, 1))
        distances, indices = nn.kneighbors(control_states['Propensity_Score'].values.reshape(-1, 1))

        # Get matched control states for the current state
        matched_states[state_code] = control_states.iloc[indices.flatten()]

        # Compare slopes of opioid consumption for the current state and matched control states
        # Use the method from the previous example to calculate slopes and compare for each state
```

2. Economic Metrics: 
    a. Propensity Score: This involves a logistic regression model to predict the probability of being in the treatment group using the covariates.
        Logistic Regression: Predict the likelihood of high opioid consumption based on available demographic variables (from population data) and any other relevant factors.
        Calculate Propensity Scores: After training the logistic regression model, use it to predict propensity scores for each county or region. These scores represent the likelihood of higher opioid consumption based on observed characteristics.
    
    ```
    from sklearn.linear_model import LogisticRegression

    # Assuming 'population_data' contains demographic variables and other relevant factors

    # Define the covariates you want to use for the logistic regression model
    covariates = ['Population', 'Other_Variables']  # Add other relevant variables

    # Assuming 'treatment_group' contains the opioid consumption data for the treatment group

    # Train a logistic regression model to predict the likelihood of high opioid consumption
    lr = LogisticRegression()
    lr.fit(population_data[covariates], treatment_group['High_Opioid_Consumption'])

    # Predict propensity scores for each county or region
    population_data['Propensity_Score'] = lr.predict_proba(population_data[covariates])[:, 1]

    ```
    Replace 'High_Opioid_Consumption' with the relevant column from your dataset that denotes high opioid consumption or any binary indicator of being in the treatment group. Adjust the covariates list to include the demographic variables and other relevant factors you want to use in the logistic regression model.
    
    b. Matching Score: This involves matching the treatment state with the control states based on the propensity score. 
        Propensity Score Matching: Match counties or regions based on their propensity scores to find similar or comparable groups regarding opioid consumption propensity.
        Nearest Neighbor Matching: Identify pairs or groups of counties with similar demographic characteristics and opioid consumption levels.
    
    ```
    from sklearn.neighbors import NearestNeighbors
    import numpy as np

    # Assuming you have propensity scores calculated in 'population_data'

    # Extract propensity scores and treatment status (FL vs other states)
    propensity_scores = population_data['Propensity_Score'].values.reshape(-1, 1)
    treatment_status = (population_data['State'] == 'FL').astype(int)

    # Combine propensity scores and treatment status
    features = np.hstack((propensity_scores, treatment_status.values.reshape(-1, 1)))

    # Separate treatment and control groups
    treatment_group = features[treatment_status == 1]
    control_group = features[treatment_status == 0]

    # Initialize Nearest Neighbors model
    nn_model = NearestNeighbors(n_neighbors=1)
    nn_model.fit(control_group)

    # Match each treated unit to its nearest neighbor in the control group
    distances, indices = nn_model.kneighbors(treatment_group)

    # 'indices' contains the indices of matched control units for each treated unit
    matched_control_indices = indices.flatten()

    # Retrieve the matched control group based on the indices
    matched_control_group = control_group[matched_control_indices]

    # Now 'matched_control_group' contains the matched control units

    ```
    
    c. Euclidean distance using N variables: This involves calculating the Euclidean distance between the treatment state and control states based on the propensity score (honestly, it's very similar to the slope approach)
        Feature Standardization: Standardize (normalize) the data, especially opioid consumption, mortality numbers, and population counts, to ensure they are on the same scale.
        Euclidean Distance Calculation: Calculate the Euclidean distance between counties or regions using normalized values of opioid consumption, mortality rates, and population counts. This method identifies similarities or dissimilarities between areas based on these factors.

```
    from sklearn.preprocessing import StandardScaler

    # Assuming you have datasets for consumption_g, Deaths, Population for each state

    # Standardize the variables
    scaler = StandardScaler()
    consumption_scaled = scaler.fit_transform(consumption_g.values.reshape(-1, 1))
    deaths_scaled = scaler.fit_transform(Deaths.values.reshape(-1, 1))
    population_scaled = scaler.fit_transform(Population.values.reshape(-1, 1))

    # Concatenate standardized variables into one matrix
    features = np.hstack((consumption_scaled, deaths_scaled, population_scaled))

    # Extract data for FL (treatment state)
    fl_features = features[fl_state_index]  # Replace 'fl_state_index' with the index of FL in your data

    # Calculate Euclidean distance with other states
    euclidean_distances = np.linalg.norm(features - fl_features, axis=1)

    # 'euclidean_distances' now contains the Euclidean distances between FL and other states
```


3. Opioid Consumption-Mortality joint metrics:
    a. Opioid related deaths per opioid consumption (here, we could use the variable MME_conversion_factor)
    This factor standardizes different opioids based on their potency relative to morphine. 
    It allows for the conversion of various opioids into a common unit (morphine equivalent), 
    enabling a more accurate comparison of opioid consumption across different types and doses. 
    Using MME can provide a standardized measure of opioid consumption, facilitating comparisons across different drugs and quantities.

    ```
    # List of file paths for each state dataset
state_files = [
    'path_to_state_1.csv',
    'path_to_state_2.csv',
    # Add paths for other states...
]

deaths_per_MME_results = {}

for state_file in state_files:
    # Read the dataset for each state
    state_data = pd.read_csv(state_file)

    # Calculate deaths per MME consumption for the state
    MME_conversion_factor = 1.5  # Replace with the actual conversion factor
    MME_consumption = state_data['consumption_g'] * MME_conversion_factor
    deaths_per_MME = state_data['Deaths'] / MME_consumption

    # Store deaths per MME consumption for the state
    state_name = extract_state_name(state_file)  # Define a function to extract state name
    deaths_per_MME_results[state_name] = deaths_per_MME

# Analyze and compare deaths per MME consumption across states
# Use statistical analysis or visualization techniques to compare the results

    ```

    b. Mortality-to-consumption Ratio: This ratio calculates the proportion of opioid-related deaths concerning the volume of opioid consumption within a specific population or region. 
    It aims to correlate mortality rates with opioid consumption levels.

    ```
    # Sample dictionary to store results
mortality_consumption_ratio = {}

for state_file in state_files:
    # Read the dataset for each state
    state_data = pd.read_csv(state_file)

    # Calculate total deaths and total consumption for the state
    total_deaths = state_data['Deaths'].sum()
    total_consumption = (state_data['consumption_g'] * MME_conversion_factor).sum()  # Adjust this based on your data

    # Calculate Mortality-to-Consumption ratio
    if total_consumption != 0:
        mortality_consumption_ratio[state_name] = total_deaths / total_consumption
    else:
        mortality_consumption_ratio[state_name] = 0  # Handle zero division

# Analyze and compare the Mortality-to-Consumption ratio across states
# You can store this data in a DataFrame or dictionary and further analyze or visualize the results.

    ```