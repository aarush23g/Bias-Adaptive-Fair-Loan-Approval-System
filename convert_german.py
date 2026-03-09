import pandas as pd

# File paths
input_path = r"D:\bias_adaptive_fair_loan_approval\data\raw\adult.data"
output_path = r"D:\bias_adaptive_fair_loan_approval\data\raw\adult.csv"

# Column names for the German Credit dataset (UCI version)
columns = [
"age","workclass","fnlwgt","education","education_num",
"marital_status","occupation","relationship","race","sex",
"capital_gain","capital_loss","hours_per_week","native_country","income"
]

# Read space-separated file
df = pd.read_csv(input_path, sep=' ', header=None)

# Assign column names
df.columns = columns

# Save as CSV
df.to_csv(output_path, index=False)

print("Conversion completed successfully!")