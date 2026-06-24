import pandas as pd
from sklearn.model_selection import train_test_split

print("--- Regenerating Authentic Test Split Samples from Excel Sources ---")

# 1. Load your two original Excel files
# (Replace these strings with your exact filenames from Phase 1)
df1 = pd.read_excel('credit_risk_file_1.xlsx')
df2 = pd.read_excel('credit_risk_file_2.xlsx')

# 2. Replicate your exact Phase 1 merge logic
# (Replace 'Prospect_ID' with the actual common column name you merged on)
raw_df = pd.merge(df1, df2, on='PROSPECTID', how='inner')
print(f"✔ Successfully merged Excel sources. Total shape: {raw_df.shape}")

# 3. Re-execute the exact same Stratified 70/30 split from Phase 4
train_set, test_set = train_test_split(
    raw_df,
    test_size=0.30,
    random_state=42,
    stratify=raw_df['Approved_Flag']
)

# 4. Take the first 100 actual, real-world holdout records
true_samples = test_set.head(100).copy()

# Rename the target column so the app displays it as the historical ground truth
true_samples['True_Approved_Flag'] = true_samples['Approved_Flag']

# 5. Save to the app directory
true_samples.to_csv('app_test_samples.csv', index=True)

print(f"✔ Success! Extracted {len(true_samples)} authentic historical rows from your real test partition.")
print("Saved as 'app_test_samples.csv'. Open your app and you're good to go!")