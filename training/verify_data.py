import pandas as pd

data = pd.read_csv("data/landmarks.csv")

print(data["label"].value_counts())
print("Total samples:", len(data))
print("Columns:", len(data.columns))
print("Missing Values:", data.isna().sum().sum())