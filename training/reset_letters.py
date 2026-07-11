import shutil
import sys 

import pandas as pd 

if len(sys.argv) < 2:
    raise SystemExit("Usage: python reset_letters.py R T")

labels = {label.upper() for label in sys.argv[1:]}

shutil.copy2(
    "data/landmarks.csv",
    "data/landmarks_before_reset.csv",
)

data = pd.read_csv("data/landmarks.csv")
data = data[~data["label"].isin(labels)]
data.to_csv("data/landmarks.csv", index=False)

print("Removed:", sorted(labels))