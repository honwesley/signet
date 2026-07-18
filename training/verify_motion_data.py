from pathlib import Path

import numpy as np


DATA_DIRECTORY = Path("data/motion")
LABELS = ["J", "Z", "OTHER"]
EXPECTED_SHAPE = (30, 63)


total_files = 0
invalid_files = 0

for label in LABELS:
    label_directory = DATA_DIRECTORY / label
    files = list(label_directory.glob("*.npy"))

    print(f"\n{label}: {len(files)} sequences")

    for filename in files:
        sequence = np.load(filename)

        if sequence.shape != EXPECTED_SHAPE:
            print(
                f"  Invalid: {filename.name} "
                f"has shape {sequence.shape}"
            )
            invalid_files += 1

    total_files += len(files)

print(f"\nTotal motion sequences: {total_files}")
print(f"Invalid motion sequences: {invalid_files}")

if invalid_files == 0:
    print("All motion files are valid.")