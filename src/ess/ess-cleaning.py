import pandas as pd
import os

def clean():
    path = "datasets/ess"
    csvNames = []
    paths = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                csvNames.append(file[:-4])
                paths.append(os.path.join(root, file))

    i = 0
    for path in paths:
        df = pd.read_csv(path, low_memory=False)

        columnsToRemove = [
            "name",
            "edition",
            "proddate",
            "dweight",
            "pspwght",
            "pweight",
            "anweight",
            "prob",
            "stratum",
            "psu"
        ]

        df = df.drop(columns=[col for col in columnsToRemove if col in df.columns])
        csvName = csvNames[i]
        i += 1
        df.to_csv(f"cleaned/ess/{csvName}-cleaned.csv", index=False)

if __name__ == "__main__":
    clean()