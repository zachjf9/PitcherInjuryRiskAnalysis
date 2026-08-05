import pandas as pd

il_files = [
    "2015IL.csv",
    "2016IL.csv",
    "2017IL.csv",
    "2018IL.csv",
    "2019IL.csv",
    "2021IL.csv",
    "2022IL.csv",
    "2023IL.csv",
    "2024IL.csv",
    "2025IL.csv",
]

dfs = []

for file in il_files:
    df = pd.read_csv(file)
    df["season"] = file[:4]
    dfs.append(df)

all_il = pd.concat(dfs, ignore_index=True)

print(all_il.head())
print(all_il.columns)

all_il.to_csv("all_IL.csv", index=False)