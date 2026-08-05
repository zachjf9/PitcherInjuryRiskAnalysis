import pandas as pd
import re

#Read combined IL file
il_df = pd.read_csv("all_IL.csv")

#Keep only pitchers
il_df = il_df[il_df["Position"].isin(["SP", "RP"])]

#Rename Player column to match feature_engineering.py
il_df = il_df.rename(columns = {"Player": "player_name"})

#Extract injury start date from InjuryOne column
il_df["injury_date"] = (il_df["InjuryOne"].str.extract(r'(\d{1,2}/\d{1,2}/\d{2})'))

#Convert to datetime format
il_df["injury_date"] = pd.to_datetime(il_df["injury_date"], format = "%m/%d/%y")

#Keep only columns needed by create_injury_labels()
il_df = il_df[["player_name", "injury_date"]]

print(il_df.head())

#Save cleaned IL file
il_df.to_csv("all_IL_clean.csv", index = False)

print("Cleaned IL file saved.")