
import pandas as pd

file_vehicules = "Final_Matching_with_BMS_unique_per_cnit_colored_deduplicated.xlsx"
sales_files_recent = [
    "ventes 2020_I4CE.xlsx",
    "ventes 2021_IPP.xlsx",
    "ventes 2023_IPP.xlsx"
]


df_vehicules = pd.read_excel(file_vehicules)

def charger_carrosseries(files):
    dfs = []
    for file in files:
        try:
            df = pd.read_excel(file, usecols=["Type", "Carrosserie"])
            dfs.append(df.dropna(subset=["Type", "Carrosserie"]).drop_duplicates())
        except:
            continue
    return pd.concat(dfs).drop_duplicates()

df_carrosseries = charger_carrosseries(sales_files_recent)

df_green = df_vehicules[df_vehicules['highlight'] == 'green'].copy()
df_green_carrosserie = pd.merge(df_green, df_carrosseries, left_on="cnit", right_on="Type", how="left")
df_green_carrosserie.to_excel("Vehicules_green_carrosseries_2020_2023.xlsx", index=False)
