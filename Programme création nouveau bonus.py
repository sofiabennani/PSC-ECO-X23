
import pandas as pd
import os

documents_path = os.path.expanduser("~/Documents")

file_co2 = os.path.join(documents_path, "CO2_véhicule.xlsx")
file_ademe = os.path.join(documents_path, "01-01-ADEME_full.xlsx")


df_co2 = pd.read_excel(file_co2)
df_ademe = pd.read_excel(file_ademe, usecols=["cnit", "DATE", "Marque", "Modele", "Energie","BONUS_MALUS", "BONUS_MALUS_BAREME", "Carrosserie", "VOLUME", "Energie"])

df_co2 = df_co2.rename(columns={"price": "Prix_vehic"})
df_ademe = df_ademe.rename(columns={"DATE": "date"})
df_co2_filtered = df_co2[["cnit", "Prix_vehic", "CO2_vehicule"]]
df_final = pd.merge(df_ademe, df_co2_filtered, on="cnit", how="inner")
output_path = os.path.join(documents_path, "fichier_final_bonus_CO2.xlsx")
df_final.to_excel(output_path, index=False)


print("réussi")
