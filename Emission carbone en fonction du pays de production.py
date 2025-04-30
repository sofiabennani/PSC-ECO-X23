import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

## Importation des fichiers

file1_path = "C:/Users/reyna_rmyk/Documents/SCHOOL/Polytechnique/2A/PSC/Bases de données/Facteur d'émission des pays en 2011.xlsx"

file2_path = "C:/Users/reyna_rmyk/Documents/SCHOOL/Polytechnique/2A/PSC/Bases de données/Lieu_production_batterie.xlsx"

file3_path = "C:/Users/reyna_rmyk/Documents/SCHOOL/Polytechnique/2A/PSC/Bases de données/Types_batteries.xlsx"

file4_path = "C:/Users/reyna_rmyk/Documents/SCHOOL/Polytechnique/2A/PSC/Bases de données/Lieu_assemblage.xlsx"

file5_path = "C:/Users/reyna_rmyk/Documents/SCHOOL/Polytechnique/2A/PSC/Bases de données/Matching.xlsx"

data_factem = pd.read_excel(file1_path,usecols=['country','facteur_emission'])
data_prodbat = pd.read_excel(file2_path)
data_battery_type = pd.read_excel(file3_path,usecols=['battery_cell_cathode_type','energy_cathode_kwh/kwh','energy_cathode_density _kwh/kg'])
data_assemblage = pd.read_excel(file4_path)
data_matching = pd.read_excel(file5_path)


## I - Première approximation : toutes les batteries sont les mêmes. On ne regarde que la puissance.

# Création d'un nouveau tableau en filtrant les véhicules électriques
data_prodbatelec = data_prodbat[data_prodbat["engine_type"].str.contains("Electric", case=False, na=False)]

# Fusionner les deux BDD en spécifiant les colonnes respectives
data = pd.merge(data_prodbatelec,data_factem, how="left", left_on="plant_cell_country",right_on="country")

# Suppression des données vides
data = data[data["plant_cell_country"] != "Not Researched"]
data = data[data["plant_cell_country"] != "Unknown"]


# Calcul des émissions
capacite = data["battery_capacity_kwh"]
pays = data["plant_cell_country"]
factem = data["facteur_emission"]

data["Emission"] = 0.328*capacite*factem
emiss = data["Emission"]
emiss = emiss/max(emiss)

# Différenciation des pays

liste_pays = ["China","Poland","United Kingdom","United States","Canada","France","Germany","Hungary","Japan","Malaysia","South Korea"]

emis_Chine = emiss[pays.str.contains(r'^'+liste_pays[0]+r'$')]
capa_Chine = capacite[pays.str.contains(r'^'+liste_pays[0]+r'$')]

emis_Polo = emiss[pays.str.contains(r'^'+liste_pays[1]+r'$')]
capa_Polo = capacite[pays.str.contains(r'^'+liste_pays[1]+r'$')]

emis_RU = emiss[pays.str.contains(r'^'+liste_pays[2]+r'$')]
capa_RU = capacite[pays.str.contains(r'^'+liste_pays[2]+r'$')]

emis_US = emiss[pays.str.contains(r'^'+liste_pays[3]+r'$')]
capa_US = capacite[pays.str.contains(r'^'+liste_pays[3]+r'$')]

emis_Canada = emiss[pays.str.contains(r'^'+liste_pays[4]+r'$')]
capa_Canada = capacite[pays.str.contains(r'^'+liste_pays[4]+r'$')]

emis_France = emiss[pays.str.contains(r'^'+liste_pays[5]+r'$')]
capa_France = capacite[pays.str.contains(r'^'+liste_pays[5]+r'$')]

emis_All = emiss[pays.str.contains(r'^'+liste_pays[6]+r'$')]
capa_All = capacite[pays.str.contains(r'^'+liste_pays[6]+r'$')]

emis_Hong = emiss[pays.str.contains(r'^'+liste_pays[7]+r'$')]
capa_Hong = capacite[pays.str.contains(r'^'+liste_pays[7]+r'$')]

emis_Japon = emiss[pays.str.contains(r'^'+liste_pays[8]+r'$')]
capa_Japon = capacite[pays.str.contains(r'^'+liste_pays[8]+r'$')]

emis_Mal = emiss[pays.str.contains(r'^'+liste_pays[9]+r'$')]
capa_Mal = capacite[pays.str.contains(r'^'+liste_pays[9]+r'$')]

emis_Coree = emiss[pays.str.contains(r'^'+liste_pays[10]+r'$')]
capa_Coree = capacite[pays.str.contains(r'^'+liste_pays[10]+r'$')]


# Tracé du graphique

plt.figure(figsize=(10,6))
plt.scatter(capa_Chine,emis_Chine,s=5,label=liste_pays[0])
plt.scatter(capa_Polo,emis_Polo,s=5,label=liste_pays[1])
plt.scatter(capa_RU,emis_RU,s=5,label=liste_pays[2])
plt.scatter(capa_US,emis_US,s=5,label=liste_pays[3])
plt.scatter(capa_Canada,emis_Canada,s=5,label=liste_pays[4])
plt.scatter(capa_France,emis_France,s=5,label=liste_pays[5])
plt.scatter(capa_All,emis_All,s=5,label=liste_pays[6])
plt.scatter(capa_Hong,emis_Hong,s=5,label=liste_pays[7])
plt.scatter(capa_Japon,emis_Japon,s=5,label=liste_pays[8])
plt.scatter(capa_Mal,emis_Mal,s=5,label=liste_pays[9])
plt.scatter(capa_Coree,emis_Coree,s=5,label=liste_pays[10])

plt.title("Taux de CO2 émis par la fabrication de la batterie des véhicules \n par rapport à la capacité de la batterie", fontsize=14)
plt.xlabel("Capacité de la batterie (kWh)",fontsize=12)
plt.ylabel("CO2 émis par la fabrication d'une batterie \n par rapport à la plus polluante",fontsize=12)
plt.legend()
plt.grid(True)
plt.show()


## II - Deuxième approximation tenant compte du type de batterie et de tous les composants

## Fonctions utiles

def nettoyage_bdd(data,column,add_feature): #excel,str,list
    """Retirer les lignes avec des informations manquantes"""
    Vide = ["Not Researched","Unknown",'Not Applicable'] + add_feature
    for i in range(len(Vide)):
        data = data[data[column] != Vide[i]]
    return data


def fusion_ville_pays(data,colonne_ville,colonne_pays): #excel,str,str
    """Ajoute les pays associés à une colonne de villes et nettoie la BDD"""
    fusion = nettoyage_bdd(data,colonne_ville,[])
    corresp_pays_ville = pd.read_excel(file2_path, usecols=[colonne_ville,colonne_pays])
    corresp_pays_ville = corresp_pays_ville.drop_duplicates()
    fusion = pd.merge(fusion, corresp_pays_ville, on=colonne_ville, how='left')
    return fusion


def fusion_factem_pays(data):
    """Ajoute les facteurs d'émission pour chaque composante de la batterie"""
    columns = ["plant_cell_country","plant_module_country","plant_bms_country","plant_pack_country"]
    Name = ["cell","module","bms","pack"]

    fusion = data.copy()

    for i in range(len(columns)):
        fusion = pd.merge(fusion,data_factem,how='inner',left_on=columns[i],right_on="country",suffixes=('',f'_'+Name[i]))

    fusion = fusion.drop(columns=['country_bms', 'country_pack', 'country_module'])
    return fusion

def fusion_type_batterie(data):
    fusion = pd.merge(data, data_battery_type, how="inner", on="battery_cell_cathode_type")
    return fusion

## Calcul des émissions pour la batterie et l'assemblage

def CO2_batterie(data):

    """Calcul le CO2 émis par la batterie
    Appeler fusion_CO2_batterie() avant !"""

    # 1. Energie totale
    Encm = 286*data['battery_capacity_kwh']
    Ex = data['energy_cathode_kwh/kwh']*data['battery_capacity_kwh']

    # 2. Energie par composante

    data['Emodule'] = 0.19*Encm
    data['Ebms'] = 0.04*Encm
    data['Epack'] = 0.15*Encm
    data['Ecell'] = Ex - (data['Emodule']+data['Ebms']+data['Epack'])

    # 3. Multiplication par le facteur d'émission

    data['CO2_battery'] = data['Ecell']*data['facteur_emission_cell'] + data['Emodule']*data['facteur_emission_module'] + data['Ebms']*data['facteur_emission_bms'] +  data['Epack']*data['facteur_emission_pack']

    n=0
    for CO2 in data['CO2_battery']:
        if CO2 < 0 :
            n+=1
    print(n)
    # n=0 => on a bien que des émissions positives, le modèle utilisé est donc validé !
    return data

#CO2_batt = CO2_batterie(data)
#CO2_batt.to_excel(r'C:\Users\reyna_rmyk\Documents\SCHOOL\Polytechnique\2A\PSC\Bases de données\Résultats\CO2_batterie.xlsx', index=False)


def CO2_assemblage(data):
    weight_nobatt = data['weight'] - data['battery_capacity_kwh']/data['energy_cathode_density _kwh/kg']
    fab_energy = 15.7*weight_nobatt
    data['CO2_fabrication'] = data['facteur_emission_assembly']*fab_energy
    return data


## 1. Association du pays d'assemblage

# Définir les colonnes de fusion pour chaque fichier
criteres = ['brand', 'model', 'year']

# Fusionner les deux dataframes sur les colonnes spécifiées
fusion = pd.merge(data_matching, data_assemblage, on=criteres, how='left')

# Ajout des facteurs d'émission
fusion = pd.merge(fusion, data_factem, left_on='country_assembly', right_on='country', how='inner')
#fusion = fusion.drop(columns=['country'])
fusion.rename(columns={'facteur_emission': 'facteur_emission_assembly'}, inplace=True)

# On supprime les colonnes en double
fusion = fusion.drop(columns=['country_y'])


## 2. Association des villes, aux pays, puis aux facteurs d'émission

fusion = nettoyage_bdd(fusion,"battery_cell_cathode_type",["Plomb"])

colonnes_pays_ville = [['plant_pack_country', 'plant_pack_city'],['plant_module_country','plant_module_city'],['plant_bms_country','plant_bms_city']]


# Ajout des pays associés aux villes de fabrication de la batterie
for i in range(len(colonnes_pays_ville)):
    fusion = fusion_ville_pays(fusion,colonnes_pays_ville[i][1],colonnes_pays_ville[i][0])

# Ajout du facteur d'émission
fusion = fusion_factem_pays(fusion)
fusion.rename(columns={'facteur_emission': 'facteur_emission_cell'}, inplace=True)


## 3. Association des informations sur la batterie

# Application des fonctions pour associer à chaque batterie son émission

fusion = fusion_type_batterie(fusion)
fusion = nettoyage_bdd(fusion,"battery_cell_cathode_type",["Plomb"])

## Sauvegarder le résultat
fusion.to_excel(r'C:\Users\reyna_rmyk\Documents\SCHOOL\Polytechnique\2A\PSC\Bases de données\Résultats\Matching_final.xlsx', index=False)


## 4. Calcul final

fusion = CO2_batterie(fusion)
fusion = CO2_assemblage(fusion)

# Calcul de l'émission carbone
fusion['CO2_vehicule'] = fusion['CO2_battery'] + fusion['CO2_fabrication']

# Sauvegarder le résultat
fusion.to_excel(r'C:\Users\reyna_rmyk\Documents\SCHOOL\Polytechnique\2A\PSC\Bases de données\Résultats\CO2_véhicule.xlsx', index=False)


## Extraction des données et statistiques

file6_path = "C:/Users/reyna_rmyk/Documents/SCHOOL/Polytechnique/2A/PSC/Bases de données/Résultats/CO2_véhicule.xlsx"

fusion = pd.read_excel(file6_path)


# 1. Moyennes
colonnes = ['CO2_battery','CO2_fabrication','CO2_vehicule']
stats = fusion[colonnes].describe().T  #transpose pour avoir les colonnes en lignes
stats = stats.round(0)

stats.to_excel(r'C:\Users\reyna_rmyk\Documents\SCHOOL\Polytechnique\2A\PSC\Bases de données\Résultats\stat_resume.xlsx', index=False)


# 2. Moyennes selon la carosserie
colonnes = ['CO2_vehicule']

stats_carrosserie = fusion.groupby('carrosserie')[colonnes].agg(['size', 'min', 'max', 'mean', 'std', 'median'])
stats_carrosserie = stats_carrosserie.round(0)

stats_carrosserie.to_excel(r'C:\Users\reyna_rmyk\Documents\SCHOOL\Polytechnique\2A\PSC\Bases de données\Résultats\stat_carrosserie.xlsx')

## Application au Nested Logit

# Par année

N_2023 = 1132

file7_path = "C:/Users/reyna_rmyk/Documents/SCHOOL/Polytechnique/2A/PSC/Bases de données/Parts_marche_2023.xlsx"
data_marche = pd.read_excel(file7_path)

stats_carrosserie = pd.read_excel(r'C:\Users\reyna_rmyk\Documents\SCHOOL\Polytechnique\2A\PSC\Bases de données\stat_carosserie.xlsx')

fusion = pd.merge(data_marche, stats_carrosserie[['carrosserie','mean']], on="carrosserie", how='left')

fusion['CO2_reel'] = N_2023*fusion['part_marche_reel']*fusion['mean']
fusion['CO2_predit'] = N_2023*fusion['part_marche_nestedlogit']*fusion['mean']

print('CO2 réel émis en 2023 =', fusion['CO2_reel'].sum())
print('CO2 prédit émis en 2023 =', fusion['CO2_predit'].sum())

## Par mois

N_2023 = [415,429,466,444,589,638,479,479,1091,576,580,656]
N_cat = 10

file8_path = "C:/Users/reyna_rmyk/Documents/SCHOOL/Polytechnique/2A/PSC/Bases de données/Parts_marche_2023_mois.xlsx"
data_marche = pd.read_excel(file8_path)

stats_carrosserie = pd.read_excel(r'C:\Users\reyna_rmyk\Documents\SCHOOL\Polytechnique\2A\PSC\Bases de données\stat_carosserie.xlsx')

fusion = pd.merge(data_marche, stats_carrosserie[['carrosserie','mean']], on="carrosserie", how='left')
fusion['CO2_reel'] = 0
fusion['CO2_monte_carlo'] = 0
fusion['CO2_nested'] = 0

for i in range(12):
    for j in range(N_cat):
        fusion['CO2_reel'][10*i+j] = N_2023[i]*fusion['part_reelle'][10*i+j]*fusion['mean'][10*i+j]
        fusion['CO2_monte_carlo'][10*i+j] = N_2023[i]*fusion['part_monte_carlo'][10*i+j]*fusion['mean'][10*i+j]
        fusion['CO2_nested'][10*i+j] = N_2023[i]*fusion['part_nested'][10*i+j]*fusion['mean'][10*i+j]

print('CO2 réel émis en 2023 =', fusion['CO2_reel'].sum())
print('CO2 prédit émis en 2023 (Monte Carlo) =', fusion['CO2_monte_carlo'].sum())
print('CO2 prédit émis en 2023 (Nested) =', fusion['CO2_nested'].sum())

fusion.to_excel(r'C:\Users\reyna_rmyk\Documents\SCHOOL\Polytechnique\2A\PSC\Bases de données\Résultats\emissions_2023_mois.xlsx')













