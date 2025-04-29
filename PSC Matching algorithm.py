

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

df = pd.read_excel("battery_merge_preliminary (1).xlsx")

# On conserve les lignes avec cnit_count = 1 et une capacité batterie renseignée
df_model_data = df[(df['cnit_count'] == 1) & (df['battery_capacity_kwh'].notna())].copy()

# Création d'un identifiant unique pour le type de batterie
df_model_data['battery_label'] = (
    df_model_data['battery_capacity_kwh'].astype(str) + '_' +
    df_model_data['battery_cell_type'].fillna('NA') + '_' +
    df_model_data['battery_cell_cathode_material'].fillna('NA')
)

valid_labels = df_model_data['battery_label'].value_counts()
df_model_data = df_model_data[df_model_data['battery_label'].isin(valid_labels[valid_labels >= 2].index)]

numerical_features = ['weight', 'price', 'component_volume']
categorical_features = ['Brand', 'Model', 'battery_cell_type', 'battery_cell_cathode_material', 'country_cell']

# Suppression des lignes incomplètes
df_model_data = df_model_data.dropna(subset=numerical_features + categorical_features)

label_encoder = LabelEncoder()
df_model_data['battery_encoded'] = label_encoder.fit_transform(df_model_data['battery_label'])

# Séparation des données en train / test
X = df_model_data[numerical_features + categorical_features]
y = df_model_data['battery_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

preprocessor = ColumnTransformer([
    ('num', 'passthrough', numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Entraînement du modèle
pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens')
plt.title("Matrice de confusion - Modèle enrichi")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.tight_layout()
plt.savefig("matrice_confusion.png")
plt.close()

# Application du modèle aux CNITs ambigus

df_ambiguous = df[df['cnit_count'] > 1].copy()


df_ambiguous_valid = df_ambiguous.dropna(subset=numerical_features + categorical_features).copy()
X_ambiguous = df_ambiguous_valid[numerical_features + categorical_features]
df_ambiguous_valid['battery_pred_encoded'] = pipeline.predict(X_ambiguous)
df_ambiguous_valid['battery_pred_label'] = label_encoder.inverse_transform(df_ambiguous_valid['battery_pred_encoded'])
df_ambiguous_valid['battery_prediction_confidence'] = pipeline.predict_proba(X_ambiguous).max(axis=1)
df_ambiguous_invalid = df_ambiguous[df_ambiguous[numerical_features + categorical_features].isna().any(axis=1)].copy()
df_ambiguous_invalid['battery_pred_label'] = None
df_ambiguous_invalid['battery_prediction_confidence'] = None
df_ambiguous_full = pd.concat([df_ambiguous_valid, df_ambiguous_invalid], axis=0).sort_index()

highlight_status = []
for cnit, group in df_ambiguous_full.groupby('cnit'):
    valid_group = group.dropna(subset=['battery_prediction_confidence'])
    if valid_group.empty or valid_group['battery_prediction_confidence'].max() <= 0.3:
        highlight_status.extend(['red'] * len(group))
    else:
        best_idx = valid_group['battery_prediction_confidence'].idxmax()
        for idx in group.index:
            if idx == best_idx:
                highlight_status.append('green')
            else:
                highlight_status.append('')
df_ambiguous_full['highlight'] = highlight_status

df_ambiguous_full.to_excel("resultats_final_modele_enrichi.xlsx", index=False)
