# %% [markdown]
"""
# TP3 : Analyse de l'Impact Environnemental dans l'Industrie Sidérurgique

## Objectifs
1. Analyser la relation entre consommation énergétique et émissions de CO2
2. Identifier les profils d'impact environnemental
3. Proposer des stratégies de réduction des émissions

## Structure
1. Préparation des données
2. Analyse des émissions de CO2
3. Clustering des profils environnementaux
4. Recommandations d'optimisation
"""

# %% [code]
# Imports et configuration
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
import os
import urllib.request
import zipfile
warnings.filterwarnings('ignore')

# Style des graphiques
plt.style.use('default')
sns.set_theme()

# %% [code]
# Téléchargement et préparation des données
if not os.path.exists('Steel_industry_data.csv'):
    print("Téléchargement des données...")
    # Télécharger le fichier zip
    !wget -O steel_industry_data.zip https://archive.ics.uci.edu/static/public/851/steel+industry+energy+consumption.zip
    # Décompresser le fichier
    !unzip -o steel_industry_data.zip
    print("Données téléchargées et décompressées.")
else:
    print("Fichier de données déjà présent.")

# Chargement des données
try:
    df = pd.read_csv('Steel_industry_data.csv')
    print(f"Données chargées avec succès : {df.shape[0]} observations, {df.shape[1]} variables")
except Exception as e:
    print(f"Erreur lors du chargement des données : {e}")
    raise

# Séparation des variables
numeric_features = [
    'Lagging_Current_Reactive.Power_kVarh',
    'Leading_Current_Reactive_Power_kVarh',
    'CO2(tCO2)',
    'Lagging_Current_Power_Factor',
    'Leading_Current_Power_Factor',
    'NSM'
]
categorical_features = ['Day_of_week', 'WeekStatus']

# Création des périodes de la journée industrielle
def create_industrial_periods(df):
    # Conversion NSM en heures
    df['hour'] = df['NSM'] / 3600
    
    # Création des périodes avec la journée commençant à 6h
    conditions = [
        (df['hour'] >= 6) & (df['hour'] < 10),   # Matin1
        (df['hour'] >= 10) & (df['hour'] < 14),  # Matin2
        (df['hour'] >= 14) & (df['hour'] < 18),  # Aprem1
        (df['hour'] >= 18) & (df['hour'] < 22),  # Aprem2
        (df['hour'] >= 22) | (df['hour'] < 2),   # Nuit1
        (df['hour'] >= 2) & (df['hour'] < 6)     # Nuit2
    ]
    
    periods = ['Matin1', 'Matin2', 'Aprem1', 'Aprem2', 'Nuit1', 'Nuit2']
    df['period'] = np.select(conditions, periods, default='Nuit2')
    return df

# Application des périodes
df = create_industrial_periods(df)

# Calcul de l'intensité carbone en évitant les divisions par zéro
df['carbon_intensity'] = df['CO2(tCO2)'] / df['Usage_kWh'].replace(0, np.nan)
df['carbon_intensity'] = df['carbon_intensity'].replace([np.inf, -np.inf], np.nan)

# Remplacement des valeurs infinies ou NaN par la médiane
median_intensity = df['carbon_intensity'].median()
df['carbon_intensity'] = df['carbon_intensity'].fillna(median_intensity)

# %% [markdown]
"""
## 1. Analyse des Émissions de CO2

Examinons les patterns d'émissions et leur relation avec la consommation énergétique.
"""

# %% [code]
# Visualisation des émissions
plt.figure(figsize=(15, 5))

# Distribution temporelle des émissions
plt.subplot(1, 2, 1)
sns.boxplot(data=df, x='period', y='CO2(tCO2)')
plt.title('Émissions de CO2 par Période')
plt.ylabel('Émissions (tCO2)')

# Relation consommation-émissions
plt.subplot(1, 2, 2)
sns.scatterplot(data=df, x='Usage_kWh', y='CO2(tCO2)', 
                hue='period', alpha=0.6)
plt.title('Relation Consommation-Émissions')
plt.xlabel('Consommation (kWh)')
plt.ylabel('Émissions (tCO2)')

plt.tight_layout()
plt.show()

# %% [code]
# Sélection des features pertinentes pour l'analyse environnementale
features = [
    'Usage_kWh',
    'CO2(tCO2)',
    'Lagging_Current_Reactive.Power_kVarh',
    'Leading_Current_Reactive_Power_kVarh',
    'Lagging_Current_Power_Factor',
    'Leading_Current_Power_Factor'
]

# Standardisation
scaler = StandardScaler()
X = scaler.fit_transform(df[features])

# ACP
pca = PCA()
X_pca = pca.fit_transform(X)

# Affichage de la variance expliquée
explained_variance_ratio = pca.explained_variance_ratio_
cumsum_variance_ratio = np.cumsum(explained_variance_ratio)

print("Variance expliquée par composante :")
for i, var in enumerate(explained_variance_ratio):
    print(f"PC{i+1}: {var:.3f} ({cumsum_variance_ratio[i]:.3f} cumulé)")

# Sélection des 2 premières composantes pour le clustering
X_pca_2d = X_pca[:, :2]

# K-means sur les composantes principales
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_pca_2d)

# Visualisation
plt.figure(figsize=(12, 8))

# Création de la grille pour les frontières
x_min, x_max = X_pca_2d[:, 0].min() - 1, X_pca_2d[:, 0].max() + 1
y_min, y_max = X_pca_2d[:, 1].min() - 1, X_pca_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Frontières des clusters
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='black', linestyles='--', alpha=0.5)

# Scatter plot
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
                     c=df['Usage_kWh'],  # Coloration selon la consommation
                     cmap='viridis',
                     alpha=0.6)

# Annotations des clusters
for i in range(3):
    cluster_data = df[df['Cluster'] == i]
    centroid = X_pca_2d[df['Cluster'] == i].mean(axis=0)
    
    # Caractéristiques du cluster
    usage_mean = cluster_data['Usage_kWh'].mean()
    co2_mean = cluster_data['CO2(tCO2)'].mean()
    pf_mean = cluster_data['Lagging_Current_Power_Factor'].mean()
    
    # Détermination du type de profil
    if usage_mean > df['Usage_kWh'].quantile(0.66):
        profile = "Forte charge"
    elif usage_mean > df['Usage_kWh'].quantile(0.33):
        profile = "Charge moyenne"
    else:
        profile = "Faible charge"
    
    plt.annotate(
        f'Cluster {i}\n'
        f'Usage: {usage_mean:.1f} kWh\n'
        f'CO2: {co2_mean:.3f} tCO2\n'
        f'PF: {pf_mean:.2f}\n'
        f'Type: {profile}',
        xy=(centroid[0], centroid[1]),
        xytext=(10, 10),
        textcoords='offset points',
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8)
    )

plt.title('Profils de Consommation (après ACP)')
plt.xlabel('Première composante principale')
plt.ylabel('Deuxième composante principale')
plt.colorbar(scatter, label='Consommation (kWh)')
plt.grid(True)
plt.show()

# Analyse des clusters
print("\nAnalyse détaillée des clusters :")
for i in range(3):
    cluster_data = df[df['Cluster'] == i]
    print(f"\nCluster {i}:")
    print(f"Nombre de points: {len(cluster_data)}")
    print(f"Consommation moyenne: {cluster_data['Usage_kWh'].mean():.1f} kWh")
    print(f"Émissions moyennes: {cluster_data['CO2(tCO2)'].mean():.3f} tCO2")
    print(f"Facteur de puissance moyen: {cluster_data['Lagging_Current_Power_Factor'].mean():.2f}")
    print(f"Période dominante: {cluster_data['period'].value_counts().index[0]}")

# %% [markdown]
"""
❓ Questions d'Analyse :

1. **Patterns d'Émissions**
   - Quelles périodes montrent les émissions les plus élevées ?
   - Comment expliquer les variations d'intensité carbone ?
   - Quels facteurs influencent le plus les émissions ?

2. **Profils Environnementaux**
   - Quelles sont les caractéristiques de chaque cluster ?
   - Pourquoi certains process sont-ils plus intensifs en carbone ?
   - Comment optimiser les process les plus émetteurs ?

3. **Pistes d'Amélioration**
   - Quelles actions concrètes pour réduire l'empreinte carbone ?
   - Comment prioriser les interventions ?
   - Quels objectifs de réduction sont réalistes ?
"""

# %% [code]
# Analyse détaillée par cluster
print("Analyse des Clusters :")
for i in range(3):
    cluster_data = df[df['Cluster'] == i]
    print(f"\nCluster {i}:")
    print(f"Nombre de points: {len(cluster_data)}")
    print(f"Émissions moyennes: {cluster_data['CO2(tCO2)'].mean():.3f} tCO2")
    print(f"Intensité carbone: {cluster_data['carbon_intensity'].mean():.3f} tCO2/kWh")
    print(f"Périodes dominantes: {cluster_data['period'].value_counts().nlargest(1).index[0]}")
    
    # Potentiel de réduction
    reduction_potential = (cluster_data['carbon_intensity'].mean() - 
                         df['carbon_intensity'].quantile(0.25)) * cluster_data['Usage_kWh'].sum()
    print(f"Potentiel de réduction: {reduction_potential:.1f} tCO2")

# %% [markdown]
"""
## Analyse des Clusters et Recommandations

### 1. Identification des Profils
"""

# %% [code]
# Visualisation principale des clusters avec leurs caractéristiques
plt.figure(figsize=(12, 8))
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
           c=df['Cluster'], cmap='viridis',
           alpha=0.6)
plt.contour(xx, yy, Z, colors='black', linestyles='--', alpha=0.5)

for i in range(3):
    cluster_data = df[df['Cluster'] == i]
    centroid = X_pca_2d[df['Cluster'] == i].mean(axis=0)
    plt.annotate(f'Cluster {i}', xy=(centroid[0], centroid[1]))

plt.title('Vue Globale des Clusters')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(label='Cluster')
plt.show()

# %% [markdown]
"""
❓ Questions sur la séparation des clusters :
- Pourquoi observe-t-on une séparation si nette entre les clusters ?
- Que représente la distance entre les points d'un même cluster ?
- Comment expliquer la forme allongée du cluster 2 ?
"""

# %% [markdown]
"""
### 2. Analyse du Cluster 1 - Problème d'Efficience Nocturne
"""

# %% [code]
# Visualisation de l'efficience par période
cluster1_data = df[df['Cluster'] == 1]

plt.figure(figsize=(15, 5))

# Distribution du facteur de puissance par période
plt.subplot(1, 2, 1)
sns.boxplot(data=cluster1_data, x='period', 
            y='Lagging_Current_Power_Factor')
plt.title('Facteur de Puissance - Cluster 1')
plt.xticks(rotation=45)

# Évolution temporelle
plt.subplot(1, 2, 2)
sns.scatterplot(data=cluster1_data, 
                x='hour', 
                y='Lagging_Current_Power_Factor',
                alpha=0.5)
plt.axhline(y=90, color='r', linestyle='--', 
            label='Objectif PF > 90')
plt.title('Variation du PF sur 24h')
plt.legend()
plt.tight_layout()
plt.show()

print("Statistiques Cluster 1 par période:")
print(cluster1_data.groupby('period')['Lagging_Current_Power_Factor'].describe())

# %% [markdown]
"""
❓ Questions sur l'inefficience nocturne :
- Pourquoi le facteur de puissance est-il plus faible la nuit ?
- Quels équipements sont probablement en cause ?
- Quel serait l'impact financier d'une amélioration du PF nocturne ?
"""

# %% [markdown]
"""
**Recommandations Cluster 1:**
- Installation de batteries de condensateurs : Justifiée par le faible PF nocturne
- Formation des équipes de nuit : Nécessaire vu la variation importante du PF
"""

# %% [markdown]
"""
### 3. Analyse du Cluster 0 - Gestion des Pics de Charge
"""

# %% [code]
# Analyse des pics de consommation
cluster0_data = df[df['Cluster'] == 0]

plt.figure(figsize=(15, 5))

# Distribution horaire de la charge
plt.subplot(1, 2, 1)
sns.histplot(data=cluster0_data, x='hour', 
             weights='Usage_kWh', bins=24)
plt.title('Distribution Horaire de la Charge')

# Relation charge-émissions
plt.subplot(1, 2, 2)
sns.scatterplot(data=cluster0_data,
                x='Usage_kWh', 
                y='CO2(tCO2)',
                hue='period')
plt.title('Impact Environnemental des Pics')
plt.tight_layout()
plt.show()

# Analyse des périodes de pic
peak_periods = cluster0_data.groupby('period')['Usage_kWh'].agg(['mean', 'count'])
print("\nAnalyse des pics par période:")
print(peak_periods.sort_values('mean', ascending=False))

# %% [markdown]
"""
❓ Questions sur les pics de charge :
- Pourquoi les pics sont-ils concentrés sur certaines périodes ?
- Quel est le compromis entre lissage de charge et contraintes de production ?
- Comment calculer le coût des pics de consommation ?
"""

# %% [markdown]
"""
**Recommandations Cluster 0:**
- Lissage de charge : Nécessaire vu la concentration des pics
- Système d'alerte : Pour gérer les dépassements identifiés
"""

# %% [markdown]
"""
### 4. Analyse du Cluster 2 - Modèle d'Efficience
"""

# %% [code]
# Comparaison des conditions opérationnelles
cluster2_data = df[df['Cluster'] == 2]

plt.figure(figsize=(15, 5))

# Comparaison des facteurs de puissance
plt.subplot(1, 2, 1)
sns.boxplot(data=df, x='Cluster', 
            y='Lagging_Current_Power_Factor')
plt.title('Comparaison des PF par Cluster')

# Analyse des conditions optimales
plt.subplot(1, 2, 2)
sns.scatterplot(data=cluster2_data,
                x='Usage_kWh',
                y='Lagging_Current_Power_Factor',
                hue='period')
plt.title('Conditions Optimales (Cluster 2)')
plt.tight_layout()
plt.show()

# Identification des meilleures pratiques
best_conditions = cluster2_data[cluster2_data['Lagging_Current_Power_Factor'] > 95]
print("\nConditions optimales d'opération:")
print(best_conditions.groupby('period').agg({
    'Usage_kWh': 'mean',
    'Lagging_Current_Power_Factor': 'mean'
}).round(2))

# %% [markdown]
"""
❓ Questions sur les bonnes pratiques :
- Quelles conditions spécifiques permettent d'atteindre un PF > 95 ?
- Comment généraliser ces conditions aux autres périodes ?
- Quel serait le coût de mise en œuvre de ces améliorations ?
"""

# %% [markdown]
"""
**Recommandations Cluster 2:**
- Documentation des procédures : Basée sur les conditions optimales identifiées
- Automatisation : Pour reproduire les séquences efficientes observées
"""

# %% [markdown]
"""
### 5. Plan de Mise en Œuvre et Suivi
"""

# %% [code]
# Création d'un dashboard de suivi
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Distribution des clusters
sns.scatterplot(data=df, x=X_pca_2d[:, 0], y=X_pca_2d[:, 1],
                hue='Cluster', ax=axes[0, 0])
axes[0, 0].set_title('Distribution des Clusters')

# 2. Évolution du PF
sns.boxplot(data=df, x='period', y='Lagging_Current_Power_Factor',
            hue='Cluster', ax=axes[0, 1])
axes[0, 1].set_title('PF par Période et Cluster')
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. Distribution de la charge
sns.histplot(data=df, x='Usage_kWh', hue='Cluster',
             multiple="stack", ax=axes[1, 0])
axes[1, 0].set_title('Distribution de la Charge')

# 4. Émissions par cluster
sns.boxplot(data=df, x='Cluster', y='CO2(tCO2)', ax=axes[1, 1])
axes[1, 1].set_title('Émissions par Cluster')

plt.tight_layout()
plt.show()

# KPIs actuels
print("\nKPIs de Base:")
for cluster in range(3):
    cluster_data = df[df['Cluster'] == cluster]
    print(f"\nCluster {cluster}:")
    print(f"PF moyen: {cluster_data['Lagging_Current_Power_Factor'].mean():.2f}")
    print(f"% points PF > 90: {(cluster_data['Lagging_Current_Power_Factor'] > 90).mean()*100:.1f}%")
    print(f"Émissions moyennes: {cluster_data['CO2(tCO2)'].mean():.3f} tCO2")

# %% [markdown]
"""
❓ Questions sur le suivi :
- Quels KPIs additionnels seraient pertinents ?
- À quelle fréquence faut-il mettre à jour cette analyse ?
- Comment détecter une dérive des performances ?
"""

# %% [markdown]
"""
## Recommandations pour la Réduction des Émissions

1. **Optimisation Opérationnelle**
   - Identifier et reproduire les conditions des périodes à faible intensité
   - Optimiser la planification des opérations énergivores
   - Former les opérateurs aux meilleures pratiques environnementales

2. **Améliorations Techniques**
   - Moderniser les équipements les plus émetteurs
   - Installer des systèmes de monitoring des émissions en temps réel
   - Mettre en place des systèmes de récupération d'énergie

3. **Stratégie Long Terme**
   - Définir des objectifs de réduction par cluster
   - Investir dans des technologies bas-carbone
   - Développer un plan de transition énergétique
"""

# %% [markdown]
"""
❓ Questions sur la mise en œuvre :
- Par où commencer concrètement ?
- Quel retour sur investissement attendre de chaque action ?
- Comment impliquer les équipes dans ces changements ?
""" 