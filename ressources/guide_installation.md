# Guide d'Installation

## 🎯 Prérequis
- Compte Google (pour Colab)
- Navigateur web 
- Connexion internet stable

## 🚀 Configuration Google Colab

### 1. Accès aux notebooks
1. Cliquer sur les badges "Open in Colab" dans le README
2. Se connecter avec votre compte Google
3. Créer une copie dans votre Drive

### 2. Configuration GPU (Pour TP4)
1. Menu "Runtime" → "Change runtime type"
2. Sélectionner "GPU" dans "Hardware accelerator"
3. Cliquer sur "Save"

## 💻 Option 2 : Installation locale

### Prérequis système
- Python 3.8 ou supérieur
- Git
- CUDA Toolkit (pour GPU) - Recommandé pour TP4

### Installation de l'environnement

#### Avec conda (recommandé)
```bash
# Création de l'environnement
conda create -n tp-siderurgie python=3.8
conda activate tp-siderurgie

# Packages de base (TP1-3)
conda install pandas numpy matplotlib seaborn scikit-learn scipy jupyter

# Packages Deep Learning (TP4)
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
conda install pillow wget
```

#### Avec pip
```bash
# Création de l'environnement virtuel
python -m venv tp-siderurgie
source tp-siderurgie/bin/activate  # Linux/Mac
tp-siderurgie\Scripts\activate     # Windows

# Installation des packages
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
pip install torch torchvision
pip install Pillow wget
```

### Configuration de l'IDE
#### VS Code
1. Installer l'extension Python
2. Installer l'extension Jupyter
3. Sélectionner l'interpréteur Python (tp-siderurgie)

#### PyCharm
1. Ouvrir le projet
2. Configurer l'interpréteur (tp-siderurgie)
3. Installer le plugin Jupyter


## 🔧 Dépannage courant

### Problèmes GPU
1. Vérifier l'activation dans Colab
2. Redémarrer le runtime si nécessaire
3. Libérer la mémoire GPU régulièrement

### Erreurs d'exécution
1. Exécuter les cellules dans l'ordre
2. Redémarrer le runtime si nécessaire

## 📚 Ressources utiles
- [Documentation PyTorch](https://pytorch.org/docs/stable/index.html)
- [Guide CUDA](https://developer.nvidia.com/cuda-toolkit)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)

---
Pour toute question ou problème d'installation, n'hésitez pas à demander de l'aide ! 🆘

## Structure des notebooks

### TP1 : Exploration des données
- Focus : Analyse exploratoire et préparation des données
- Compétences : Pandas, visualisation

### TP2 : Prédiction de la consommation
- Focus : Modèles de régression
- Compétences : Scikit-learn, évaluation de modèles

### TP3 : Analyse par clustering
- Focus : Clustering et réduction de dimensionnalité
- Compétences : K-means, PCA

### TP4 : Détection d'anomalies par Deep Learning
- Focus : Deep Learning et détection d'anomalies
- Compétences : PyTorch, CNN, traitement d'images

## Conseils pratiques

### Exécution du code
1. Exécutez les cellules dans l'ordre (Shift+Enter)
2. Attendez qu'une cellule soit terminée avant d'exécuter la suivante
3. En cas d'erreur, relancez le runtime (Runtime → Redémarrer le runtime)

### Sauvegarde du travail
- Sauvegardez régulièrement (Ctrl+S)
- Exportez vos résultats si nécessaire (graphiques, modèles)

### Résolution des problèmes courants
1. **Erreur de mémoire**
   - Relancez le runtime
   - Réduisez la taille des données si possible

2. **Package manquant**
   - Réexécutez la cellule d'installation des packages
   - Vérifiez la syntaxe d'import

3. **Graphiques ne s'affichent pas**
   - Vérifiez que %matplotlib inline est exécuté
   - Relancez la cellule de visualisation

### Pour le TP4 spécifiquement
1. **Gestion du GPU**
   - Vérifiez l'activation du GPU au début
   - Surveillez l'utilisation mémoire
   - Utilisez `del` et `torch.cuda.empty_cache()` si nécessaire

2. **Traitement des images**
   - Vérifiez les dimensions des tenseurs
   - Surveillez les normalisations
   - Validez les reconstructions visuellement

3. **Entraînement du modèle**
   - Observez la courbe de loss
   - Sauvegardez les meilleurs modèles

## Exercices supplémentaires

### Pour aller plus loin
1. **Exploration des données**
   - Créez de nouvelles visualisations
   - Analysez les patterns temporels
   - Identifiez d'autres corrélations

2. **Modélisation**
   - Testez d'autres algorithmes
   - Optimisez les hyperparamètres
   - Combinez plusieurs modèles

3. **Analyse des résultats**
   - Proposez des recommandations business
   - Estimez les gains potentiels
   - Identifiez les limites des modèles

## Ressources complémentaires

### Documentation
- [Pandas](https://pandas.pydata.org/docs/)
- [Scikit-learn](https://scikit-learn.org/stable/documentation.html)
- [Matplotlib](https://matplotlib.org/stable/contents.html)
- [Seaborn](https://seaborn.pydata.org/tutorial.html)

### Tutoriels recommandés
- [Google Colab Tutorial](https://colab.research.google.com/notebooks/intro.ipynb)
- [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)

## Contenu des TPs

### TP1 : Exploration des données
- Chargement et analyse du dataset
- Visualisation des distributions
- Analyse des corrélations
- Préparation des données

### TP2 : Prédiction de la consommation
- Régression linéaire
- K-Nearest Neighbors
- Arbres de décision
- Random Forest
- Réseaux de neurones

### TP3 : Analyse par clustering
- K-means
- Analyse en Composantes Principales

### TP4 : Détection d'anomalies
- Préparation des données images
- Implémentation d'un autoencoder convolutif
- Entraînement et optimisation du modèle
- Évaluation et visualisation des anomalies

## Utilisation

Chaque TP peut être exécuté dans un environnement Jupyter ou VS Code avec l'extension Python

### 1. Notebooks principaux
- `notebooks/TP1_Exploration_Données.ipynb` : Exploration et préparation des données
- `notebooks/TP2_Prediction_Consommation.ipynb` : Apprentissage supervisé
- `notebooks/TP3_Clustering.ipynb` : Apprentissage non supervisé
- `notebooks/TP4_Detection_Anomalies.ipynb` : Détection d'anomalies par Deep Learning

### Ressources additionnelles pour TP4
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [Anomaly Detection Papers](https://paperswithcode.com/task/anomaly-detection)