# Prévision de Série Temporelle avec ARIMA

## Description

Ce projet permet de prédire la température moyenne à partir d'une série temporelle historique en utilisant un modèle ARIMA. Il fournit des estimations pour des dates spécifiques dans la plage des données existantes, tout en affichant un intervalle de confiance pour chaque prédiction.

Le projet est concentré sur les températures de **Londres**

## Fonctionnalités

- **Visualisation des données** : Affichage des séries temporelles avec `matplotlib`.
- **Décomposition saisonnière** : Analyse des tendances, des saisons et du bruit dans les données.
- **Test de stationnarité (ADF)** : Vérification de la stationnarité des données à l'aide du test d'Augmented Dickey-Fuller.
- **Auto ARIMA** : Choix automatique des meilleurs paramètres ARIMA à l'aide de `pmdarima`.
- **Prédiction pour une date spécifique** : Estimation de la température pour une date déjà présente dans les données.
- **Intervalle de confiance** : Affichage de l'intervalle de confiance pour chaque prédiction.

## Installation

1. **Cloner le projet** :

```bash
git clone https://github.com/ton-utilisateur/ton-projet.git
cd ton-projet
```

2. **Crée un environnement virtuel et installe les dépendances nécessaires avec `pip`** :

```bash
python -m venv venv
source venv/bin/activate  # Sur Linux/macOS
venv\Scripts\activate     # Sur Windows
pip install -r requirements.txt
```

3. **Lance l'application Streamlit avec la commande suivante** :

```bash
streamlit run app.py
```

Cela ouvrira l'application dans ton navigateur à l'adresse [http://localhost:8501](http://localhost:8501).

### Structure des fichiers
.
├── app.py               # Code principal de l'application
├── requirements.txt     # Liste des dépendances Python
├── data/                # Dossier contenant les dossiers de données (raw et processed)
├── notebooks/           # Dossier contenant les fichiers de tests bac à sable
└── README.md            # Ce fichier


## Auteurs

- Seb
- Elliot
- Yann

## Remerciements

Merci à toutes les personnes ayant contribué à ce projet !
