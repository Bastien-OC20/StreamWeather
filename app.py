import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os

def load_data(uploaded_file):
    # Chargement des données
    data = pd.read_csv(uploaded_file)
    data['date'] = pd.to_datetime(data['date'].astype(str), format='%Y%m%d')
    data = data.drop(columns=['sunshine', 'cloud_cover', 'global_radiation', 'max_temp', 'min_temp', 'precipitation', 'pressure', 'snow_depth'])
    data.set_index('date', inplace=True)

    # Re-sample des données sur une base hebdomadaire
    data_weekly = data.resample('W').mean()

    # Gérer les valeurs manquantes
    data_weekly = data_weekly.replace([np.inf, -np.inf], np.nan)
    data_weekly['mean_temp'] = data_weekly['mean_temp'].fillna(data_weekly['mean_temp'].mean())  # Imputation par la moyenne

    # Transformation de la température (ajout de 10 pour rendre les valeurs positives)
    data_weekly['mean_temp'] = data_weekly['mean_temp'] + 10

    # Vérifier si le répertoire existe, sinon le créer
    output_dir = '../data/processed'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Sauvegarde les données hebdomadaires prétraitées en CSV
    data_weekly.to_csv(f'{output_dir}/data_weekly.csv')

    return data_weekly

# Fonction pour afficher les prévisions avec intervalle de confiance
def plot_forecast(fc, train, test, upper=None, lower=None):
    is_confidence_int = isinstance(upper, np.ndarray) and isinstance(lower, np.ndarray)
    
    fc_series = pd.Series(fc, index=test.index)
    lower_series = pd.Series(upper, index=test.index) if is_confidence_int else None
    upper_series = pd.Series(lower, index=test.index) if is_confidence_int else None

    # Plot
    fig, ax = plt.subplots(figsize=(10,4), dpi=100)
    ax.plot(train, label='training', color='b')
    ax.plot(test, label='actual', color='b', ls='--')
    ax.plot(fc_series, label='forecast', color='orange')
    if is_confidence_int:
        ax.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=0.15)
    ax.set_title('Forecast vs Actuals')
    ax.legend(loc='upper left', fontsize=8)
    st.pyplot(fig)  # Pass the figure explicitly

# Configuration de Streamlit
st.title('Prévision de Série Temporelle avec ARIMA')
st.write("Ce projet permet de prédire la température et d'afficher les intervalles de confiance pour une série temporelle.")

# Chargement du fichier CSV
uploaded_file = st.file_uploader("Téléchargez votre fichier de données", type=["csv"])

if uploaded_file is not None:
    # Charger et prétraiter les données
    data = load_data(uploaded_file)
    st.write(f"Nombre d'observations: {len(data)}")
    st.write(data.head())

    # Visualisation de la série temporelle
    st.subheader("Visualisation des Données")
    fig, ax = plt.subplots(figsize=(16,4))
    ax.plot(data['mean_temp'])
    ax.set_title('Température moyenne')
    st.pyplot(fig)  # Pass the figure explicitly

    # Decomposition saisonnière (Méthode Multiplicative)
    st.subheader("Décomposition Saisonnière (Multiplicative)")
    decompositionM = seasonal_decompose(data['mean_temp'], model='multiplicative', period=52)
    fig = decompositionM.plot()
    st.pyplot(fig)  # Pass the figure explicitly

    # Test ADF pour la stationnarité
    st.subheader("Test de Stationnarité (ADF)")
    adf_stat, p_value, _, _, _, _ = adfuller(data['mean_temp'])
    st.write(f"p-value du test ADF : {p_value}")
    if p_value < 0.05:
        st.write("La série est stationnaire.")
    else:
        st.write("La série n'est pas stationnaire.")

    # ACF/PACF
    st.subheader("ACF et PACF")
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    plot_acf(data['mean_temp'], ax=ax[0])
    plot_pacf(data['mean_temp'], ax=ax[1])
    st.pyplot(fig)  # Pass the figure explicitly

    # Choisir le meilleur modèle avec auto_arima
    st.subheader("Choix automatique du modèle ARIMA avec auto_arima")
    model_auto = auto_arima(data['mean_temp'], seasonal=True, trend='t', stepwise=True, trace=True)
    st.write("Meilleurs paramètres ARIMA trouvés :")
    st.write(model_auto.summary())

    # Ajustement du modèle ARIMA avec les paramètres optimaux de auto_arima
    st.subheader("Ajustement du Modèle ARIMA avec auto_arima")
    p, d, q = model_auto.order  # Les meilleurs paramètres p, d, q de auto_arima
    model = ARIMA(data['mean_temp'], order=(p, d, q))
    model_fit = model.fit()

    # Train le modele en fonction des meilleurs paramètres
    train = data[1:520]
    test = data[520:624]

    model = SARIMAX(train, order=(3, 0, 2), seasonal_order=(1, 1, 1, 52))
    model_fit = model.fit()

    # Visualisation des résidus pour déterminer si le modèle est robuste
    st.subheader("Résidus > Robustesse modèle")
    residuals = model_fit.resid
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    plot_acf(residuals, ax=ax[0])
    plot_pacf(residuals, ax=ax[1])
    st.pyplot(fig)  # Pass the figure explicitly

    # Vérification de la stationnarité des résidus avec ADF
    st.subheader("Stationnarité des résidus")
    result_resid = adfuller(residuals)[1]
    st.write('p-value des résidus:', result_resid)

    # Affichage des prévisions
    st.subheader("Prévisions")
    forecast = model_fit.get_forecast(steps=100)
    forecast_mean = forecast.predicted_mean - 10
    forecast_conf_int = forecast.conf_int() - 10

    # Affichage des prévisions avec intervalle de confiance
    fig, ax = plt.subplots(figsize=(10,4), dpi=100)
    ax.plot(data[520:624] - 10, label="Données observées")
    ax.plot(forecast_mean, label="Prédictions")
    ax.fill_between(forecast_conf_int.index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='grey', alpha=0.3)
    ax.legend()
    ax.set_title('Prévisions avec Intervalle de Confiance')
    st.pyplot(fig)  # Pass the figure explicitly

    # Vérification de la première et dernière date des données
    min_date = data.index.min().date()
    max_date = data.index.max().date()

    # Afficher les dates minimales et maximales
    st.write(f"Date minimale disponible: {min_date}")
    st.write(f"Date maximale disponible: {max_date}")

    # Section d'estimation selon une date donnée
    st.subheader("Estimation de la Température pour une Date Spécifique")

    # Demande de la date pour estimation (ajustée à la granularité hebdomadaire)
    date_input = st.date_input(
        "Sélectionnez une date", 
        min_value=min_date, 
        max_value=max_date,
        value=max_date  # Par défaut, sélectionner la dernière date disponible
    )

    # Vérification de la date sélectionnée pour qu'elle soit compatible avec les données hebdomadaires
    week_start = pd.to_datetime(date_input) - pd.to_timedelta(pd.to_datetime(date_input).weekday(), unit='D')

    # Faire la prédiction pour la date sélectionnée avec la méthode 'get_prediction'
    prediction = model_fit.get_prediction(start=week_start, end=week_start)
    predicted_temp = prediction.predicted_mean.iloc[0] - 10
    st.write(f"Température estimée pour la semaine débutant le {week_start.date()}: {predicted_temp:.2f}°C")
