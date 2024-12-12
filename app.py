import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import datetime
from io import StringIO

# Fonction pour charger les données CSV
def load_data(uploaded_file):
    # Chargement des données
    data = pd.read_csv(uploaded_file)
    data['date'] = pd.to_datetime(data['date'].astype(str), format='%Y%m%d')
    data = data.drop(columns=['sunshine', 'cloud_cover', 'global_radiation', 'max_temp', 'min_temp', 'precipitation', 'pressure', 'snow_depth'])
    data.set_index('date', inplace=True)
    
    # Gérer les valeurs manquantes
    data = data.replace([np.inf, -np.inf], np.nan)
    data['mean_temp'] = data['mean_temp'].fillna(data['mean_temp'].mean())  # Imputation par la moyenne
    
    # Assurer que la fréquence est définie
    data = data.asfreq('D')  # Assure la fréquence quotidienne des dates
    return data

# Fonction pour afficher les prévisions avec intervalle de confiance
def plot_forecast(fc, train, test, upper=None, lower=None):
    is_confidence_int = isinstance(upper, np.ndarray) and isinstance(lower, np.ndarray)
    fc_series = pd.Series(fc, index=test.index)
    lower_series = pd.Series(upper, index=test.index) if is_confidence_int else None
    upper_series = pd.Series(lower, index=test.index) if is_confidence_int else None

    fig, ax = plt.subplots(figsize=(10,4), dpi=100)
    ax.plot(train, label='training', color='b')
    ax.plot(test, label='actual', color='b', ls='--')
    ax.plot(fc_series, label='forecast', color='orange')
    if is_confidence_int:
        ax.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
    ax.set_title('Forecast vs Actuals')
    ax.legend(loc='upper left', fontsize=8)
    st.pyplot(fig)  # Pass the figure explicitly

# Configuration de Streamlit
st.title('Prévision de Série Temporelle avec ARIMA')
st.write("Ce projet permet de prédire la température et d'afficher les intervalles de confiance pour une série temporelle.")

# Chargement du fichier CSV
uploaded_file = st.file_uploader("Téléchargez votre fichier de données", type=["csv"])

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write(f"Nombre d'observations: {len(data)}")
    st.write(data.head())

    # Visualisation de la série temporelle
    st.subheader("Visualisation des Données")
    fig, ax = plt.subplots(figsize=(16,4))
    ax.plot(data['mean_temp'])
    ax.set_title('Température moyenne')
    st.pyplot(fig)  # Pass the figure explicitly

    # Decomposition saisonnière
    st.subheader("Décomposition Saisonnière")
    decomposition = seasonal_decompose(data['mean_temp'], model='additive', period=3)
    fig = decomposition.plot()
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
    model_auto = auto_arima(data['mean_temp'], seasonal=True, stepwise=True, trace=True)
    st.write("Meilleurs paramètres ARIMA trouvés :")
    st.write(model_auto.summary())

    # Ajustement du modèle ARIMA avec les paramètres optimaux de auto_arima
    st.subheader("Ajustement du Modèle ARIMA avec auto_arima")
    p, d, q = model_auto.order  # Les meilleurs paramètres p, d, q de auto_arima
    model = ARIMA(data['mean_temp'], order=(p, d, q))
    model_fit = model.fit()

    # Diviser les données en training et test
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]

    # Prédictions
    forecast = model_fit.get_forecast(steps=len(test))
    forecast_mean = forecast.predicted_mean
    forecast_conf_int = forecast.conf_int()

    # Affichage des prévisions
    st.subheader("Prévisions")
    plot_forecast(forecast_mean, train, test, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1])
    st.pyplot()  # Pass the figure explicitly

    # Estimation pour une date existante dans les données
    st.subheader("Estimation à une Date Précise")
    
    # Sélectionner une date parmi les dates disponibles dans les données
    available_dates = data.index.date
    date_picker = st.selectbox("Choisissez une date", available_dates)

    if date_picker:
        # Convertir la date sélectionnée en datetime et la prédire
        date_picker = pd.to_datetime(date_picker)

        # Faire la prédiction pour cette date spécifique
        forecast_point = model_fit.get_prediction(start=date_picker, end=date_picker)
        forecast_value = forecast_point.predicted_mean[0]  # Prévision pour cette date spécifique
        forecast_conf_int = forecast_point.conf_int()

        st.write(f"Valeur estimée pour la date {date_picker.date()}: {forecast_value}")
        st.write("Intervalle de confiance :")
        st.write(forecast_conf_int)
