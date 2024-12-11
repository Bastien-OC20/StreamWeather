import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm


# Titre de l'application
st.title("Prévision de la température avec ARIMA")

uploaded_file = "./data/london_weather.csv"  # Remplacez par votre fichier chargé
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Aperçu des données")
    st.write(data.head())

    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        data['date_YYYYMMDD'] = data['date'].dt.strftime('%Y%m%d')
        data.set_index('date', inplace=True)

        # Ne garder que la colonne 'mean_temp'
        if 'mean_temp' in data.columns:
            series = data['mean_temp'].dropna()
        else:
            st.error("La colonne 'mean_temp' n'existe pas dans les données.")
            series = pd.Series()  # Série vide en cas d'erreur

        st.subheader("Série temporelle brute")
        st.line_chart(series)

        st.subheader("Test de stationnarité (ADF)")

        def adf_test(series):
            result = adfuller(series)
            return result[0], result[1]

        adf_stat, p_value = adf_test(series)
        st.write(f"Statistique ADF : {adf_stat:.3f}")
        st.write(f"P-valeur : {p_value:.3f}")
        if p_value > 0.05:
            st.write("La série n'est pas stationnaire.")
        else:
            st.write("La série est stationnaire.")

        st.subheader("Transformation pour stationnarité")
        diff_order = st.sidebar.slider("Ordre de différenciation", 0, 2, 1)
        diff_series = series.diff(diff_order).dropna()
        st.line_chart(diff_series)

        st.sidebar.subheader("Ajustement ARIMA")
        if st.sidebar.button("Exécuter Auto-ARIMA"):
            model = pm.auto_arima(
                series, seasonal=False, stepwise=True, suppress_warnings=True
            )
            st.sidebar.write("Meilleur modèle ARIMA trouvé :", model.order)
        else:
            p = st.sidebar.number_input(
                "Paramètre p (AR)", min_value=0, max_value=5, value=1
            )
            d = st.sidebar.number_input(
                "Paramètre d (diff)", min_value=0, max_value=2, value=diff_order
            )
            q = st.sidebar.number_input(
                "Paramètre q (MA)", min_value=0, max_value=5, value=1
            )
            model = ARIMA(series, order=(p, d, q)).fit()
            st.write("Modèle ajusté.")

            st.subheader("Prévision et intervalle de confiance")
        steps = st.number_input(
            "Nombre de pas pour la prévision", min_value=1, max_value=100, value=10
        )
        forecast = model.get_forecast(steps=steps)
        forecast_mean = forecast.predicted_mean
        conf_int = forecast.conf_int()

        st.write("Prévisions :")
        st.write(forecast_mean)
        
        fig, ax = plt.subplots()
        ax.plot(series, label="Données historiques")
        ax.plot(forecast_mean, label="Prévisions", color='red')
        ax.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
        ax.legend()
        st.pyplot(fig)
