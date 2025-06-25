import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

st.set_page_config(page_title="Poissonverteilung", layout="centered")

# Titel
st.title("Poissonverteilung interaktiv")

# Sidebar mit Eingabeparametern
st.sidebar.header("Parameter")
lambda_val = st.sidebar.slider("λ (Erwartungswert)", min_value=0.1, max_value=60.0, value=4.0, step=0.1)
x_min = st.sidebar.number_input("x-Minimum", min_value=0, value=0, step=1)
x_max = st.sidebar.number_input("x-Maximum", min_value=1, value=15, step=1)

# Eingabe für k
k = st.sidebar.number_input("k-Wert für P(X = k)", min_value=0, value=2, step=1)

if x_min >= x_max:
    st.error("x-Minimum muss kleiner als x-Maximum sein.")
else:
    # x-Werte
    x = np.arange(x_min, x_max + 1)
    y = poisson.pmf(x, mu=lambda_val)

    # Darstellung
    fig, ax = plt.subplots()
    ax.bar(x, y, color='skyblue', edgecolor='black')
    ax.set_title(f"Poisson-Verteilung (λ = {lambda_val})")
    ax.set_xlabel("x")
    ax.set_ylabel("Wahrscheinlichkeit P(X = x)")
    ax.grid(True, linestyle='--', alpha=0.5)

    # k-Wert hervorheben (wenn im Bereich)
    if x_min <= k <= x_max:
        ax.bar(k, poisson.pmf(k, mu=lambda_val), color='orange', edgecolor='black', label=f"P(X={k})")
        ax.legend()

    st.pyplot(fig)

    # Tabelle anzeigen
    st.subheader("Wahrscheinlichkeiten")
    st.dataframe({"x": x, f"P(X=x) bei λ={lambda_val}": np.round(y, 4)})

    # P(X=k) anzeigen
    st.subheader(f"Wahrscheinlichkeit für k = {k}")
    prob_k = poisson.pmf(k, mu=lambda_val)
    st.write(f"**P(X = {k}) = {prob_k:.4f}**")
