import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm
import time

# === Seiteneinstellungen ===
st.set_page_config(page_title="Poissonverteilung", layout="centered")
st.title("Poissonverteilung mit Verlauf und Normalapproximation")

# === Session State initialisieren ===
for key in ["animate", "pause", "repeat", "lambda_index", "history"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "history" else 0 if key == "lambda_index" else False

# === Platzhalter für Diagramm ===
plot_placeholder = st.empty()

# === Sidebar-Parameter ===
st.sidebar.header("Parameter")
lambda_val = st.sidebar.slider("λ (Erwartungswert)", 0.1, 60.0, 4.0, 0.1)
x_min = st.sidebar.number_input("x-Minimum", min_value=0, value=0, step=1)
x_max = st.sidebar.number_input("x-Maximum", min_value=1, value=20, step=1)
speed = st.sidebar.slider("Geschwindigkeit der Animation (Sekunden)", 0.01, 1.0, 0.1, 0.01)

# === x-Werte definieren ===
if x_min >= x_max:
    st.error("x-Minimum muss kleiner als x-Maximum sein.")
    st.stop()

x = np.arange(x_min, x_max + 1)

# === Funktion zur Darstellung ===
def plot_poisson_with_history(current_lam):
    y = poisson.pmf(x, mu=current_lam)
    st.session_state.history.append((current_lam, y))

    fig, ax = plt.subplots()

    # Frühere Verteilungen in Grau
    for lam_hist, y_hist in st.session_state.history[:-1]:
        ax.plot(x, y_hist, color='gray', alpha=0.3)

    # Aktuelle Verteilung als Balken
    ax.bar(x, y, color='skyblue', edgecolor='black', label=f"λ = {current_lam:.2f}")

    # Statistik
    mu = current_lam
    sigma = np.sqrt(current_lam)
    varianz = current_lam

    st.markdown(f"""
    **Statistische Kennwerte bei λ = {mu:.2f}:**  
    • Erwartungswert (μ) = {mu:.2f}  
    • Varianz (σ²) = {varianz:.2f}  
    • Standardabweichung (σ) = √λ = {sigma:.2f}
    """)

    # Normalverteilung ab λ ≥ 30
    if current_lam >= 30:
        st.info("die Poissonverteilung nähert sich der Normalverteilung.")
        x_fine = np.linspace(x_min, x_max, 500)
        y_norm = norm.pdf(x_fine, mu, sigma)
        y_norm_scaled = y_norm * np.sum(poisson.pmf(x, mu=mu))
        ax.plot(x_fine, y_norm_scaled, color='red', linestyle='--', label='Normalapproximation')

    ax.set_title("Fortlaufender Verlauf der Poissonverteilung")
    ax.set_xlabel("x")
    ax.set_ylabel("P(X = x)")
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plot_placeholder.pyplot(fig)

# === Steuerungs-Buttons ===
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    if st.button("▶️ Starten"):
        st.session_state.animate = True
        st.session_state.pause = False
        st.session_state.repeat = False
        st.session_state.lambda_index = 0
        st.session_state.history = []
with col2:
    if st.button("⏸️ Pause"):
        st.session_state.pause = True
        if st.session_state.lambda_index > 0:
            lam = np.linspace(1, lambda_val, 40)[st.session_state.lambda_index - 1]
            plot_poisson_with_history(lam)
with col3:
    if st.button("▶️ Weiter"):
        if st.session_state.pause:
            st.session_state.pause = False
with col4:
    if st.button("⏹️ Stoppen"):
        st.session_state.animate = False
        st.session_state.pause = False
        st.session_state.lambda_index = 0
        st.session_state.history = []
with col5:
    if st.button("🔁 Wiederholen"):
        st.session_state.animate = True
        st.session_state.repeat = True
        st.session_state.pause = False
        st.session_state.lambda_index = 0
        st.session_state.history = []

# === Animation starten ===
lambda_range = np.linspace(1, lambda_val, 40)

if st.session_state.animate:
    while st.session_state.animate and st.session_state.lambda_index < len(lambda_range):
        if st.session_state.pause:
            break
        lam = lambda_range[st.session_state.lambda_index]
        plot_poisson_with_history(lam)
        st.session_state.lambda_index += 1
        time.sleep(speed)

    # Wenn fertig
    if st.session_state.lambda_index >= len(lambda_range):
        if st.session_state.repeat:
            st.session_state.lambda_index = 0
            st.session_state.history = []
        else:
            st.session_state.animate = False
            st.session_state.lambda_index = 0

# === Einzeldarstellung, falls keine Animation ===
if not st.session_state.animate:
    y = poisson.pmf(x, mu=lambda_val)
    fig, ax = plt.subplots()
    ax.bar(x, y, color='skyblue', edgecolor='black')
    ax.set_title(f"Poisson-Verteilung (λ = {lambda_val})")
    ax.set_xlabel("x")
    ax.set_ylabel("P(X = x)")
    ax.grid(True, linestyle='--', alpha=0.5)
    plot_placeholder.pyplot(fig)

    st.subheader("Wahrscheinlichkeiten")
    st.dataframe({"x": x, f"P(X=x) bei λ={lambda_val}": np.round(y, 4)})
