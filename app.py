"""

En local :  streamlit run app.py
"""

import numpy as np
import pandas as pd
import joblib as jb
import streamlit as st

st.set_page_config(page_title="Performance étudiante", page_icon="📚", layout="centered")


# ---------- Objets issus du notebook (chargés une seule fois) ----------
@st.cache_resource
def charger_objets():
    encoders = jb.load("encoders.joblib")              # encodeur de Extracurricular Activities
    pipe_from_grid = jb.load("pipe_from_grid.joblib")  # pipeline du meilleur modèle
    return encoders, pipe_from_grid


encoders, pipe_from_grid = charger_objets()
modalites_activites = list(encoders[0].classes_)  # ['No', 'Yes']


# ---------- Prédiction pour un étudiant ----------
def Pred_func(heures, notes_prec, activites, sommeil, sujets):
    code_activites = encoders[0].transform([activites])[0]
    # même ordre de colonnes que dans le notebook
    vecteur = np.array([heures, notes_prec, code_activites, sommeil, sujets]).reshape(1, -1)
    indice = pipe_from_grid.predict(vecteur)[0]   # le pipeline normalise lui-même les données
    return round(float(indice), 1)


# ---------- Prédiction pour un fichier ----------
def Pred_func_csv(fichier):
    tableau = pd.read_csv(fichier)
    resultats = []
    for ligne in tableau.values:
        resultats.append(Pred_func(ligne[0], ligne[1], ligne[2], ligne[3], ligne[4]))
    tableau["Performance Index prédit"] = resultats
    return tableau


st.title("📚 Indice de performance")
st.caption("Estimation de l'indice de performance (sur 100) d'un étudiant.")
onglet_un, onglet_csv = st.tabs(["Un étudiant", "Fichier CSV"])

with onglet_un:
    gauche, droite = st.columns(2)
    with gauche:
        heures = st.slider("Heures de révision", 1, 9, 5)
        notes_prec = st.slider("Notes précédentes", 40, 99, 70)
        activites = st.radio("Activités extrascolaires", modalites_activites, horizontal=True)
    with droite:
        sommeil = st.slider("Heures de sommeil", 4, 9, 7)
        sujets = st.slider("Sujets d'entraînement traités", 0, 9, 4)

    if st.button("Prédire", type="primary", use_container_width=True):
        try:
            resultat = Pred_func(heures, notes_prec, activites, sommeil, sujets)
            st.success(f"**Indice de performance estimé (sur 100) :** {resultat}")
        except Exception as erreur:
            st.error(f"Prédiction impossible : {erreur}")
with onglet_csv:
    st.info("Colonnes attendues, dans cet ordre : Hours Studied, Previous Scores, Extracurricular Activities, "
            "Sleep Hours, Sample Question Papers Practiced.")
    fichier = st.file_uploader("Choisir un fichier CSV", type="csv")
    if fichier is not None:
        try:
            tableau = Pred_func_csv(fichier)
            st.dataframe(tableau, use_container_width=True)
            st.download_button("Télécharger les résultats", tableau.to_csv(index=False).encode("utf-8"),
                               "resultats_etudiants.csv", "text/csv")
        except Exception as erreur:
            st.error(f"Fichier non traité : {erreur}")
