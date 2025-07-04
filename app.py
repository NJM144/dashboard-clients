from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.express as px
import plotly.io as pio
from datetime import datetime
import plotly

app = Flask(__name__)

# Charger les données
DATA_PATH = "data/ListeTransfertdu_2025-06-01_au_2025-06-25.csv"
df = pd.read_csv(DATA_PATH, sep=';', on_bad_lines='skip')
df["DATE DU TRANSFERT"] = pd.to_datetime(df["DATE DU TRANSFERT"], errors='coerce')

@app.route('/', methods=['GET', 'POST'])
def index():
    # Récupération des filtres
    selected_client = request.form.get("client")
    selected_type = request.form.get("type_colis")
    selected_annee = request.form.get("annee")
    selected_mois = request.form.get("mois")

    df_filtered = df.copy()

    # Filtres dynamiques
    if selected_client and selected_client != "Tous":
        df_filtered = df_filtered[df_filtered["EXPEDITEUR"] == selected_client]
    if selected_type and selected_type != "Tous":
        df_filtered = df_filtered[df_filtered["TYPE COLIS"] == selected_type]
    if selected_annee and selected_annee != "Tous":
        df_filtered = df_filtered[df_filtered["DATE DU TRANSFERT"].dt.year == int(selected_annee)]
    if selected_mois and selected_mois != "Tous":
        df_filtered = df_filtered[df_filtered["DATE DU TRANSFERT"].dt.month == int(selected_mois)]

    # KPIs
    kpi_ca = int(df_filtered["MONTANT PAYER"].sum())
    kpi_volume = int(df_filtered["QUANTITE"].sum())
    kpi_colis = df_filtered.shape[0]

    # Graphique 1 : Top 10 clients (cliquable)
    ca_client = df_filtered.groupby("EXPEDITEUR")["MONTANT PAYER"].sum().reset_index()
    ca_client = ca_client.sort_values("MONTANT PAYER", ascending=False).head(10)
    fig1 = px.bar(ca_client, x="EXPEDITEUR", y="MONTANT PAYER",
                  title="Top 10 Clients - Chiffre d'affaires",
                  labels={"EXPEDITEUR": "Client", "MONTANT PAYER": "Montant Payé (FCFA)"})
    fig1.update_traces(customdata=ca_client["EXPEDITEUR"],
                       hovertemplate='Client: %{x}<br>Montant: %{y} FCFA<br>')
    fig1.update_layout(clickmode='event+select')

    # Graphique 2 :  Top 5 types de colis
    colis_type = df_filtered["TYPE COLIS"].value_counts().reset_index().head(5)
    colis_type.columns = ["TYPE COLIS", "COUNT"]
    fig2 = px.pie(colis_type, names="TYPE COLIS", values="COUNT", title="Top 5 des types de colis")

    # Graphique 3 : Fréquence d'envoi quotidienne
    daily_freq = df_filtered.dropna(subset=["DATE DU TRANSFERT"])
    daily_freq = daily_freq.groupby(daily_freq["DATE DU TRANSFERT"].dt.date).size().reset_index(name="nb_envois")
    fig3 = px.line(daily_freq, x="DATE DU TRANSFERT", y="nb_envois", title="Fréquence d'envoi quotidienne")

    # Valeurs des menus déroulants
    clients = ["Tous"] + sorted(df["EXPEDITEUR"].dropna().unique().tolist())
    types = ["Tous"] + sorted(df["TYPE COLIS"].dropna().unique().tolist())
    annees = ["Tous"] + sorted(df["DATE DU TRANSFERT"].dropna().dt.year.unique().astype(str))
    mois = ["Tous"] + [str(m).zfill(2) for m in sorted(df["DATE DU TRANSFERT"].dropna().dt.month.unique())]

    return render_template("index.html",
                           graph1=pio.to_html(fig1, full_html=False),
                           graph2=pio.to_html(fig2, full_html=False),
                           graph3=pio.to_html(fig3, full_html=False),
                           clients=clients,
                           types=types,
                           annees=annees,
                           mois=mois,
                           selected_client=selected_client,
                           selected_type=selected_type,
                           selected_annee=selected_annee,
                           selected_mois=selected_mois,
                           kpi_ca=kpi_ca,
                           kpi_volume=kpi_volume,
                           kpi_colis=kpi_colis)

@app.route('/client_detail', methods=['POST'])
def client_detail():
    client = request.json.get("client")
    client_data = df[df["EXPEDITEUR"] == client][["DATE DU TRANSFERT", "DESTINATAIRE", "TYPE COLIS", "QUANTITE", "MONTANT PAYER"]]
    records = client_data.to_dict(orient="records")
    return jsonify(records)

if __name__ == '__main__':
    app.run(debug=True)
