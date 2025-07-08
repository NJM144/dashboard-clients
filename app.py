from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.express as px
import plotly.io as pio
from datetime import datetime
import plotly

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

# Charger les données
DATA_PATH = "data/Transferts_classes.csv"
df = pd.read_csv(DATA_PATH, sep=';')
df["DATE DU TRANSFERT"] = pd.to_datetime(df["DATE DU TRANSFERT"], format="%d/%m/%Y %H:%M", errors="coerce")


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
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
    kpi_nb_livraisons = df_filtered.shape[0]
    kpi_taux_paiement= round((df_filtered['MONTANT PAYER'] / df_filtered['PRIX']).mean() * 100, 2)
    kpi_restant_total=int(df_filtered["RESTANT A PAYER"].sum())

    # Graphique 1 : Top 15 IMPAYS par clients
    # ca_client = df_filtered.groupby("EXPEDITEUR")["MONTANT PAYER"].sum().reset_index()
    # ca_client = ca_client.sort_values("MONTANT PAYER", ascending=False).head(10)
    # fig1 = px.bar(ca_client, x="EXPEDITEUR", y="MONTANT PAYER",
    #              title="Top 10 Clients - Chiffre d'affaires",
    #              labels={"EXPEDITEUR": "Client", "MONTANT PAYER": "Montant Payé (FCFA)"})

    df_clients = df_filtered.groupby('EXPEDITEUR')[['MONTANT PAYER', 'RESTANT A PAYER']].sum().reset_index()
    df_clients = df_clients.sort_values(['RESTANT A PAYER'], ascending=False).head(10)  # Top 10

    import plotly.express as px
    fig1 = px.bar(df_clients, x='EXPEDITEUR', y=['MONTANT PAYER', 'RESTANT A PAYER'],
                        title="Impayés par client",labels={"EXPEDITEUR": "Client", "value": "Montant", "variable": "Type"},barmode='stack')


    # Graphique 2 :  Top 5 types de colis
    # colis_type = df_filtered["TYPE COLIS"].value_counts().reset_index().head(5)
    # colis_type.columns = ["TYPE COLIS", "COUNT"]
    # fig2 = px.pie(colis_type, names="TYPE COLIS", values="COUNT", title="Top 5 des types de colis")

    df_filtered['Statut Paiement'] = df_filtered['MONTANT PAYER'].apply(lambda x: 'Paiement partiel' if x > 0 else 'Aucun paiement')
    statut_counts = df_filtered['Statut Paiement'].value_counts().reset_index()
    statut_counts.columns = ['Statut', 'Nombre']

    fig2 = px.pie(statut_counts, names='Statut', values='Nombre',
                    )

    # Graphique 3 : Fréquence d'envoi quotidienne
    daily_freq = df_filtered.dropna(subset=["DATE DU TRANSFERT"])
    daily_freq = daily_freq.groupby(daily_freq["DATE DU TRANSFERT"].dt.date).size().reset_index(name="nombre d'expéditions")
    fig3 = px.line(daily_freq, x="DATE DU TRANSFERT", y="nombre d'expéditions")

    

    # Valeurs des menus déroulants
    clients = ["Tous"] + sorted(df["EXPEDITEUR"].dropna().unique().tolist())
    types = ["Tous"] + sorted(df["TYPE COLIS"].dropna().unique().tolist())
    annees = ["Tous"] + sorted(df["DATE DU TRANSFERT"].dropna().dt.year.unique().astype(str))
    mois = ["Tous"] + [str(m).zfill(2) for m in sorted(df["DATE DU TRANSFERT"].dropna().dt.month.unique())]

    return render_template("dashboard.html",graph1=pio.to_html(fig1, full_html=False),
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
                           kpi_nb_livraisons=kpi_nb_livraisons,
                           kpi_taux_paiement=kpi_taux_paiement,
                           kpi_restant_total=kpi_restant_total
                           ) # Mets tes variables ici

@app.route('/prediction')
def prediction():
    return "<h2>Page de prédiction à venir...</h2>"  # à remplacer plus tard






@app.route('/client_detail', methods=['POST'])
def client_detail():
    client = request.json.get("client")
    client_data = df[df["EXPEDITEUR"] == client][["DATE DU TRANSFERT", "DESTINATAIRE", "TYPE COLIS", "QUANTITE", "MONTANT PAYER"]]
    records = client_data.to_dict(orient="records")
    return jsonify(records)


@app.route('/clients', methods=['GET', 'POST'])
def clients():
      # Récupération des filtres
    selected_client = request.form.get("client")
    selected_type = request.form.get("classe_colis")
    selected_annee = request.form.get("annee")
    selected_mois = request.form.get("mois")

    df_filtered = df.copy()

    # Filtres dynamiques
    if selected_client and selected_client != "Tous":
        df_filtered = df_filtered[df_filtered["EXPEDITEUR"] == selected_client]
    if selected_type and selected_type != "Tous":
        df_filtered = df_filtered[df_filtered["CLASSE_COLIS"] == selected_type]
    if selected_annee and selected_annee != "Tous":
        df_filtered = df_filtered[df_filtered["DATE DU TRANSFERT"].dt.year == int(selected_annee)]
    if selected_mois and selected_mois != "Tous":
        df_filtered = df_filtered[df_filtered["DATE DU TRANSFERT"].dt.month == int(selected_mois)]

    # KPIs
    kpi_nb_client = int(df_filtered['EXPEDITEUR'].nunique())
    kpi_top1= df_filtered.groupby('EXPEDITEUR')['MONTANT PAYER'].sum().reset_index().sort_values(by='MONTANT PAYER', ascending=False).head(1)
    # kpi_best_client=df.groupby('EXPEDITEUR').count()
    # kpi_impayeParClients=
    
    

    # Valeurs des menus déroulants
    clients = ["Tous"] + sorted(df["EXPEDITEUR"].dropna().unique().tolist())
    types = ["Tous"] + sorted(df["CLASSE_COLIS"].dropna().unique().tolist())
    annees = ["Tous"] + sorted(df["DATE DU TRANSFERT"].dropna().dt.year.unique().astype(str))
    mois = ["Tous"] + [str(m).zfill(2) for m in sorted(df["DATE DU TRANSFERT"].dropna().dt.month.unique())]

    return render_template("clients.html",
                           clients=clients,
                           types=types,
                           annees=annees,
                           mois=mois,
                           selected_client=selected_client,
                           selected_type=selected_type,
                           selected_annee=selected_annee,
                           selected_mois=selected_mois,
                           kpi_nb_client=kpi_nb_client,
                           kpi_top1=kpi_top1
                           ) # Mets tes variables ici    
@app.route('/tournees')
def tournees():
    return render_template("tournees.html")

@app.route('/logistique')
def logistique():
    return render_template("logistique.html")

@app.route('/alertes')
def alertes():
    return render_template("alertes.html")

if __name__ == '__main__':
    app.run(debug=True)
