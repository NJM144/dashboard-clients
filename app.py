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
    selected_date = request.form.get("date_specifique")


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
    if selected_date:
        date_selected = pd.to_datetime(selected_date, errors='coerce')
        df_filtered = df_filtered[df_filtered['DATE DU TRANSFERT'].dt.date == date_selected.date()]


    # KPIs
    kpi_ca = int(df_filtered["MONTANT PAYER"].sum())
    kpi_volume = int(df_filtered["QUANTITE"].sum())
    kpi_nb_livraisons = df_filtered.shape[0]
    kpi_taux_paiement= round((df_filtered['MONTANT PAYER'] / df_filtered['PRIX']).mean() * 100, 2)
    kpi_restant_total=int(df_filtered["RESTANT A PAYER"].sum())

 

    df_clients = df_filtered.groupby('EXPEDITEUR')[['MONTANT PAYER', 'RESTANT A PAYER']].sum().reset_index()
    df_clients = df_clients.sort_values(['RESTANT A PAYER'], ascending=False).head(10)  # Top 10

    import plotly.express as px
    fig1 = px.bar(df_clients, x='EXPEDITEUR', y=['MONTANT PAYER', 'RESTANT A PAYER'],
                        title="Impayés par client",labels={"EXPEDITEUR": "Client", "value": "Montant", "variable": "Type"},barmode='stack')


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
=======
# ────────────────────────────────────────────────────────────
#  ROUTE  /dashboard  (remplace entièrement l’ancienne)
# ────────────────────────────────────────────────────────────
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    # ---------- 1. Récupération des filtres ---------------------------------
    selected_client   = request.form.get("client")
    selected_type     = request.form.get("type_colis")       # ou classe_colis selon ton HTML
    selected_annee    = request.form.get("annee")
    selected_mois     = request.form.get("mois")
    selected_date     = request.form.get("date_specifique")

    # ---------- 2. Filtrage centralisé --------------------------------------
    df_filtered = filter_df(df, request.form)  # << utilise ta fonction utilitaire

    # ---------- 3. KPI "généraux" (ceux que tu avais déjà) ------------------
    kpi_ca            = int(df_filtered["MONTANT PAYER"].sum())
    kpi_volume        = int(df_filtered["QUANTITE"].sum())
    kpi_nb_livraisons = len(df_filtered)
    kpi_taux_paiement = round(
    df_filtered["MONTANT PAYER"].sum() /
    df_filtered["PRIX"].sum() * 100, 2
)

    kpi_restant_total = int(df_filtered["RESTANT A PAYER"].sum())

    # ---------- 4. Anciens graphiques (fig1 à fig3) --------------------------
    df_clients = (df_filtered.groupby('EXPEDITEUR')[['MONTANT PAYER','RESTANT A PAYER']]
                  .sum().reset_index()
                  .sort_values('RESTANT A PAYER', ascending=False)
                  .head(10))
    fig1 = px.bar(df_clients, x='EXPEDITEUR',
                  y=['MONTANT PAYER','RESTANT A PAYER'],
                  barmode='stack', title="Impayés par client")

    df_filtered['Statut Paiement'] = df_filtered['MONTANT PAYER']\
        .apply(lambda x: 'Paiement partiel' if x > 0 else 'Aucun paiement')
    statut_counts = df_filtered['Statut Paiement'].value_counts().reset_index()
    statut_counts.columns = ['Statut', 'Nombre']
    fig2 = px.pie(statut_counts, names='Statut', values='Nombre')

    daily_freq = (df_filtered.dropna(subset=['DATE DU TRANSFERT'])
                  .groupby(df_filtered['DATE DU TRANSFERT'].dt.date)
                  .size().reset_index(name="nombre d'expéditions"))
    fig3 = px.line(daily_freq, x='DATE DU TRANSFERT', y="nombre d'expéditions")

    # ---------- 5. ➜ KPI + graphes PERFORMANCE ------------------------------
    col_class = 'CLASSE_COLIS' if 'CLASSE_COLIS' in df_filtered.columns else 'TYPE COLIS'

    perf_kpi = {
        "volume_total"    : kpi_volume,
        "nb_expeditions"  : kpi_nb_livraisons,
        "nb_types_colis"  : df_filtered[col_class].nunique(),
        "type_plus_frequent": (df_filtered[col_class].mode()[0]
                               if not df_filtered[col_class].empty else 'N/A')
    }

    df_vol_type = (df_filtered.groupby(col_class)['QUANTITE']
                   .sum().reset_index())
    fig_perf1 = px.pie(df_vol_type, names=col_class, values='QUANTITE',
                       title="Répartition du volume par type")

    df_vol_client = (df_filtered.groupby('EXPEDITEUR')['QUANTITE']
                     .sum().sort_values(ascending=False).head(10).reset_index())
    fig_perf2 = px.bar(df_vol_client, x='EXPEDITEUR', y='QUANTITE',
                       title="Top 10 clients – volume")

    df_temps = (df_filtered.assign(Jour=df_filtered['DATE DU TRANSFERT'].dt.date)
                .groupby('Jour')['QUANTITE'].sum().reset_index())
    fig_perf3 = px.line(df_temps, x='Jour', y='QUANTITE',
                        title="Évolution quotidienne des volumes")

    # ---------- 6. ➜ KPI + graphes FINANCES ---------------------------------
    fin_kpi = {
        "ca_total"        : kpi_ca,
        "restant_total"   : kpi_restant_total,
        "taux_encaissement": kpi_taux_paiement
    }

    top_ca = (df_filtered.groupby('EXPEDITEUR')['MONTANT PAYER']
              .sum().sort_values(ascending=False).head(10).reset_index())
    fig_fin1 = px.bar(top_ca, x='EXPEDITEUR', y='MONTANT PAYER',
                      title="Top 10 CA")

    top_imp = (df_filtered.groupby('EXPEDITEUR')['RESTANT A PAYER']
               .sum().sort_values(ascending=False).head(10).reset_index())
    fig_fin2 = px.bar(top_imp, x='EXPEDITEUR', y='RESTANT A PAYER',
                      title="Top 10 impayés")

    df_month = df_filtered.copy()
    df_month['Mois'] = df_month['DATE DU TRANSFERT'].dt.to_period('M').astype(str)
    df_month = (df_month.groupby('Mois')[['MONTANT PAYER','RESTANT A PAYER']]
                .sum().reset_index())
    fig_fin3 = px.line(df_month, x='Mois',
                       y=['MONTANT PAYER','RESTANT A PAYER'],
                       title="CA vs impayés (mensuel)")

    # ---------- 7. Listes déroulantes (inchangées) --------------------------
    clients = ["Tous"] + sorted(df["EXPEDITEUR"].dropna().unique())
    types   = ["Tous"] + sorted(df[col_class].dropna().unique())
    annees  = ["Tous"] + sorted(df["DATE DU TRANSFERT"].dt.year.dropna().unique().astype(str))
    mois    = ["Tous"] + [str(m).zfill(2) for m in sorted(df["DATE DU TRANSFERT"].dt.month.dropna().unique())]

    # ---------- 8. Envoi au template ---------------------------------------
    return render_template(
        "dashboard.html",

        # --- filtres & KPIs historiques ---
        graph1              = pio.to_html(fig1, full_html=False),
        graph2              = pio.to_html(fig2, full_html=False),
        graph3              = pio.to_html(fig3, full_html=False),
        clients             = clients,
        types               = types,
        annees              = annees,
        mois                = mois,
        selected_client     = selected_client,
        selected_type       = selected_type,
        selected_annee      = selected_annee,
        selected_mois       = selected_mois,
        selected_date       = selected_date,
        kpi_ca              = kpi_ca,
        kpi_volume          = kpi_volume,
        kpi_nb_livraisons   = kpi_nb_livraisons,
        kpi_taux_paiement   = kpi_taux_paiement,
        kpi_restant_total   = kpi_restant_total,

        # --- variables pour les ONGLETs ---
        perf_kpi = perf_kpi,
        fin_kpi  = fin_kpi,
        perf_g1  = pio.to_html(fig_perf1, full_html=False),
        perf_g2  = pio.to_html(fig_perf2, full_html=False),
        perf_g3  = pio.to_html(fig_perf3, full_html=False),
        fin_g1   = pio.to_html(fig_fin1, full_html=False),
        fin_g2   = pio.to_html(fig_fin2, full_html=False),
        fin_g3   = pio.to_html(fig_fin3, full_html=False)
    )

from flask import request, render_template

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    # 1) Date saisie ou valeur par défaut (demain)
    date_str = request.form.get("date_cible") or str((_date.today()).replace(day=_date.today().day + 1))
    target = pd.to_datetime(date_str)

    # 2) Features
    features = pd.DataFrame([{
        "jour_semaine": target.dayofweek,
        "mois": target.month
    }])
    pred = model_pred.predict(features)[0]
    pred_colis     = int(pred[0])
    pred_benef     = round(pred[1], 2)
    pred_credit    = round(pred[2], 2)

    # 3) Préparation mini-graphes historiques + point prédit (facultatif)
    import plotly.express as px, plotly.io as pio
    df_plot = df_daily_hist.copy()
    df_plot = df_plot.sort_values('jour')
    future_label = target.date().isoformat()

    def make_fig(col, title):
        df_aux = df_plot[['jour', col]].copy()
        df_aux.loc[len(df_aux)] = [future_label, pred_colis if col=='QUANTITE'
                                                else pred_benef if col=='BENEFICE'
                                                else pred_credit]
        fig = px.line(df_aux, x='jour', y=col, title=title)
        fig.update_traces(mode='lines+markers')
        return pio.to_html(fig, full_html=False)

    fig_colis  = make_fig('QUANTITE', "Historique quantité + prédiction")
    fig_benef  = make_fig('BENEFICE', "Historique bénéfice + prédiction")
    fig_credit = make_fig('RESTANT A PAYER', "Historique crédit + prédiction")

    return render_template(
        "prediction.html",
        selected_date=date_str,
        pred_colis=pred_colis,
        pred_benef=pred_benef,
        pred_credit=pred_credit,
        graph_colis=fig_colis,
        graph_benef=fig_benef,
        graph_credit=fig_credit
    )

@app.route('/performance', methods=['GET', 'POST'])
def performance():
    df_filtered = filter_df(df, request.form)

    # ---------- KPIs ----------
    volume_total = int(df_filtered['QUANTITE'].sum())
    nb_expeditions = len(df_filtered)
    nb_types_colis = df_filtered['CLASSE_COLIS'].nunique()
    type_plus_frequent = (
        df_filtered['CLASSE_COLIS'].mode()[0] if nb_types_colis > 0 else 'N/A'
    )

    # ---------- Graphiques ----------
    # 1. Répartition du volume par type de colis
    df_volume_type = (
        df_filtered.groupby('CLASSE_COLIS')['QUANTITE']
        .sum().reset_index()
    )
    fig1 = px.pie(
        df_volume_type, names='CLASSE_COLIS', values='QUANTITE',
        title="Répartition du volume par type de colis"
    )

    # 2. Top 10 clients par volume
    df_vol_client = (
        df_filtered.groupby('EXPEDITEUR')['QUANTITE']
        .sum().sort_values(ascending=False).head(10).reset_index()
    )
    fig2 = px.bar(
        df_vol_client, x='EXPEDITEUR', y='QUANTITE',
        title="Top 10 clients – volume expédié"
    )

    # 3. Volume expédié dans le temps
    df_temps = df_filtered.copy()
    df_temps['DATE'] = df_temps['DATE DU TRANSFERT'].dt.date
    df_temps = df_temps.groupby('DATE')['QUANTITE'].sum().reset_index()
    fig3 = px.line(df_temps, x='DATE', y='QUANTITE',
                   title="Évolution quotidienne des volumes")

    # ---------- Options des menus ----------
    clients = ["Tous"] + sorted(df["EXPEDITEUR"].dropna().unique())
    types   = ["Tous"] + sorted(df["CLASSE_COLIS"].dropna().unique())
    annees  = ["Tous"] + sorted(df["DATE DU TRANSFERT"].dt.year.dropna().unique().astype(str))
    mois    = ["Tous"] + [str(m).zfill(2) for m in sorted(df["DATE DU TRANSFERT"].dt.month.dropna().unique())]

    return render_template(
        "performance.html",
        graph1=pio.to_html(fig1, full_html=False),
        graph2=pio.to_html(fig2, full_html=False),
        graph3=pio.to_html(fig3, full_html=False),
        clients=clients, types=types, annees=annees, mois=mois,
        selected_client=request.form.get("client"),
        selected_type=request.form.get("classe_colis"),
        selected_annee=request.form.get("annee"),
        selected_mois=request.form.get("mois"),
        selected_date=request.form.get("date_specifique"),
        volume_total=volume_total,
        nb_expeditions=nb_expeditions,
        nb_types_colis=nb_types_colis,
        type_plus_frequent=type_plus_frequent
    )

@app.route('/finances', methods=['GET', 'POST'])
def finances():
    df_filtered = filter_df(df, request.form)

    # ---------- KPIs ----------
    ca_total        = int(df_filtered["MONTANT PAYER"].sum())
    restant_total   = int(df_filtered["RESTANT A PAYER"].sum())
    taux_encaissemt = round(
        (df_filtered['MONTANT PAYER'] / df_filtered['PRIX']).mean()
        * 100, 2
    )

    # ---------- Graphiques ----------
    top_ca = (
        df_filtered.groupby('EXPEDITEUR')['MONTANT PAYER']
        .sum().sort_values(ascending=False).head(10).reset_index()
    )
    fig1 = px.bar(
        top_ca, x='EXPEDITEUR', y='MONTANT PAYER',
        title="Top 10 clients – chiffre d'affaires"
    )

    top_impaye = (
        df_filtered.groupby('EXPEDITEUR')['RESTANT A PAYER']
        .sum().sort_values(ascending=False).head(10).reset_index()
    )
    fig2 = px.bar(
        top_impaye, x='EXPEDITEUR', y='RESTANT A PAYER',
        title="Top 10 clients – impayés"
    )

    # CA & impayés par mois
    df_month = df_filtered.copy()
    df_month['Mois'] = df_month['DATE DU TRANSFERT'].dt.to_period('M').astype(str)
    df_month = (
        df_month.groupby('Mois')[['MONTANT PAYER', 'RESTANT A PAYER']]
        .sum().reset_index()
    )
    fig3 = px.line(
        df_month, x='Mois',
        y=['MONTANT PAYER', 'RESTANT A PAYER'],
        title="CA vs impayés (mensuel)"
    )

    # ---------- Options menus (mêmes que plus haut) ----------
    clients = ["Tous"] + sorted(df["EXPEDITEUR"].dropna().unique())
    types   = ["Tous"] + sorted(df["TYPE COLIS"].dropna().unique())
    annees  = ["Tous"] + sorted(df["DATE DU TRANSFERT"].dt.year.dropna().unique().astype(str))
    mois    = ["Tous"] + [str(m).zfill(2) for m in sorted(df["DATE DU TRANSFERT"].dt.month.dropna().unique())]

    return render_template(
        "finances.html",
        graph1=pio.to_html(fig1, full_html=False),
        graph2=pio.to_html(fig2, full_html=False),
        graph3=pio.to_html(fig3, full_html=False),
        clients=clients, types=types, annees=annees, mois=mois,
        selected_client=request.form.get("client"),
        selected_type=request.form.get("type_colis"),
        selected_annee=request.form.get("annee"),
        selected_mois=request.form.get("mois"),
        selected_date=request.form.get("date_specifique"),
        ca_total=ca_total,
        restant_total=restant_total,
        taux_encaissemt=taux_encaissemt
    )


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
                           selected_date=selected_date,

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
    selected_date = request.form.get("date_specifique")


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
    if selected_date:
        date_selected = pd.to_datetime(selected_date, errors='coerce')
        df_filtered = df_filtered[df_filtered['DATE DU TRANSFERT'].dt.date == date_selected.date()]


    # KPIs
    kpi_nb_client = int(df_filtered['EXPEDITEUR'].nunique())

    ca_par_client = df_filtered.groupby('EXPEDITEUR')['MONTANT PAYER'].sum().reset_index()
    top_client_ca = ca_par_client.sort_values(by='MONTANT PAYER', ascending=False).iloc[0]
    kpi_top1=top_client_ca['EXPEDITEUR']
    top_ca =float(top_client_ca['MONTANT PAYER'])

    livraisons_par_client = df_filtered['EXPEDITEUR'].value_counts().reset_index()
    livraisons_par_client.columns = ['Client', 'Nb Livraisons']
    top_client_livraisons = livraisons_par_client.iloc[0]
    client_name= top_client_livraisons['Client']
    nb_livraisons= float(top_client_livraisons['Nb Livraisons'])

    df_filtered['PRIX'].replace(0, pd.NA, inplace=True)
    df_filtered['Taux Impaye'] = df_filtered['RESTANT A PAYER'] / df_filtered['PRIX']
    taux_moyen_impaye = round(df_filtered.groupby('EXPEDITEUR')['Taux Impaye'].mean().mean(skipna=True) * 100,2)
    
     # Graphiques Clients
    import plotly.express as px
    import plotly.io as pio

    # 1. Top 10 CA
    top10_ca = ca_par_client.sort_values(by='MONTANT PAYER', ascending=False).head(10)
    fig1 = px.bar(top10_ca, x='EXPEDITEUR', y='MONTANT PAYER',
                  title="Top 10 Clients par Chiffre d'Affaires")

    # 2. Top 10 Livraisons
    top10_livraisons = livraisons_par_client.head(10)
    fig2 = px.bar(top10_livraisons, x='Client', y='Nb Livraisons',
                  title="Top 10 Clients par Nombre de Livraisons")

    # 3. Répartition CA
    fig3 = px.pie(top10_ca, names='EXPEDITEUR', values='MONTANT PAYER',
                  title="Répartition du CA entre les 10 premiers clients", hole=0.4)

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
                           selected_date=selected_date,
                           kpi_nb_client=kpi_nb_client,
                           kpi_top1=kpi_top1,
                           top_ca=top_ca,
                           client_name=client_name,
                           nb_livraisons=nb_livraisons,
                           taux_moyen_impaye=taux_moyen_impaye,
                           graph1=pio.to_html(fig1, full_html=False),
                           graph2=pio.to_html(fig2, full_html=False),
                           graph3=pio.to_html(fig3, full_html=False)
                           ) # Mets tes variables ici    


@app.route('/tournees')
def tournees():
    return render_template("tournees.html")

@app.route('/logistique',methods=['GET', 'POST'])
def logistique():
    # Récupération des filtres
    selected_client = request.form.get("client")
    selected_type = request.form.get("classe_colis")
    selected_annee = request.form.get("annee")
    selected_mois = request.form.get("mois")
    selected_date = request.form.get("date_specifique")


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
    if selected_date:
        date_selected = pd.to_datetime(selected_date, errors='coerce')
        df_filtered = df_filtered[df_filtered['DATE DU TRANSFERT'].dt.date == date_selected.date()]


     # KPI Logistique & Stock
    nb_expéditions = len(df_filtered)
    volume_total = int(df_filtered['QUANTITE'].sum())
    nb_types_colis = df_filtered['CLASSE_COLIS'].nunique()
    type_plus_frequent = df_filtered['CLASSE_COLIS'].mode()[0] if nb_types_colis > 0 else 'N/A'
    client_top_volume = df_filtered.groupby('EXPEDITEUR')['QUANTITE'].sum().idxmax()

    # Graphique 1 : Volume par type de colis
    df_volume_type = df_filtered.groupby('CLASSE_COLIS')['QUANTITE'].sum().reset_index()
    fig_volume_type = px.pie(df_volume_type, names='CLASSE_COLIS', values='QUANTITE',
                             title="Répartition du Volume par Type de Colis")

    # Graphique 2 : Top 10 Clients par Volume
    df_volume_clients = df_filtered.groupby('EXPEDITEUR')['QUANTITE'].sum().reset_index()
    df_volume_clients = df_volume_clients.sort_values(by='QUANTITE', ascending=False).head(10)
    fig_top_clients = px.bar(df_volume_clients, x='EXPEDITEUR', y='QUANTITE',
                             title="Top 10 Clients par Volume Expédié")

    # Graphique 3 : Volume expédié dans le temps (par mois)
    df_filtered['Mois'] = df_filtered['DATE DU TRANSFERT'].dt.to_period('M')
    df_volume_mensuel = df_filtered.groupby('Mois')['QUANTITE'].sum().reset_index()
    df_volume_mensuel['Mois'] = df_volume_mensuel['Mois'].astype(str)
    fig_volume_temps = px.line(df_volume_mensuel, x='Mois', y='QUANTITE',
                               title="Évolution Mensuelle des Volumes Expédiés")


    
    # Valeurs des menus déroulants
    clients = ["Tous"] + sorted(df["EXPEDITEUR"].dropna().unique().tolist())
    types = ["Tous"] + sorted(df["CLASSE_COLIS"].dropna().unique().tolist())
    annees = ["Tous"] + sorted(df["DATE DU TRANSFERT"].dropna().dt.year.unique().astype(str))
    mois = ["Tous"] + [str(m).zfill(2) for m in sorted(df["DATE DU TRANSFERT"].dropna().dt.month.unique())]

    return render_template("logistique.html",
                            clients=clients,
                           types=types,
                           annees=annees,
                           mois=mois,
                           selected_client=selected_client,
                           selected_type=selected_type,
                           selected_annee=selected_annee,
                           selected_mois=selected_mois,
                           selected_date=selected_date,
                           nb_expéditions=nb_expéditions,
                           volume_total=volume_total,
                           nb_types_colis=nb_types_colis,
                           type_plus_frequent=type_plus_frequent,
                           client_top_volume=client_top_volume,
                           graph1=pio.to_html(fig_volume_type, full_html=False),
                           graph2=pio.to_html(fig_top_clients, full_html=False),
                           graph3=pio.to_html(fig_volume_temps, full_html=False)
                           )

@app.route('/alertes')
def alertes():
    return render_template("alertes.html")

if __name__ == '__main__':
    app.run(debug=True)
