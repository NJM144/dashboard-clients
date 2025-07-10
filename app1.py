# app.py

# ... (garder tous les imports et le chargement des données au début)
from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.express as px
import plotly.io as pio
from datetime import datetime
import plotly
# imports généraux

from datetime import date as _date      #  ←←  AJOUTE (ou vérifie) CETTE LIGNE
import plotly.express as px
import plotly.io as pio
from flask import request, render_template

app = Flask(__name__)

#  Configuration du cache (simple, en mémoire)
config = {
    "DEBUG": True,
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 300 # 300 secondes = 5 minutes
}
app.config.from_mapping(config)
cache = Cache(app) # Initialiser le cache


app.debug = True

import os
from joblib import load


from joblib import load
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__),"model",  "model_random_forest.joblib")

try:
    model_pred = load(MODEL_PATH)
except Exception as e:
    print("❌ Erreur chargement modèle :", e)
    model_pred = None


# ── Charge et entraîne une seule fois ────────────────────────────────────
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import date as _date

def train_prediction_model(csv_path="data/Transferts_complet.csv"):
    df = pd.read_csv(csv_path, sep=';')
    df['DATE DU TRANSFERT'] = pd.to_datetime(df['DATE DU TRANSFERT'], errors='coerce')
    df['jour'] = df['DATE DU TRANSFERT'].dt.date

    daily = (df.groupby('jour')
               .agg({'QUANTITE':'sum','MONTANT PAYER':'sum',
                     'RESTANT A PAYER':'sum','PRIX':'sum'})
               .reset_index())
    daily['BENEFICE'] = daily['PRIX']
    daily['jour_semaine'] = pd.to_datetime(daily['jour']).dt.dayofweek
    daily['mois'] = pd.to_datetime(daily['jour']).dt.month

    X = daily[['jour_semaine', 'mois']]
    y = daily[['QUANTITE','BENEFICE','RESTANT A PAYER']]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(Xtr, ytr)

    return model, daily  # on renvoie aussi l'historique pour les graphiques

model_pred, df_daily_hist = train_prediction_model()





# ── Charge et entraîne une seule fois ────────────────────────────────────
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import date as _date

def train_prediction_model(csv_path="data/Transferts_complet.csv"):
    df = pd.read_csv(csv_path, sep=';')
    df['DATE DU TRANSFERT'] = pd.to_datetime(df['DATE DU TRANSFERT'], errors='coerce')
    df['jour'] = df['DATE DU TRANSFERT'].dt.date

    daily = (df.groupby('jour')
               .agg({'QUANTITE':'sum','MONTANT PAYER':'sum',
                     'RESTANT A PAYER':'sum','PRIX':'sum'})
               .reset_index())
    daily['BENEFICE'] = daily['PRIX']
    daily['jour_semaine'] = pd.to_datetime(daily['jour']).dt.dayofweek
    daily['mois'] = pd.to_datetime(daily['jour']).dt.month

    X = daily[['jour_semaine', 'mois']]
    y = daily[['QUANTITE','BENEFICE','RESTANT A PAYER']]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(Xtr, ytr)

    return model, daily  # on renvoie aussi l'historique pour les graphiques

model_pred, df_daily_hist = train_prediction_model()



@app.route('/')
def home():
    return render_template('home.html')

# Charger les données
DATA_PATH = "data/Transferts_classes.csv"
df = pd.read_csv(DATA_PATH, sep=';')
df["DATE DU TRANSFERT"] = pd.to_datetime(df["DATE DU TRANSFERT"], format="%d/%m/%Y %H:%M", errors="coerce")


# Charger les données
DATA_map = "data/ListeTransfert_geocode (2).csv"
df_geo = pd.read_csv(DATA_map, sep=';')
df_geo["DATE DU TRANSFERT"] = pd.to_datetime(df_geo["DATE DU TRANSFERT"], format="%d/%m/%Y %H:%M", errors="coerce")

# ── Filtres communs ───────────────────────────────────────────────────────
from typing import Dict
import pandas as pd

def filter_df(df_source: pd.DataFrame, form: Dict[str, str]) -> pd.DataFrame:
    """
    Retourne une copie filtrée de df_source selon les champs présents dans request.form.
    Acceptés : client, type_colis ou classe_colis, annee, mois, date_specifique.
    """
    df_out = df_source.copy()

    # Client
    if (client := form.get("client")) and client != "Tous":
        df_out = df_out[df_out["EXPEDITEUR"] == client]

    # Type ou classe de colis
    if (typ := form.get("type_colis") or form.get("classe_colis")) and typ != "Tous":
        col = "TYPE COLIS" if "type_colis" in form else "CLASSE_COLIS"
        df_out = df_out[df_out[col] == typ]

    # Année
    if (annee := form.get("annee")) and annee != "Tous":
        df_out = df_out[df_out["DATE DU TRANSFERT"].dt.year == int(annee)]

    # Mois
    if (mois := form.get("mois")) and mois != "Tous":
        df_out = df_out[df_out["DATE DU TRANSFERT"].dt.month == int(mois)]

    # Date précise
    if (date_str := form.get("date_specifique")):
        date_sel = pd.to_datetime(date_str, errors="coerce")
        df_out = df_out[df_out["DATE DU TRANSFERT"].dt.date == date_sel.date()]

     # Plage de dates
    if (date_debut := form.get("date_debut")) and (date_fin := form.get("date_fin")):
        try:
            debut = pd.to_datetime(date_debut)
            fin = pd.to_datetime(date_fin)
            df_out = df_out[(df_out["DATE DU TRANSFERT"] >= debut) & (df_out["DATE DU TRANSFERT"] <= fin)]
        except Exception as e:
            print("❌ Erreur de plage de dates :", e)

    return df_out


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if model_pred is None:
        return "<h3>⚠️ Modèle indisponible. Vérifiez le fichier joblib.</h3>"

    # Date cible : saisie dans le formulaire ou demain par défaut
    date_str = request.form.get("date_cible") \
               or str(_date.today() + pd.Timedelta(days=1))
    target = pd.to_datetime(date_str)

    # Construction des features
    features = pd.DataFrame([{
        "jour_semaine": target.dayofweek,
        "mois":         target.month
    }])

    # Prédiction
    pred = model_pred.predict(features)[0]
    pred_colis, pred_benef, pred_credit = \
        int(pred[0]), round(pred[1], 2), round(pred[2], 2)

    # Mini-graphiques historiques + point prédit
    df_hist = df_daily_hist.sort_values("jour").copy()  # df_daily_hist vient du dashboard
    fut_label = target.date().isoformat()

    def make_fig(col, title, value):
        aux = df_hist[["jour", col]].copy()
        aux.loc[len(aux)] = [fut_label, value]
        fig = px.line(aux, x="jour", y=col, title=title)
        fig.update_traces(mode="lines+markers")
        return pio.to_html(fig, full_html=False)

    graph_colis  = make_fig("QUANTITE",        "Historique quantité + prédiction", pred_colis)
    graph_benef  = make_fig("BENEFICE",        "Historique bénéfice + prédiction", pred_benef)
    graph_credit = make_fig("RESTANT A PAYER", "Historique crédit + prédiction",   pred_credit)





    return render_template(
        "prediction.html",
        selected_date=date_str,
        pred_colis=pred_colis,
        pred_benef=pred_benef,
        pred_credit=pred_credit,


        graph_colis=graph_colis,
        graph_benef=graph_benef,
        graph_credit=graph_credit
    )


# ────────────────────────────────────────────────────────────
#  ROUTE PRINCIPALE /dashboard (fusionne tout)
# ────────────────────────────────────────────────────────────
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():

    # ---------- 1. Récupération des filtres ---------------------------------
    selected_client   = request.form.get("client")
    selected_type     = request.form.get("type_colis")       # ou classe_colis selon ton HTML
    selected_annee    = request.form.get("annee")
    selected_mois     = request.form.get("mois")
    selected_date     = request.form.get("date_specifique")
    selected_date_debut = request.form.get("date_debut", "")
    selected_date_fin = request.form.get("date_fin", "")
    # ---------- 1. Filtres (inchangé) --------------------------------------
    df_filtered = filter_df(df, request.form)
    col_class = 'CLASSE_COLIS' if 'CLASSE_COLIS' in df_filtered.columns else 'TYPE COLIS'

   
    # ---------- 2. Variables pour les filtres (inchangé) -------------------
    clients = ["Tous"] + sorted(df["EXPEDITEUR"].dropna().unique())
    types   = ["Tous"] + sorted(df[col_class].dropna().unique())
    annees  = ["Tous"] + sorted(df["DATE DU TRANSFERT"].dt.year.dropna().unique().astype(str))
    mois    = ["Tous"] + [str(m).zfill(2) for m in sorted(df["DATE DU TRANSFERT"].dt.month.dropna().unique())]
    
    
       
    # =======================================================================
    # SECTION 1 : PERFORMANCE (déjà présent, juste pour le contexte)
    # =======================================================================

     # Taux de croissance sur le dernier mois
    df_monthly = df.groupby(df["DATE DU TRANSFERT"].dt.to_period("M"))["QUANTITE"].sum().reset_index()
    df_monthly["taux_croissance"] = df_monthly["QUANTITE"].pct_change() * 100
    dernier_taux = round(df_monthly["taux_croissance"].iloc[-1], 2) if len(df_monthly) > 1 else 0
      #taux de récurrence client  
    nb_total_clients = df["EXPEDITEUR"].nunique()
    nb_recurrents = df["EXPEDITEUR"].value_counts().gt(1).sum()
    taux_recurrence = round(nb_recurrents / nb_total_clients * 100, 2) if nb_total_clients > 0 else 0

    perf_kpi = {
        "volume_total"    : int(df_filtered["QUANTITE"].sum()),
        "nb_expeditions"  : len(df_filtered),
        "nb_types_colis"  : df_filtered[col_class].nunique(),
        "type_plus_frequent": (df_filtered[col_class].mode()[0]
                               if not df_filtered[col_class].empty else 'N/A'),
        "top_client_volume": df_filtered.groupby("EXPEDITEUR")["QUANTITE"].sum().idxmax() if not df_filtered.empty else "N/A",
        "jour_plus_charge": df_filtered["DATE DU TRANSFERT"].dt.date.value_counts().idxmax() if not df_filtered.empty else "N/A",
        "volume_moyen_jour": round(df_filtered.groupby(df_filtered["DATE DU TRANSFERT"].dt.date)["QUANTITE"].sum().mean(), 2) if not df_filtered.empty else 0,
        "taux_croissance": f"{dernier_taux:+.2f}",  # exemple : "+2.02"
        "taux_recurrence_client": taux_recurrence


    }
    # ... (les graphiques perf_g1, perf_g2, perf_g3 restent les mêmes)
     # ---------- Graphiques ----------
    # 1. Répartition du volume par type de colis
    df_volume_type = (
        df_filtered.groupby('CLASSE_COLIS')['QUANTITE']
        .sum().reset_index()
    )
    fig_perf1 = px.pie(
        df_volume_type, names='CLASSE_COLIS', values='QUANTITE',
        title="Répartition du volume par type de colis"
    )

    # 2. Top 10 clients par volume
    df_vol_client = (
        df_filtered.groupby('EXPEDITEUR')['QUANTITE']
        .sum().sort_values(ascending=False).head(10).reset_index()
    )
    fig_perf2 = px.bar(
        df_vol_client, x='EXPEDITEUR', y='QUANTITE',
        title="Top 10 clients – volume expédié"
    )

    # 3. Volume expédié dans le temps
    df_temps = df_filtered.copy()
    df_temps['DATE'] = df_temps['DATE DU TRANSFERT'].dt.date
    df_temps = df_temps.groupby('DATE')['QUANTITE'].sum().reset_index()
    fig_perf3 = px.line(df_temps, x='DATE', y='QUANTITE',
                   title="Évolution quotidienne des volumes")
    
    # 4. heatmap heure/Jour
    #heatmap_html = pio.to_html(fig_heatmap, full_html=False)

    # 4. Tableau taux de croissance mensuelle
        # Construire df_monthly depuis df_filtered (ou df principal filtré)
    df_filtered["mois"] = df_filtered["DATE DU TRANSFERT"].dt.to_period("M")
    df_monthly = df_filtered.groupby("mois")["QUANTITE"].sum().reset_index()
    df_monthly["taux_croissance"] = df_monthly["QUANTITE"].pct_change() * 100

    # Sécuriser la colonne "mois" en chaîne
    df_monthly["mois"] = df_monthly["mois"].astype(str)

    # Ajouter flèches
    def get_arrow(val):
        if pd.isna(val):
            return ""
        elif val > 0:
            return f"<span class='text-green-600 text-semibold text-center'>↗️ +{val:.2f}%</span>"
        elif val < 0:
            return f"<span class='text-red-600 text-semibold text-center'>↘️ {val:.2f}%</span>"
        else:
            return f"<span class='text-gray-500 text-semibold text-center'>→ 0%</span>"

    df_monthly["Évolution"] = df_monthly["taux_croissance"].apply(get_arrow)
    df_monthly["QUANTITE"] = df_monthly["QUANTITE"].astype(int)

    # Export HTML avec flèches
    growth_table_html = df_monthly.tail(6)[["mois", "QUANTITE", "Évolution"]].to_html(
        index=False,
        escape=False,
        classes="table-auto w-full text-md text-center text-gray-700",
        border=0
    )


    
    # =======================================================================
    # SECTION 2 : FINANCES 
    # =======================================================================

        # Taux d'encaissement global
    total_prix = df_filtered["PRIX"].sum()
    ca_total = df_filtered["MONTANT PAYER"].sum()
    restant_total = df_filtered["RESTANT A PAYER"].sum()

    top_client_ca = df_filtered.groupby("EXPEDITEUR")["MONTANT PAYER"].sum().idxmax()
    top_client_impaye = df_filtered.groupby("EXPEDITEUR")["RESTANT A PAYER"].sum().idxmax()

    df_filtered["Taux Impaye"] = (df_filtered["RESTANT A PAYER"] / df_filtered["PRIX"]).fillna(0)

    fin_kpi = {
        "ca_total": int(df_filtered["MONTANT PAYER"].sum()),
        "restant_total": int(df_filtered["RESTANT A PAYER"].sum()),
        "taux_encaissement": round(df_filtered["MONTANT PAYER"].sum() / df_filtered["PRIX"].sum() * 100, 2) if df_filtered["PRIX"].sum() > 0 else 0,
        "top_client_ca": top_client_ca,
        "top_client_impaye": top_client_impaye,
        "ca_moyen_expedition": int(df_filtered["MONTANT PAYER"].mean()) if len(df_filtered) > 0 else 0,
        "taux_impaye_moyen": round(df_filtered["Taux Impaye"].mean() * 100, 2)

    }
    # ... (les graphiques fin_g1, fin_g2, fin_g3 restent les mêmes)
     # ---------- Graphiques ----------
    top_ca = (
        df_filtered.groupby('EXPEDITEUR')['MONTANT PAYER']
        .sum().sort_values(ascending=False).head(10).reset_index()
    )
    fig_fin1 = px.bar(
        top_ca, x='EXPEDITEUR', y='MONTANT PAYER',
        title="Top 10 clients – chiffre d'affaires"
    )

    top_impaye = (
        df_filtered.groupby('EXPEDITEUR')['RESTANT A PAYER']
        .sum().sort_values(ascending=False).head(10).reset_index()
    )
    fig_fin2 = px.bar(
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
    fig_fin3 = px.line(
        df_month, x='Mois',
        y=['MONTANT PAYER', 'RESTANT A PAYER'],
        title="CA vs impayés (mensuel)"
    )
    # CA vs impayé mensuel
    df_filtered["mois"] = df_filtered["DATE DU TRANSFERT"].dt.to_period("M")
    df_mensuel = df_filtered.groupby("mois")[["MONTANT PAYER", "RESTANT A PAYER"]].sum().reset_index()
    df_mensuel["mois"] = df_mensuel["mois"].astype(str)
    fig_fin4 = px.bar(df_mensuel, x="mois", y=["MONTANT PAYER", "RESTANT A PAYER"], barmode="group", title="CA vs impayés (mensuel)")

    # =======================================================================
    # SECTION 3 : ANALYSE CLIENTS (fusion de la logique de /clients)
    # =======================================================================
    # --- KPIs Clients ---
    ca_par_client = df_filtered.groupby('EXPEDITEUR')['MONTANT PAYER'].sum().reset_index()
    livraisons_par_client = df_filtered['EXPEDITEUR'].value_counts().reset_index()
    livraisons_par_client.columns = ['Client', 'Nb Livraisons']
    
    top_client_ca_row = ca_par_client.sort_values(by='MONTANT PAYER', ascending=False).iloc[0] if not ca_par_client.empty else {'EXPEDITEUR': 'N/A', 'MONTANT PAYER': 0}
    top_client_liv_row = livraisons_par_client.iloc[0] if not livraisons_par_client.empty else {'Client': 'N/A', 'Nb Livraisons': 0}
    
    df_filtered['Taux Impaye'] = (df_filtered['RESTANT A PAYER'] / df_filtered['PRIX']).fillna(0)

    clients_kpi = {
        'nb_client': df_filtered['EXPEDITEUR'].nunique(),
        'top1_ca_nom': top_client_ca_row['EXPEDITEUR'],
        'top1_ca_valeur': int(top_client_ca_row['MONTANT PAYER']),
        'top1_liv_nom': top_client_liv_row['Client'],
        'top1_liv_valeur': int(top_client_liv_row['Nb Livraisons']),
        'taux_moyen_impaye': round(df_filtered.groupby('EXPEDITEUR')['Taux Impaye'].mean().mean() * 100, 2) if not df_filtered.empty else 0
    }

    # --- Graphes Clients ---
    top10_ca = ca_par_client.sort_values(by='MONTANT PAYER', ascending=False).head(10)
    clients_g1 = px.bar(top10_ca, x='EXPEDITEUR', y='MONTANT PAYER', title="Top 10 Clients par Chiffre d'Affaires")

    top10_livraisons = livraisons_par_client.head(10)
    clients_g2 = px.bar(top10_livraisons, x='Client', y='Nb Livraisons', title="Top 10 Clients par Nombre de Livraisons")
    
    clients_g3 = px.pie(top10_ca, names='EXPEDITEUR', values='MONTANT PAYER', title="Répartition du CA (Top 10)", hole=0.4)

    # =======================================================================
    # SECTION 4 : LOGISTIQUE & STOCK (fusion de la logique de /logistique)
    # =======================================================================
    # --- KPIs Logistique ---
    logistique_kpi = {
        'nb_expeditions': len(df_filtered),
        'volume_total': int(df_filtered['QUANTITE'].sum()),
        'nb_types_colis': df_filtered[col_class].nunique(),
        'type_plus_frequent': df_filtered[col_class].mode()[0] if not df_filtered.empty else 'N/A',
        'client_top_volume': df_filtered.groupby('EXPEDITEUR')['QUANTITE'].sum().idxmax() if not df_filtered.empty else 'N/A'
    }

    # --- Graphes Logistique ---
    df_volume_type = df_filtered.groupby(col_class)['QUANTITE'].sum().reset_index()
    logistique_g1 = px.pie(df_volume_type, names=col_class, values='QUANTITE', title="Répartition du Volume par Type de Colis")
    
    df_volume_clients = df_filtered.groupby('EXPEDITEUR')['QUANTITE'].sum().nlargest(10).reset_index()
    logistique_g2 = px.bar(df_volume_clients, x='EXPEDITEUR', y='QUANTITE', title="Top 10 Clients par Volume Expédié")
    
    df_volume_mensuel = df_filtered.set_index('DATE DU TRANSFERT').resample('M')['QUANTITE'].sum().reset_index()
    df_volume_mensuel['Mois'] = df_volume_mensuel['DATE DU TRANSFERT'].dt.strftime('%Y-%m')
    logistique_g3 = px.line(df_volume_mensuel, x='Mois', y='QUANTITE', title="Évolution Mensuelle des Volumes Expédiés")

    # ... (Tu peux ajouter ici la logique pour 'Tournées' et 'Alertes' de la même manière)
    # =======================================================================
    # SECTION 5 : OPTIMISATION DES TOURNEES
    # =======================================================================
    # --- Carte de toutes les livraisons (selon les filtres) ---
    df_map_filtered = filter_df(df_geo, request.form) # On utilise le df géocodé
    df_map_filtered = df_map_filtered.dropna(subset=['lat', 'lon'])

    fig_map = px.scatter_mapbox(
        df_map_filtered,
        lat='lat',
        lon='lon',
        hover_name='EXPEDITEUR',
        hover_data={'ADRESSES': True, 'DATE DU TRANSFERT': True, 'TYPE COLIS': True},
        color=col_class, # col_class a été défini au début de la fonction
        zoom=4,
        height=600,
        title="Carte interactive des livraisons filtrées"
    )
    fig_map.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":40,"l":0,"b":0})
    tournees_map = pio.to_html(fig_map, full_html=False)

    # --- Itinéraire optimisé pour une date spécifique ---
    # On utilise la date du filtre 'date_specifique'. Sinon, on prend la date la plus fréquente.
    selected_date_str = request.form.get("date_specifique")
    df_geo_valid = df_geo.dropna(subset=['lat', 'lon', 'DATE DU TRANSFERT'])
    df_geo_valid['DATE'] = df_geo_valid['DATE DU TRANSFERT'].dt.date

    if selected_date_str:
        target_date = pd.to_datetime(selected_date_str).date()
        date_title = f"le {target_date.strftime('%d/%m/%Y')}"
    else:
        target_date = df_geo_valid['DATE'].mode()[0] if not df_geo_valid.empty else None
        date_title = f"la date la plus fréquente ({target_date.strftime('%d/%m/%Y') if target_date else 'N/A'})"

    tournees_route = "<p class='text-center text-gray-500 mt-8'>Veuillez sélectionner une date spécifique pour calculer un itinéraire optimisé.</p>"
    if target_date:
        df_day = df_geo_valid[df_geo_valid['DATE'] == target_date].copy()

        # Fonction de calcul d'itinéraire (la même que tu avais)
        def compute_naive_route(df_route_calc):
            from geopy.distance import geodesic
            if df_route_calc.empty:
                return pd.DataFrame()
            
            visited = []
            remaining = df_route_calc.copy()
            current = remaining.iloc[0]
            visited.append(current)
            remaining = remaining.drop(current.name)
            
            while not remaining.empty:
                current_point = (current['lat'], current['lon'])
                distances = remaining.apply(lambda row: geodesic(current_point, (row['lat'], row['lon'])).km, axis=1)
                next_index = distances.idxmin()
                current = remaining.loc[next_index]
                visited.append(current)
                remaining = remaining.drop(next_index)
            return pd.DataFrame(visited)

        df_route = compute_naive_route(df_day)

        if not df_route.empty:
            # Ajout d'un numéro pour l'ordre de passage
            df_route['Ordre'] = range(1, len(df_route) + 1)
            
            fig_route = px.line_mapbox(
                df_route.sort_values('Ordre'),  # ✅ assure l'ordre
                lat='lat',
                lon='lon',
                text='Ordre',
                hover_name='EXPEDITEUR',
                hover_data={'ADRESSES': True, 'Ordre': True},
                color_discrete_sequence=["#facc15"],
                zoom=8,
                height=600,
                title=f"Itinéraire optimisé pour {date_title}"
            )

            fig_route.update_traces(textposition="top right", textfont=dict(color="black", size=12))
            fig_route.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":40,"l":0,"b":0})
            tournees_route = pio.to_html(fig_route, full_html=False)
        else:
            tournees_route = f"<p class='text-center text-gray-500 mt-8'>Aucune donnée géolocalisée trouvée pour {date_title}.</p>"


    # ---------- Envoi de TOUTES les données au template ---------------------
    return render_template(
        "dashboard.html",
        # Variables des filtres
        clients=clients, types=types, annees=annees, mois=mois,
        selected_client     = selected_client,
        selected_type       = selected_type,
        selected_annee      = selected_annee,
        selected_mois       = selected_mois,
        selected_date       = selected_date,

        # Finance 
        fin_kpi=fin_kpi,
        fin_g1=pio.to_html(fig_fin1, full_html=False),
        fin_g2=pio.to_html(fig_fin2, full_html=False),
        fin_g3=pio.to_html(fig_fin3, full_html=False),
        fin_g4=pio.to_html(fig_fin4, full_html=False),

        
        # ➜ NOUVEAU : Données pour l'onglet Clients
        clients_kpi=clients_kpi,
        clients_g1=pio.to_html(clients_g1, full_html=False),
        clients_g2=pio.to_html(clients_g2, full_html=False),
        clients_g3=pio.to_html(clients_g3, full_html=False),

        # ➜ NOUVEAU : Données pour l'onglet Logistique
        logistique_kpi=logistique_kpi,
        logistique_g1=pio.to_html(logistique_g1, full_html=False),
        logistique_g2=pio.to_html(logistique_g2, full_html=False),
        logistique_g3=pio.to_html(logistique_g3, full_html=False),
        
        # ... (n'oublie pas de passer aussi les variables perf et fin !)
        # --- variables pour les ONGLETs ---
        perf_kpi = perf_kpi,
        
        perf_g1  = pio.to_html(fig_perf1, full_html=False),
        perf_g2  = pio.to_html(fig_perf2, full_html=False),
        perf_g3  = pio.to_html(fig_perf3, full_html=False),
        
        #perf_heatmap=heatmap_html,
        growth_table=growth_table_html,



        # ➜ NOUVEAU : Données pour l'onglet Tournées
        tournees_map=tournees_map,
        tournees_route=tournees_route
    )

# Les routes /clients, /logistique, /tournees, etc. ne sont plus nécessaires. Tu peux les supprimer.

if __name__ == '__main__':
    app.run(debug=True)