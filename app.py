

# ===================================================================
# 1. IMPORTS ET CONFIGURATION
# ===================================================================
import os
from datetime import date as _date
from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.io as pio
from flask import Flask, render_template, request, jsonify
from flask_caching import Cache  # 👈 CORRECTION: Import manquant
from joblib import load
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Configuration du cache
config = {
    "DEBUG": True,
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 300  # 5 minutes
}
app.config.from_mapping(config)
cache = Cache(app)


# ===================================================================
# 2. CHARGEMENT DES DONNÉES ET MODÈLES (une seule fois au démarrage)
# ===================================================================
# --- Chargement des DataFrames ---
try:
    df = pd.read_csv("data/Transferts_classes.csv", sep=';')
    df["DATE DU TRANSFERT"] = pd.to_datetime(df["DATE DU TRANSFERT"], format="%d/%m/%Y %H:%M", errors="coerce")
    df_geo = pd.read_csv("data/ListeTransfert_geocode (2).csv", sep=';')
    df_geo["DATE DU TRANSFERT"] = pd.to_datetime(df_geo["DATE DU TRANSFERT"], format="%d/%m/%Y %H:%M", errors="coerce")
except FileNotFoundError as e:
    print(f"❌ Erreur: Fichier de données non trouvé. {e}")
    df, df_geo = pd.DataFrame(), pd.DataFrame()

# --- Entraînement/Chargement du modèle de prédiction ---
# NOTE: J'ai supprimé la définition de fonction dupliquée
def train_prediction_model(csv_path="data/Transferts_complet.csv"):
    # ... (le corps de ta fonction reste le même)
    df_model = pd.read_csv(csv_path, sep=';')
    df_model['DATE DU TRANSFERT'] = pd.to_datetime(df_model['DATE DU TRANSFERT'], errors='coerce')
    df_model['jour'] = df_model['DATE DU TRANSFERT'].dt.date
    daily = (df_model.groupby('jour')
               .agg({'QUANTITE':'sum','MONTANT PAYER':'sum', 'RESTANT A PAYER':'sum','PRIX':'sum'})
               .reset_index())
    daily['BENEFICE'] = daily['PRIX']
    daily['jour_semaine'] = pd.to_datetime(daily['jour']).dt.dayofweek
    daily['mois'] = pd.to_datetime(daily['jour']).dt.month
    X = daily[['jour_semaine', 'mois']]
    y = daily[['QUANTITE','BENEFICE','RESTANT A PAYER']]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(Xtr, ytr)
    return model, daily

model_pred, df_daily_hist = train_prediction_model()

# ===================================================================
# 3. FONCTIONS UTILITAIRES (comme le filtrage)
# ===================================================================
def filter_df(df_source: pd.DataFrame, form: Dict[str, str]) -> pd.DataFrame:
    # ... (ta fonction filter_df reste inchangée)
    df_out = df_source.copy()
    if (client := form.get("client")) and client != "Tous":
        df_out = df_out[df_out["EXPEDITEUR"] == client]
    if (typ := form.get("type_colis") or form.get("classe_colis")) and typ != "Tous":
        col = "TYPE COLIS" if "type_colis" in form else "CLASSE_COLIS"
        df_out = df_out[df_out[col] == typ]
    if (annee := form.get("annee")) and annee != "Tous":
        df_out = df_out[df_out["DATE DU TRANSFERT"].dt.year == int(annee)]
    if (mois := form.get("mois")) and mois != "Tous":
        df_out = df_out[df_out["DATE DU TRANSFERT"].dt.month == int(mois)]
    if (date_str := form.get("date_specifique")):
        date_sel = pd.to_datetime(date_str, errors="coerce")
        df_out = df_out[df_out["DATE DU TRANSFERT"].dt.date == date_sel.date()]
    if (date_debut := form.get("date_debut")) and (date_fin := form.get("date_fin")):
        try:
            debut = pd.to_datetime(date_debut)
            fin = pd.to_datetime(date_fin)
            df_out = df_out[(df_out["DATE DU TRANSFERT"] >= debut) & (df_out["DATE DU TRANSFERT"] <= fin)]
        except Exception as e:
            print(f"❌ Erreur de plage de dates : {e}")
    return df_out

# ===================================================================
# 4. FONCTIONS DE GÉNÉRATION DE DONNÉES (CACHÉES)
# ===================================================================
# Le décorateur @cache.memoize() fait la magie : il stocke le résultat
# de la fonction pour un jeu d'arguments donné.

@cache.memoize()
def generate_performance_data(filters_tuple):
    filters_dict = dict(filters_tuple)
    df_filtered = filter_df(df, filters_dict)
    col_class = 'CLASSE_COLIS' if 'CLASSE_COLIS' in df_filtered.columns else 'TYPE COLIS'
    # ... (Copie ici toute la logique de calcul des KPIs et graphiques de PERFORMANCE)

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
        "taux_recurrence_client": taux_recurrence,
        # "nb_retours" :df[df['RETOUR COLIS'] == 'Oui'].shape[0],
        # "moyenne_livreurs" : round(df['NB LIVREURS DISPONIBLES'].mean(), 1)


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

    #  5.Volume total expédié par livreur (Bar chart)
    # volume_livreur = df.groupby('LIVREUR')['VOLUME COLIS (m³)'].sum().reset_index()
    # fig_perf5 = px.bar(volume_livreur, x='LIVREUR', y='VOLUME COLIS (m³)', title='Volume total expédié par livreur')

    #6. Nombre d'expéditions par jour de livraison (Line chart)
    # df['DATE DE LIVRAISON'] = pd.to_datetime(df['DATE DE LIVRAISON'])
    # daily_deliveries = df.groupby(df['DATE DE LIVRAISON'].dt.date).size().reset_index(name='Nb Livraison')

    # fig_perf2 = px.line(daily_deliveries, x='DATE DE LIVRAISON', y='Nb Livraison', title='Nombre de livraisons par jour')

    return {
        "perf_kpi": perf_kpi,
        "perf_g1": pio.to_html(fig_perf1, full_html=False),
        "perf_g2" :pio.to_html(fig_perf2, full_html=False),
        "perf_g3" :pio.to_html(fig_perf3, full_html=False),
        "growth_table":growth_table_html,
        #"perf_g5" :pio.to_html(fig_perf5, full_html=False),
        #"perf_g6" :pio.to_html(fig_perf6, full_html=False),


        }


# =======================================================================
    # SECTION 2 : FINANCES 
# =======================================================================

@cache.memoize()
def generate_finance_data(filters_tuple):
    filters_dict = dict(filters_tuple)
    df_filtered = filter_df(df, filters_dict)
    col_class = 'CLASSE_COLIS' if 'CLASSE_COLIS' in df_filtered.columns else 'TYPE COLIS'
    # ... (Copie ici toute la logique de calcul des KPIs et graphiques de PERFORMANCE)

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
    return{
        "fin_kpi":fin_kpi,
        "fin_g1":pio.to_html(fig_fin1, full_html=False),
        "fin_g2":pio.to_html(fig_fin2, full_html=False),
        "fin_g3":pio.to_html(fig_fin3, full_html=False),
        "fin_g4":pio.to_html(fig_fin4, full_html=False),

    }
    
# Fais de même pour les autres sections : clients, logistique, tournees...
# =======================================================================
    # SECTION 3 : ANALYSE CLIENTS (fusion de la logique de /clients)
# =======================================================================

    
@cache.memoize()
def generate_clients_data(filters_tuple):
    filters_dict = dict(filters_tuple)
    df_filtered = filter_df(df, filters_dict)
    col_class = 'CLASSE_COLIS' if 'CLASSE_COLIS' in df_filtered.columns else 'TYPE COLIS'
    # ... (Copie ici toute la logique de calcul des KPIs et graphiques de PERFORMANCE)
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
    return {
        "clients_kpi":clients_kpi,
        "clients_g1":pio.to_html(clients_g1, full_html=False),
        "clients_g2":pio.to_html(clients_g2, full_html=False),
        "clients_g3":pio.to_html(clients_g3, full_html=False),

    }
# =======================================================================
    # SECTION 4 : LOGISTIQUE & STOCK (fusion de la logique de /logistique)
# =======================================================================
    
@cache.memoize()
def generate_logistique_data(filters_tuple):
    filters_dict = dict(filters_tuple)
    df_filtered = filter_df(df, filters_dict)
    col_class = 'CLASSE_COLIS' if 'CLASSE_COLIS' in df_filtered.columns else 'TYPE COLIS'
    
    # --- KPIs Logistique ---
    logistique_kpi = {
        'nb_expeditions': len(df_filtered),
        'volume_total': int(df_filtered['QUANTITE'].sum()),
        'nb_types_colis': df_filtered[col_class].nunique(),
        'type_plus_frequent': df_filtered[col_class].mode()[0] if not df_filtered.empty else 'N/A',
        'client_top_volume': df_filtered.groupby('EXPEDITEUR')['QUANTITE'].sum().idxmax() if not df_filtered.empty else 'N/A',
        # 'volume_livre' : df[df['STATUT EXPEDITION'] == 'Livré']['VOLUME COLIS (m³)'].sum()
    }

    # --- Graphes Logistique ---
    df_volume_type = df_filtered.groupby(col_class)['QUANTITE'].sum().reset_index()
    logistique_g1 = px.pie(df_volume_type, names=col_class, values='QUANTITE', title="Répartition du Volume par Type de Colis")
    
    df_volume_clients = df_filtered.groupby('EXPEDITEUR')['QUANTITE'].sum().nlargest(10).reset_index()
    logistique_g2 = px.bar(df_volume_clients, x='EXPEDITEUR', y='QUANTITE', title="Top 10 Clients par Volume Expédié")
    
    #Répartition des statuts d’expédition (Pie chart)
    # import plotly.express as px
    # logistique_g2 = px.pie(df, names='STATUT EXPEDITION', title='Répartition des statuts d’expédition')


    df_volume_mensuel = df_filtered.set_index('DATE DU TRANSFERT').resample('ME')['QUANTITE'].sum().reset_index()
    df_volume_mensuel['Mois'] = df_volume_mensuel['DATE DU TRANSFERT'].dt.strftime('%Y-%m')
    logistique_g3 = px.line(df_volume_mensuel, x='Mois', y='QUANTITE', title="Évolution Mensuelle des Volumes Expédiés")
    return{
        "logistique_kpi":logistique_kpi,
        "logistique_g1":pio.to_html(logistique_g1, full_html=False),
        "logistique_g2":pio.to_html(logistique_g2, full_html=False),
        "logistique_g3":pio.to_html(logistique_g3, full_html=False),
        
    }

# =======================================================================
    # SECTION 5 : OPTIMISATION DES TOURNEES
# =======================================================================
@cache.memoize()
def generate_tournees_data(filters_tuple):
    filters_dict = dict(filters_tuple)
    df_filtered = filter_df(df, filters_dict)
    col_class = 'CLASSE_COLIS' if 'CLASSE_COLIS' in df_filtered.columns else 'TYPE COLIS'
    
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
    return{
        "tournees_map":tournees_map,
        "tournees_route":tournees_route
    }


@cache.memoize()
def generate_alertes_data(filters_tuple):
    filters_dict = dict(filters_tuple)
    df_filtered = filter_df(df, filters_dict)
    col_class = 'CLASSE_COLIS' if 'CLASSE_COLIS' in df_filtered.columns else 'TYPE COLIS'
    # ... (Copie ici toute la logique de calcul des KPIs et graphiques de PERFORMANCE)
    
    #les graphiques
    # motif_counts = df[df['RETOUR COLIS'] == 'Oui']['MOTIF RETOUR'].value_counts().reset_index()
    # motif_counts.columns = ['Motif', 'Nombre']

    # fig_g1 = px.bar(motif_counts, x='Motif', y='Nombre', title='Motifs de retour des colis')
    # return{
    #     "alertes_g1":pio.to_html(fig_g1, full_html=False),}
    

# ===================================================================
# 5. ROUTES FLASK
# ===================================================================
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    """
    Cette route est maintenant beaucoup plus simple. Son rôle est de:
    1. Récupérer les listes pour les menus déroulants des filtres.
    2. Appeler les fonctions cachées pour obtenir les données.
    3. Envoyer le tout au template.
    """
    # Créer un tuple des filtres pour le passer aux fonctions cachées
    filters_for_cache = tuple(request.form.items())
    
    # Appeler les fonctions qui vont soit calculer, soit récupérer depuis le cache
    performance_data = generate_performance_data(filters_for_cache)
    finance_data = generate_finance_data(filters_for_cache)
    clients_data = generate_clients_data(filters_for_cache)
    logistique_data = generate_logistique_data(filters_for_cache)
    tournees_data = generate_tournees_data(filters_for_cache)

    # Récupérer les listes pour les menus déroulants (peut aussi être mis en cache !)
    col_class = 'CLASSE_COLIS' if 'CLASSE_COLIS' in df.columns else 'TYPE COLIS'
    clients = ["Tous"] + sorted(df["EXPEDITEUR"].dropna().unique())
    types   = ["Tous"] + sorted(df[col_class].dropna().unique())
    annees  = ["Tous"] + sorted(df["DATE DU TRANSFERT"].dt.year.dropna().unique().astype(str))
    mois    = ["Tous"] + [str(m).zfill(2) for m in sorted(df["DATE DU TRANSFERT"].dt.month.dropna().unique())]
    
    return render_template(
        "dashboard.html",
        # Listes pour les filtres
        clients=clients, types=types, annees=annees, mois=mois,
        
        # Valeurs sélectionnées pour les filtres
        selected_client=request.form.get("client", "Tous"),
        selected_type=request.form.get("type_colis", "Tous"),
        selected_annee=request.form.get("annee", "Tous"),
        selected_mois=request.form.get("mois", "Tous"),
        selected_date=request.form.get("date_specifique", ""),
        selected_date_debut=request.form.get("date_debut", ""),
        selected_date_fin=request.form.get("date_fin", ""),
        
        # Données des onglets (on utilise ** pour "déballer" les dictionnaires)
        **performance_data,
        **finance_data,
        **clients_data,
        **logistique_data,
        **tournees_data,
      
    )
    
# La route /prediction reste inchangée
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

if __name__ == '__main__':
    app.run(debug=True)