# ===================================================================
# 1. IMPORTS ET CONFIGURATION
# ===================================================================
import os
from datetime import date as _date
from typing import Dict
import requests
import pandas as pd
import plotly.express as px
import plotly.io as pio
from flask import Flask, render_template, request, jsonify
from flask_caching import Cache
from joblib import load
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import polyline
# Point de départ : Tour Eiffel
START_POINT = {"lat": 48.8584, "lon": 2.2945}
GOOGLE_MAPS_API_KEY = "AIzaSyBGlGZg7QgWNMaK9E901QUV7lp4srXO25A"

def get_google_directions_route(start, waypoints):
    """
    Appelle l'API Google Directions pour optimiser l'ordre des livraisons.
    """
    base_url = "https://maps.googleapis.com/maps/api/directions/json"
    origin = f"{start['lat']},{start['lon']}"
    destination = origin  # boucle fermée

    wp_str = "|".join([f"{p['lat']},{p['lon']}" for p in waypoints])
    waypoints_str = f"optimize:true|{wp_str}"

    params = {
        "origin": origin,
        "destination": destination,
        "waypoints": waypoints_str,
        "key": GOOGLE_MAPS_API_KEY
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if data["status"] == "OK":
        return data["routes"][0]
    else:
        print("❌ Erreur Google Directions:", data.get("status"), data.get("error_message"))
        return None

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
try:
    df = pd.read_csv("data/Transferts_classes.csv", sep=';')
    df["DATE DU TRANSFERT"] = pd.to_datetime(df["DATE DU TRANSFERT"], format="%d/%m/%Y %H:%M", errors="coerce")
    df_geo = pd.read_csv("data/ListeTransfert_geocode (2).csv", sep=';')
    df_geo["DATE DU TRANSFERT"] = pd.to_datetime(df_geo["DATE DU TRANSFERT"], format="%d/%m/%Y %H:%M", errors="coerce")
except FileNotFoundError as e:
    print(f"❌ Erreur: Fichier de données non trouvé. {e}")
    df, df_geo = pd.DataFrame(), pd.DataFrame()
    
# ===================================================================
# FONCTION DE FILTRAGE (UTILITAIRE)
# ===================================================================
def filter_df(df_source: pd.DataFrame, form: Dict[str, str]) -> pd.DataFrame:
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
# FONCTION PERFORMANCE
# ===================================================================
@cache.memoize()
def generate_performance_data(filters_tuple):
    filters_dict = dict(filters_tuple)
    df_filtered = filter_df(df, filters_dict)
    col_class = 'CLASSE_COLIS' if 'CLASSE_COLIS' in df_filtered.columns else 'TYPE COLIS'
    df_monthly = df.groupby(df["DATE DU TRANSFERT"].dt.to_period("M"))["QUANTITE"].sum().reset_index()
    df_monthly["taux_croissance"] = df_monthly["QUANTITE"].pct_change() * 100
    dernier_taux = round(df_monthly["taux_croissance"].iloc[-1], 2) if len(df_monthly) > 1 else 0
    nb_total_clients = df["EXPEDITEUR"].nunique()
    nb_recurrents = df["EXPEDITEUR"].value_counts().gt(1).sum()
    taux_recurrence = round(nb_recurrents / nb_total_clients * 100, 2) if nb_total_clients > 0 else 0

    perf_kpi = {
        "volume_total": int(df_filtered["QUANTITE"].sum()),
        "nb_expeditions": len(df_filtered),
        "nb_types_colis": df_filtered[col_class].nunique(),
        "type_plus_frequent": (df_filtered[col_class].mode()[0] if not df_filtered[col_class].empty else 'N/A'),
        "top_client_volume": df_filtered.groupby("EXPEDITEUR")["QUANTITE"].sum().idxmax() if not df_filtered.empty else "N/A",
        "jour_plus_charge": df_filtered["DATE DU TRANSFERT"].dt.date.value_counts().idxmax() if not df_filtered.empty else "N/A",
        "volume_moyen_jour": round(df_filtered.groupby(df_filtered["DATE DU TRANSFERT"].dt.date)["QUANTITE"].sum().mean(), 2) if not df_filtered.empty else 0,
        "taux_croissance": f"{dernier_taux:+.2f}",
        "taux_recurrence_client": taux_recurrence,
    }
    df_volume_type = (
        df_filtered.groupby('CLASSE_COLIS')['QUANTITE']
        .sum().reset_index()
    )
    fig_perf1 = px.pie(
        df_volume_type, names='CLASSE_COLIS', values='QUANTITE',
        title="Répartition du volume par type de colis"
    )
    df_vol_client = (
        df_filtered.groupby('EXPEDITEUR')['QUANTITE']
        .sum().sort_values(ascending=False).head(10).reset_index()
    )
    fig_perf2 = px.bar(
        df_vol_client, x='EXPEDITEUR', y='QUANTITE',
        title="Top 10 clients – volume expédié"
    )
    df_temps = df_filtered.copy()
    df_temps['DATE'] = df_temps['DATE DU TRANSFERT'].dt.date
    df_temps = df_temps.groupby('DATE')['QUANTITE'].sum().reset_index()
    fig_perf3 = px.line(df_temps, x='DATE', y='QUANTITE',
                   title="Évolution quotidienne des volumes")
    return {
        "perf_kpi": perf_kpi,
        "perf_g1": pio.to_html(fig_perf1, full_html=False),
        "perf_g2": pio.to_html(fig_perf2, full_html=False),
        "perf_g3": pio.to_html(fig_perf3, full_html=False)
    }

# ===================================================================
# FONCTION FINANCE
# ===================================================================
@cache.memoize()
def generate_finance_data(filters_tuple):
    filters_dict = dict(filters_tuple)
    df_filtered = filter_df(df, filters_dict)
    col_class = 'CLASSE_COLIS' if 'CLASSE_COLIS' in df_filtered.columns else 'TYPE COLIS'
    total_prix = df_filtered["PRIX"].sum()
    ca_total = df_filtered["MONTANT PAYER"].sum()
    restant_total = df_filtered["RESTANT A PAYER"].sum()
    top_client_ca = df_filtered.groupby("EXPEDITEUR")["MONTANT PAYER"].sum().idxmax() if not df_filtered.empty else "N/A"
    top_client_impaye = df_filtered.groupby("EXPEDITEUR")["RESTANT A PAYER"].sum().idxmax() if not df_filtered.empty else "N/A"
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
    return {
        "fin_kpi": fin_kpi,
        "fin_g1": pio.to_html(fig_fin1, full_html=False),
        "fin_g2": pio.to_html(fig_fin2, full_html=False),
        "fin_g3": pio.to_html(fig_fin3, full_html=False)
    }

# ===================================================================
# FONCTION CLIENTS
# ===================================================================
@cache.memoize()
def generate_clients_data(filters_tuple):
    filters_dict = dict(filters_tuple)
    df_filtered = filter_df(df, filters_dict)
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
    top10_ca = ca_par_client.sort_values(by='MONTANT PAYER', ascending=False).head(10)
    clients_g1 = px.bar(top10_ca, x='EXPEDITEUR', y='MONTANT PAYER', title="Top 10 Clients par Chiffre d'Affaires")
    top10_livraisons = livraisons_par_client.head(10)
    clients_g2 = px.bar(top10_livraisons, x='Client', y='Nb Livraisons', title="Top 10 Clients par Nombre de Livraisons")
    clients_g3 = px.pie(top10_ca, names='EXPEDITEUR', values='MONTANT PAYER', title="Répartition du CA (Top 10)", hole=0.4)
    return {
        "clients_kpi": clients_kpi,
        "clients_g1": pio.to_html(clients_g1, full_html=False),
        "clients_g2": pio.to_html(clients_g2, full_html=False),
        "clients_g3": pio.to_html(clients_g3, full_html=False),
    }

# ===================================================================
# FONCTION LOGISTIQUE
# ===================================================================
@cache.memoize()
def generate_logistique_data(filters_tuple):
    filters_dict = dict(filters_tuple)
    df_filtered = filter_df(df, filters_dict)
    col_class = 'CLASSE_COLIS' if 'CLASSE_COLIS' in df_filtered.columns else 'TYPE COLIS'
    logistique_kpi = {
        'nb_expeditions': len(df_filtered),
        'volume_total': int(df_filtered['QUANTITE'].sum()),
        'nb_types_colis': df_filtered[col_class].nunique(),
        'type_plus_frequent': df_filtered[col_class].mode()[0] if not df_filtered.empty else 'N/A',
        'client_top_volume': df_filtered.groupby('EXPEDITEUR')['QUANTITE'].sum().idxmax() if not df_filtered.empty else 'N/A',
    }
    df_volume_type = df_filtered.groupby(col_class)['QUANTITE'].sum().reset_index()
    logistique_g1 = px.pie(df_volume_type, names=col_class, values='QUANTITE', title="Répartition du Volume par Type de Colis")
    df_volume_clients = df_filtered.groupby('EXPEDITEUR')['QUANTITE'].sum().nlargest(10).reset_index()
    logistique_g2 = px.bar(df_volume_clients, x='EXPEDITEUR', y='QUANTITE', title="Top 10 Clients par Volume Expédié")
    df_volume_mensuel = df_filtered.set_index('DATE DU TRANSFERT').resample('ME')['QUANTITE'].sum().reset_index()
    df_volume_mensuel['Mois'] = df_volume_mensuel['DATE DU TRANSFERT'].dt.strftime('%Y-%m')
    logistique_g3 = px.line(df_volume_mensuel, x='Mois', y='QUANTITE', title="Évolution Mensuelle des Volumes Expédiés")
    return {
        "logistique_kpi": logistique_kpi,
        "logistique_g1": pio.to_html(logistique_g1, full_html=False),
        "logistique_g2": pio.to_html(logistique_g2, full_html=False),
        "logistique_g3": pio.to_html(logistique_g3, full_html=False),
    }

# ===================================================================
# UTILITAIRE POUR LES ÉTAPES GOOGLE DIRECTIONS
# ===================================================================
def extract_directions_text(route_data):
    steps_html = []
    legs = route_data.get("legs", [])
    for leg in legs:
        for step in leg.get("steps", []):
            instr = step.get("html_instructions", "")
            distance = step.get("distance", {}).get("text", "")
            duration = step.get("duration", {}).get("text", "")
            steps_html.append(f"<li>{instr} <span class='text-gray-500'>({distance}, {duration})</span></li>")
    return "<ol class='list-decimal list-inside space-y-1'>" + "\n".join(steps_html) + "</ol>"

# ===================================================================
# OPTIMISATION DES TOURNEES
# ===================================================================
@cache.memoize()
def generate_tournees_data(filters_tuple):
    import json
    filters_dict = dict(filters_tuple)
    df_filtered = filter_df(df, filters_dict)
    col_class = 'CLASSE_COLIS' if 'CLASSE_COLIS' in df_filtered.columns else 'TYPE COLIS'
    df_map_filtered = filter_df(df_geo, request.form)
    df_map_filtered = df_map_filtered.dropna(subset=['lat', 'lon'])
    df_map_filtered['DATE DU TRANSFERT'] = pd.to_datetime(df_map_filtered['DATE DU TRANSFERT'], errors='coerce')
    df_map_filtered['DATE_STR'] = df_map_filtered['DATE DU TRANSFERT'].dt.strftime('%Y-%m-%d')
    
    fig_map = px.scatter_mapbox(
        df_map_filtered,
        lat='lat',
        lon='lon',
        hover_name='EXPEDITEUR',
        hover_data={
            'ADRESSES': True,
            'DATE DU TRANSFERT': True,
            'TYPE COLIS': True
        },
        color='DATE_STR',
        zoom=4,
        height=600,
        title="Carte interactive des livraisons par date"
    )
    fig_map.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":40,"l":0,"b":0})
    tournees_map = pio.to_html(fig_map, full_html=False)
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
    directions_text = ""
    # Cas où on a une date cible
    if target_date:
        df_day = df_geo_valid[df_geo_valid['DATE'] == target_date].copy()
        if not df_day.empty and len(df_day) > 1:
            waypoints = df_day[["lat", "lon"]].to_dict(orient="records")
            route_data = get_google_directions_route(START_POINT, waypoints)
            print("🧭 Données retournées par Google Directions API :")
            print(json.dumps(route_data, indent=2) if route_data else "Aucune donnée route_data")
            if route_data:
              directions_text = extract_directions_text(route_data)
              print("=== TEST AFFICHAGE DEBUG ===")
              print(directions_text)
              
              overview_polyline = route_data.get("overview_polyline", {}).get("points")
              if overview_polyline:
                    decoded_points = polyline.decode(overview_polyline)
                    df_route = pd.DataFrame(decoded_points, columns=["lat", "lon"])
                    fig_route = px.line_mapbox(
                        df_route,
                        lat='lat',
                        lon='lon',
                        zoom=10,
                        height=600,
                        title=f"Trajet optimisé via Google Maps – {date_title}"
                    )
                    fig_route.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":40,"l":0,"b":0})
                    tournees_route = pio.to_html(fig_route, full_html=False)
              else:
                    tournees_route = "<p class='text-red-600'>Impossible d'obtenir le tracé précis de la route.</p>"
            else:
                directions_text = "<p class='text-red-600'>❌ Aucune instruction disponible (Google API n'a rien retourné).</p>"
        else:
            directions_text = "<p class='text-gray-500'>Pas assez de points pour calculer une tournée ce jour-là.</p>"
    else:
        directions_text = "<p class='text-gray-500'>Aucune date sélectionnée ou donnée disponible.</p>"

    # Points JS pour carte Google si besoin
    if target_date and not df_day.empty:
        waypoints_js = [
            {"location": f"{row['lat']},{row['lon']}", "stopover": True}
            for _, row in df_day.iterrows()
        ]
    else:
        waypoints_js = []
    start_js = f"{START_POINT['lat']},{START_POINT['lon']}"
    waypoints_json = json.dumps(waypoints_js)
    print("✅ directions_text =", directions_text[:200])
    print('====== directions_text FINAL ======')
    print(directions_text)
    return {
        "tournees_map": tournees_map,
        "tournees_route": tournees_route,
        "waypoints_json": waypoints_json,
        "start_point": start_js,
        "directions_text": directions_text
    }

# ===================================================================
# 5. ROUTES FLASK
# ===================================================================
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    filters_for_cache = tuple(request.form.items())
    performance_data = generate_performance_data(filters_for_cache)
    finance_data = generate_finance_data(filters_for_cache)
    clients_data = generate_clients_data(filters_for_cache)
    logistique_data = generate_logistique_data(filters_for_cache)
    tournees_data = generate_tournees_data(filters_for_cache)
    dates_disponibles = sorted(df_geo["DATE DU TRANSFERT"].dt.date.dropna().unique())
    return render_template(
        "dashboard.html",
        dates_disponibles=dates_disponibles,
        selected_date=request.form.get("date_specifique", ""),
        **performance_data,
        **finance_data,
        **clients_data,
        **logistique_data,
        **tournees_data
    )

@app.route("/tournees", methods=["GET", "POST"])
def tournees():
    filters_for_cache = tuple(request.form.items())
    tournees_data = generate_tournees_data(filters_for_cache)
    dates_disponibles = sorted(df_geo["DATE DU TRANSFERT"].dt.date.dropna().unique())
    return render_template(
        "tournee.html",
        dates_disponibles=dates_disponibles,
        selected_date=request.form.get("date_specifique", ""),
        **tournees_data
    )

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if model_pred is None:
        return "<h3>⚠️ Modèle indisponible. Vérifiez le fichier joblib.</h3>"
    date_str = request.form.get("date_cible") or str(_date.today() + pd.Timedelta(days=1))
    target = pd.to_datetime(date_str)
    features = pd.DataFrame([{
        "jour_semaine": target.dayofweek,
        "mois":         target.month
    }])
    pred = model_pred.predict(features)[0]
    pred_colis, pred_benef, pred_credit = int(pred[0]), round(pred[1], 2), round(pred[2], 2)
    df_hist = df_daily_hist.sort_values("jour").copy()
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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
