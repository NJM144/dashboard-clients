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

df = df_geo

def train_prediction_model(csv_path="data/Transferts_complet.csv"):
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
# 3. FONCTIONS UTILITAIRES
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
# EXTRACT DIRECTIONS TEXT UTILITY
# ===================================================================
def extract_directions_text(route_data):
    """
    Extrait les instructions étape par étape du trajet Google Directions.
    Retourne une chaîne HTML avec les rues, distances et durées.
    """
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
# 4. OPTIMISATION DES TOURNEES
# ===================================================================
@cache.memoize()
def generate_tournees_data(filters_tuple):
    import json
    filters_dict = dict(filters_tuple)
    df_filtered = filter_df(df, filters_dict)
    col_class = 'CLASSE_COLIS' if 'CLASSE_COLIS' in df_filtered.columns else 'TYPE COLIS'
    
    # --- Carte de toutes les livraisons (selon les filtres) ---
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

    # --- Itinéraire optimisé pour une date spécifique ---
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
    directions_text = ""  # Toujours présent

    if target_date:
        df_day = df_geo_valid[df_geo_valid['DATE'] == target_date].copy()
        waypoints = df_day[["lat", "lon"]].to_dict(orient="records")

        # Appel API Google Directions
        route_data = get_google_directions_route(START_POINT, waypoints)
        print("🧭 Données retournées par Google Directions API :")
        print(json.dumps(route_data, indent=2) if route_data else "Aucune donnée route_data")
        if route_data:
            steps = route_data["legs"]
            route_coords = []
            directions_text = extract_directions_text(route_data)  # ✅ Utilisation dynamique
            for leg in steps:
                start_loc = leg["start_location"]
                route_coords.append((start_loc["lat"], start_loc["lng"]))
            # Dernier point
            route_coords.append((steps[-1]["end_location"]["lat"], steps[-1]["end_location"]["lng"]))

            df_route = pd.DataFrame(route_coords, columns=["lat", "lon"])
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
            directions_text = "<p class='text-red-600'>❌ Aucune instruction disponible.</p>"
            tournees_route = f"<p class='text-center text-red-600 mt-8'>Erreur lors de la récupération de l’itinéraire pour {date_title}.</p>"
    else:
        directions_text = "<p class='text-gray-500'>Aucune date sélectionnée ou donnée disponible.</p>"

    # Préparer les points (waypoints) pour Google Maps JS
    if target_date and not df_day.empty:
        waypoints_js = [
            {"location": f"{row['lat']},{row['lon']}", "stopover": True}
            for _, row in df_day.iterrows()
        ]
    else:
        waypoints_js = []

    start_js = f"{START_POINT['lat']},{START_POINT['lon']}"
    waypoints_json = json.dumps(waypoints_js)

    print("✅ directions_text =", directions_text[:200])  # Pour debug

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
    # Placeholders : tu peux ajouter les autres sections ici si besoin
    tournees_data = generate_tournees_data(filters_for_cache)
    dates_disponibles = sorted(df_geo["DATE DU TRANSFERT"].dt.date.dropna().unique())
    return render_template(
        "dashboard.html",
        dates_disponibles=dates_disponibles,
        selected_date=request.form.get("date_specifique", ""),
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
