from flask import Flask, render_template, jsonify
import pandas as pd
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

@app.route("/")
def dashboard():
    fichier = "ListeTransfertdu_2025-06-01_au_2025-06-25.csv"
    try:
        df = pd.read_csv(fichier, sep=";", encoding="utf-8", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(fichier, sep=",", encoding="utf-8", on_bad_lines="skip")

    # Données pour les graphiques
    ca_client = df.groupby("EXPEDITEUR")["PRIX"].sum().reset_index()
    freq_envois = df.groupby("EXPEDITEUR")["REFERENCE"].count().reset_index()
    colis_type = df["TYPE COLIS"].value_counts().reset_index()
    colis_type.columns = ["Type", "Quantité"]

    # Graphiques Plotly
    fig_ca = px.bar(ca_client, x="EXPEDITEUR", y="PRIX", title="💰 CA par client")
    fig_freq = px.bar(freq_envois, x="EXPEDITEUR", y="REFERENCE", title="📦 Fréquence des envois")
    fig_colis = px.pie(colis_type, names="Type", values="Quantité", title="📦 Consommation de cartons")

    html_ca = pio.to_html(fig_ca, full_html=False, include_plotlyjs='cdn', div_id="ca_plot")
    html_freq = pio.to_html(fig_freq, full_html=False, include_plotlyjs='cdn', div_id="freq_plot")
    html_colis = pio.to_html(fig_colis, full_html=False, include_plotlyjs='cdn', div_id="colis_plot")

    page = """
    <html>
    <head>
        <title>Dashboard Clients & Produits</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <h1>Dashboard Clients & Produits</h1>

        <h2>💰 CA par client (cliquable)</h2>
        {{ html_ca | safe }}
        <div id="details_ca"></div>

        <h2>📦 Fréquence des envois</h2>
        {{ html_freq | safe }}

        <h2>📦 Consommation de cartons</h2>
        {{ html_colis | safe }}

        <script>
            const ca_plot = document.getElementById('ca_plot');
            ca_plot.on('plotly_click', function(data){
                const client = data.points[0].x;
                fetch(`/details/${client}`)
                    .then(res => res.json())
                    .then(info => {
                        let details = `<h3>Détails pour ${info.client}</h3>
                            <p>📦 Nombre d'envois : ${info.total_envois}</p>
                            <p>💰 CA total : ${info.ca_total} €</p>
                            <p>📅 Dernier envoi : ${info.dernier_envoi}</p>
                            <p>🗂️ Types de colis :<ul>`;
                        for (let type in info.types_colis) {
                            details += `<li>${type} → ${info.types_colis[type]}</li>`;
                        }
                        details += `</ul></p>`;
                        document.getElementById("details_ca").innerHTML = details;
                    });
            });
        </script>
    </body>
    </html>
    """
    return render_template("index.html", html_ca=html_ca, html_freq=html_freq, html_colis=html_colis)


@app.route("/details/<client>")
def detail_client(client):
    try:
        df = pd.read_csv("ListeTransfertdu_2025-06-01_au_2025-06-25.csv", sep=";", encoding="utf-8", on_bad_lines="skip")
    except:
        df = pd.read_csv("ListeTransfertdu_2025-06-01_au_2025-06-25.csv", sep=",", encoding="utf-8", on_bad_lines="skip")

    sous_df = df[df["EXPEDITEUR"] == client]
    total = sous_df["PRIX"].sum()
    count = len(sous_df)
    last_date = sous_df["DATE DU TRANSFERT"].max()
    colis_types = sous_df["TYPE COLIS"].value_counts().to_dict()

    return jsonify({
        "client": client,
        "total_envois": int(count),
        "ca_total": float(total),
        "dernier_envoi": last_date,
        "types_colis": colis_types
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
