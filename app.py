from flask import Flask, render_template_string
import pandas as pd
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

@app.route("/")
def dashboard():
    fichier = "data.csv"  # chemin vers votre fichier CSV
    try:
        df = pd.read_csv(fichier, sep=";", encoding="utf-8", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(fichier, sep=",", encoding="utf-8", on_bad_lines="skip")

    ca_client = df.groupby("EXPEDITEUR")["PRIX"].sum().reset_index()
    freq_envois = df.groupby("EXPEDITEUR")["REFERENCE"].count().reset_index()

    fig_ca = px.bar(ca_client, x="EXPEDITEUR", y="PRIX", title="CA par client")
    fig_freq = px.bar(freq_envois, x="EXPEDITEUR", y="REFERENCE", title="Fréquence des envois")

    html_ca = pio.to_html(fig_ca, full_html=False, include_plotlyjs='cdn')
    html_freq = pio.to_html(fig_freq, full_html=False, include_plotlyjs='cdn')

    page = """
    <html>
    <head><title>Mini Dashboard</title></head>
    <body>
        <h1>Dashboard Clients & Produits</h1>
        <h2>CA par client</h2>
        {{ html_ca|safe }}
        <h2>Fréquence des envois</h2>
        {{ html_freq|safe }}
    </body>
    </html>
    """
    return render_template_string(page, html_ca=html_ca, html_freq=html_freq)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
