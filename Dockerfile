# Étape 1 : Choisir une image Python légère
FROM python:3.11-slim

# Étape 2 : Définir le répertoire de travail
WORKDIR /app

# Étape 3 : Copier le code source dans l’image
COPY . /app

# Étape 4 : Éviter le cache pip et créer l’environnement virtuel
ENV PIP_NO_CACHE_DIR=yes

# Étape 5 : Installer les dépendances
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

# Étape 6 : Exposer le port (si Flask par exemple)
EXPOSE 5000

# Étape 7 : Démarrer l'application Flask
CMD ["/opt/venv/bin/python", "app.py"]
