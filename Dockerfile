# 1. Image de base officielle Python (léger, rapide)
FROM python:3.11-slim

# 2. Répertoire de travail dans le conteneur
WORKDIR /app

# 3. Copier le contenu local dans l'image
COPY . /app

# 4. Désactiver le cache pip pour éviter les conflits
ENV PIP_NO_CACHE_DIR=yes

# 5. Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# 6. Exposer le port Flask par défaut
EXPOSE 5050

# 7. Démarrer l'application Flask
CMD ["python", "app.py"]
