FROM python:3.10-slim

# Crée un répertoire pour l'app
WORKDIR /app

# Copie les fichiers nécessaires
COPY . .

# Installe les dépendances
RUN pip install --upgrade pip && pip install -r requirements.txt

# Port exposé
EXPOSE 5000

# Commande de démarrage
CMD ["python", "app.py"]
