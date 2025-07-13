FROM python:3.10-slim

# Crée un répertoire pour l'app
WORKDIR /app

# Copie les fichiers nécessaires
COPY . .
RUN apt-get update && \
    apt-get install -y locales && \
    locale-gen fr_FR.UTF-8 && \
    update-locale
ENV LANG fr_FR.UTF-8
ENV LANGUAGE fr_FR:fr
ENV LC_ALL fr_FR.UTF-8

# Installe les dépendances
RUN pip install --upgrade pip && pip install -r requirements.txt

# Port exposé
EXPOSE 5000

# Commande de démarrage
CMD ["python", "app.py"]
