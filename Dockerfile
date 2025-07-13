FROM python:3.10-slim

WORKDIR /app

# Copie ton code dans l'image
COPY . .

# Installe les dépendances Python, dont Babel
RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
