FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y locales && \
    locale-gen fr_FR.UTF-8 && \
    update-locale LANG=fr_FR.UTF-8

ENV LANG fr_FR.UTF-8
ENV LANGUAGE fr_FR:fr
ENV LC_ALL fr_FR.UTF-8

WORKDIR /app
COPY . .
RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]

