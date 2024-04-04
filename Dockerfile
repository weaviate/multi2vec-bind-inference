FROM python:3.11-slim

WORKDIR /app

RUN apt-get update
RUN apt-get -y install git libgomp1 wget
RUN pip install --upgrade --user pip setuptools

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

ENV PYTHONPATH="/app:/app/ImageBind"

RUN ./download.py

ENTRYPOINT ["/bin/sh", "-c"]
CMD ["uvicorn app:app --host 0.0.0.0 --port 8080"]
