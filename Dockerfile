FROM python:3.11-slim

LABEL maintainer="dimdasci <dimds@fastmail.com>"

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1

WORKDIR "/app"

COPY src/ src/
COPY models/ models/

COPY requirements.txt ./requirements.txt
COPY params.yaml ./params.yaml
COPY setup.py ./setup.py

RUN pip install -U pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["src.api.main:app", "--host", "0.0.0.0"]