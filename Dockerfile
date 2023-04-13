FROM python:3.8-slim-buster

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt && pip install -e .
EXPOSE 5000

ENV FLASK_APP /app/src/run_server.py

CMD ["flask", "run", "--host", "0.0.0.0"]
