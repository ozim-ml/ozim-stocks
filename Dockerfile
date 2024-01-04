FROM python:3
RUN python3 -m venv /opt/venv
COPY requirements.txt .
RUN . /opt/venv/bin/activate && pip install -r requirements.txt
COPY . /app
CMD . /opt/venv/bin/activate && ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

