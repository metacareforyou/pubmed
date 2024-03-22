FROM python:3.10 as build

WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt && pip install torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu
FROM build
WORKDIR /app
COPY .env /app/
COPY app.py /app
CMD python app.py