FROM python:3.10.10

COPY requirements.txt .
COPY cuda.py .
COPY server.py .
COPY inference.py .
COPY categories.csv .
COPY food-sg-233-empower-food-matched.csv .
COPY checkpoints ./checkpoints

RUN apt-get update
RUN apt-get install -y ffmpeg libsm6 libxext6
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir

EXPOSE 5000

CMD ["python", "server.py"]
