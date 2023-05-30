FROM python:3.10.10

COPY requirements.txt .
COPY cuda.py .
COPY predict.py .
COPY model_docker.py .
COPY predict.proto .
COPY categories.csv .
COPY checkpoints ./checkpoints

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir

RUN python -m grpc_tools.protoc --python_out=. --grpc_python_out=. -I. predict.proto
