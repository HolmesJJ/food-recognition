import cv2
import grpc
import glob
import json
import logging
import numpy as np
import pandas as pd

from concurrent import futures
from predict_pb2 import PredictionRequest
from predict_pb2 import PredictionResponse
from predict_pb2_grpc import PredictionServiceServicer
from predict_pb2_grpc import add_PredictionServiceServicer_to_server
from keras.models import load_model
from keras.metrics import top_k_categorical_accuracy


logging.basicConfig(level=logging.INFO)

FOOD_PATH = "food/"
FOOD_DIR = glob.glob("food/*")
CATEGORIES_PATH = "categories.csv"

MODELS = ["sg-food-233-densenet121", "sg-food-233-densenet201", "sg-food-233-xception"]
CHECKPOINT_PATHS = ["checkpoints/" + MODEL + ".h5" for MODEL in MODELS]
MODEL_PATHS = ["models/" + MODEL + ".h5" for MODEL in MODELS]

IMAGE_SIZE = 512
MAX_WORKERS = 10
PORT = 50051

categories = list(pd.read_csv(CATEGORIES_PATH, header=None)[0])


def acc_top5(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


def load_models():
    loaded_models = []
    for CHECKPOINT_PATH in CHECKPOINT_PATHS:
        loaded_models.append(load_model(CHECKPOINT_PATH, custom_objects={"acc_top5": acc_top5}))
    print("Models Loaded")
    return loaded_models


def predict(model, image, top_n=1):
    image = np.expand_dims(image, axis=0)
    image = image / 255.
    prediction = model.predict(image, verbose=0)
    predicted_label = np.argsort(prediction[0])[::-1][:top_n]
    predicted_score = prediction[0][predicted_label]
    return predicted_label, predicted_score


def ensemble_predict(models, image, top_n=1):
    predicted_labels = []
    predicted_scores = []
    prediction = {}
    for model in models:
        predicted_label, predicted_score = predict(model, image, top_n)
        predicted_labels = predicted_labels + list(predicted_label)
        predicted_scores = predicted_scores + list(predicted_score)
        for i, label in enumerate(predicted_labels):
            if predicted_labels[i] in prediction:
                if prediction[categories[predicted_labels[i]]] < predicted_scores[i]:
                    prediction[categories[predicted_labels[i]]] = predicted_scores[i]
            else:
                prediction[categories[predicted_labels[i]]] = predicted_scores[i]
    prediction = dict(sorted(prediction.items(), key=lambda item: item[1], reverse=True)[:top_n])
    return prediction


class PredictionService(PredictionServiceServicer):
    def predict(self, request, context):
        image = np.frombuffer(request.image, dtype=np.uint8)
        shape = tuple(json.loads(request.shape))
        image = image.reshape(shape)
        top_n = request.top_n
        models = load_models()
        prediction = ensemble_predict(models, image, top_n)
        prediction = {key: float(value) for key, value in prediction.items()}
        message = {"data": prediction}
        message = json.dumps(message)
        response = PredictionResponse(message=message)
        return response


def run():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS))
    add_PredictionServiceServicer_to_server(PredictionService(), server)
    server.add_insecure_port("[::]:{0}".format(PORT))
    server.start()
    logging.info("grpc_server_start")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("grpc_server_stop")
        server.stop(0)


def test():
    image = cv2.cvtColor(cv2.imread(FOOD_DIR[0]), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    models = load_models()
    prediction = ensemble_predict(models, image, 5)
    print(prediction)


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # test()
    run()
