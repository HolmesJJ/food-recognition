import json
import numpy as np
import pandas as pd

from keras.utils import load_img
from keras.utils import img_to_array
from keras.models import load_model
from keras.metrics import top_k_categorical_accuracy


CATEGORIES_PATH = "categories.csv"
MATCHES_PATH = "food-sg-233-empower-food-matched.csv"

MODELS = ["sg-food-233-densenet121", "sg-food-233-densenet201", "sg-food-233-xception"]
CHECKPOINT_PATHS = ["checkpoints/" + MODEL + ".h5" for MODEL in MODELS]

IMAGE_SIZE = 512
TOP_N = 5


def acc_top5(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


def load_models():
    categories = list(pd.read_csv(CATEGORIES_PATH, header=None)[0])
    matches = pd.read_csv(MATCHES_PATH)
    matches["empower_food_names"] = matches["empower_food_names"].apply(json.loads)
    matches["empower_food_alt_names"] = matches["empower_food_alt_names"].apply(json.loads)
    empower_food_names = matches.set_index("food_sg_233")["empower_food_names"].to_dict()
    empower_food_alt_names = matches.set_index("food_sg_233")["empower_food_alt_names"].to_dict()
    models = []
    for CHECKPOINT_PATH in CHECKPOINT_PATHS:
        models.append(load_model(CHECKPOINT_PATH, custom_objects={"acc_top5": acc_top5}))
    print("Models Loaded")
    return categories, empower_food_names, empower_food_alt_names, models


def predict(model, filepath, top_n=1):
    test_image = load_img(filepath, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    test_image_array = img_to_array(test_image)
    test_image_array = np.expand_dims(test_image_array, axis=0)
    test_image_array = test_image_array / 255.
    prediction = model.predict(test_image_array, verbose=0)
    predicted_label = np.argsort(prediction[0])[::-1][:top_n]
    predicted_score = prediction[0][predicted_label]
    return predicted_label, predicted_score


def ensemble_predict(categories, empower_food_names, empower_food_alt_names, models, filepath,
                     top_n_predictions=1, top_n_matches=1):
    predicted_labels = []
    predicted_scores = []
    predictions = {}
    for model in models:
        predicted_label, predicted_score = predict(model, filepath, top_n_predictions)
        predicted_labels = predicted_labels + list(predicted_label)
        predicted_scores = predicted_scores + list(predicted_score)
        for i, label in enumerate(predicted_labels):
            if predicted_labels[i] in predictions:
                if predictions[categories[predicted_labels[i]]] < predicted_scores[i]:
                    predictions[categories[predicted_labels[i]]] = predicted_scores[i]
            else:
                predictions[categories[predicted_labels[i]]] = predicted_scores[i]
    predictions = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True)[:top_n_predictions])
    matched_predictions = {}
    for key in predictions:
        matches = empower_food_names[key] + empower_food_alt_names[key]
        matches = sorted(matches, key=lambda item: (-item["similarity"], item["id"], item["name"]))
        unique_ids = set()
        matches = [item for item in matches if item["id"] not in unique_ids and not unique_ids.add(item["id"])]
        prediction = {
            "accuracy": predictions[key],
            "matches": matches[:top_n_matches]
        }
        matched_predictions[key] = prediction
    return matched_predictions


if __name__ == "__main__":
    cats, names, alt_names, mods = load_models()
    p = ensemble_predict(cats, names, alt_names, mods, "food/test1.png", top_n_predictions=5, top_n_matches=5)
    print(p)
