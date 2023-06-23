import io
import json
import base64
import numpy as np
import pandas as pd

from keras.utils import load_img
from keras.utils import img_to_array
from keras.models import load_model
from keras.metrics import top_k_categorical_accuracy
from vit_keras.vit import vit_l32  # keep this line for VIT model!


CATEGORIES_PATH = "categories.csv"
MATCHES_PATH = "food-sg-233-empower-food-matched.csv"

MODELS = ["sg-food-233-xception", "sg-food-233-densenet121", "sg-food-233-densenet201",
          "sg-food-233-resnet152v2", "sg-food-233-inceptionv3", "sg-food-233-inceptionresnetv2"]
CHECKPOINT_PATHS = ["checkpoints/" + MODEL + ".h5" for MODEL in MODELS]

IMAGE_SIZE = 512
TOP_N = 5


def filepath_to_base64(filepath):
    with open(filepath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def acc_top5(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


def load_models():
    categories = list(pd.read_csv(CATEGORIES_PATH, header=None)[0])
    matches = pd.read_csv(MATCHES_PATH)
    matches["empower_food_names"] = matches["empower_food_names"].apply(json.loads)
    matches["empower_food_alt_names"] = matches["empower_food_alt_names"].apply(json.loads)
    matches["empower_categories"] = matches["empower_categories"].apply(json.loads)
    matches["empower_subcategories"] = matches["empower_subcategories"].apply(json.loads)
    empower_food_names = matches.set_index("food_sg_233")["empower_food_names"].to_dict()
    empower_food_alt_names = matches.set_index("food_sg_233")["empower_food_alt_names"].to_dict()
    empower_categories = matches.set_index("food_sg_233")["empower_categories"].to_dict()
    empower_subcategories = matches.set_index("food_sg_233")["empower_subcategories"].to_dict()
    models = []
    for CHECKPOINT_PATH in CHECKPOINT_PATHS:
        models.append(load_model(CHECKPOINT_PATH, custom_objects={"acc_top5": acc_top5}))
    print("Models Loaded")
    return categories, (empower_food_names, empower_food_alt_names, empower_categories, empower_subcategories), models


def get_label(row, reduction="mean"):
    max_freq_predictions = []
    predictions = row.iloc[:len(MODELS)]
    label_counts = row.iloc[len(MODELS):]
    max_freq_labels = label_counts[label_counts == label_counts.max()].index.tolist()
    for max_freq_label in max_freq_labels:
        scores = []
        for prediction in predictions:
            label, score = prediction
            if label == max_freq_label:
                scores.append(score)
        if reduction == "mean":
            max_freq_score = 0 if len(scores) == 0 else sum(scores) / len(scores)
        else:
            max_freq_score = 0 if len(scores) == 0 else max(scores)
        max_freq_predictions.append([max_freq_label, max_freq_score])
    max_freq_prediction = max(max_freq_predictions, key=lambda item: item[1])
    return {max_freq_prediction[0]: max_freq_prediction[1]}


def predict(model, filepath=None, encoded_string=None, top_n=None):
    assert filepath or encoded_string
    if filepath:
        test_image = load_img(filepath, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    else:
        image_bytes = base64.b64decode(encoded_string)
        image_file = io.BytesIO(image_bytes)
        test_image = load_img(image_file, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    test_image_array = img_to_array(test_image)
    test_image_array = np.expand_dims(test_image_array, axis=0)
    test_image_array = test_image_array / 255.
    predictions = model.predict(test_image_array, verbose=0)
    if top_n:
        predicted_labels = np.argsort(predictions[0])[::-1][:top_n]
    else:
        predicted_labels = np.argsort(predictions[0])[::-1]
    predicted_scores = predictions[0][predicted_labels]
    return predicted_labels, predicted_scores


def ensemble_predict(categories, empowers, models, filepath=None, encoded_string=None,
                     top_n_predictions=None, top_n_matches=None,
                     ensemble_type=1, reduction="mean",
                     name=False, similarity=False):
    """
        Ensemble prediction allow filepath or base64, and contains 2 types:

        Parameters:
        ensemble_type (int): 1 or 2.
            For 1: top_n_predictions and top_n_matches are available.
            For 2: top_n_matches and reduction are available.
        reduction (str): mean or max
    """
    assert filepath or encoded_string
    empower_food_names, empower_food_alt_names, empower_categories, empower_subcategories = empowers
    if ensemble_type == 1:
        predictions = {}
        for model in models:
            predicted_labels, predicted_scores = predict(model, filepath, encoded_string)
            for i, label in enumerate(predicted_labels):
                if categories[predicted_labels[i]] in predictions:
                    if predictions[categories[predicted_labels[i]]] < predicted_scores[i]:
                        predictions[categories[predicted_labels[i]]] = predicted_scores[i]
                else:
                    predictions[categories[predicted_labels[i]]] = predicted_scores[i]
        if top_n_predictions:
            predictions = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True)[:top_n_predictions])
        else:
            predictions = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))
    else:
        predictions = {}
        for i, model in enumerate(models):
            predicted_labels, predicted_scores = predict(model, filepath, encoded_string, top_n=1)
            predictions[MODELS[i]] = [[categories[predicted_labels[0]], predicted_scores[0]]]
        predictions = pd.DataFrame(predictions)
        predictions.columns = ["Prediction" + str(i + 1) for i in range(predictions.shape[1])]
        label_counts = predictions.apply(lambda row: pd.Series([item[0] for item in row]).value_counts(), axis=1).fillna(0)
        predictions = pd.concat([predictions, label_counts], axis=1)
        predictions = list(predictions.apply(lambda row: get_label(row, reduction), axis=1))[0]
    matched_predictions = {}
    for key in predictions:
        matches = empower_food_names[key] + empower_food_alt_names[key] + empower_categories[key] + empower_subcategories[key]
        matches = sorted(matches, key=lambda item: (-item["similarity"], item["_id"], item["name"]))
        unique_ids = set()
        matches = [item for item in matches if item["_id"] not in unique_ids and not unique_ids.add(item["_id"])]
        if not name:
            matches = [{k: v for k, v in item.items() if k != "name"} for item in matches]
        if not similarity:
            matches = [{k: v for k, v in item.items() if k != "similarity"} for item in matches]
        prediction = {
            "accuracy": predictions[key],
            "matches": matches[:top_n_matches] if top_n_matches else matches
        }
        matched_predictions[key] = prediction
    return matched_predictions


if __name__ == "__main__":
    cats, eps, mods = load_models()
    es = filepath_to_base64("food/chicken rice.jpg")
    p = ensemble_predict(cats, eps, mods, encoded_string=es,
                         top_n_predictions=1, top_n_matches=1,
                         ensemble_type=2, reduction="mean",
                         name=False, similarity=False)
    print(p)
