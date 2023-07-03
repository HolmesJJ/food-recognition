# https://towardsdatascience.com/cifar-100-transfer-learning-using-efficientnet-ed3ed7b89af2
# https://www.kaggle.com/code/elcaiseri/mnist-simple-cnn-keras-accuracy-0-99-top-1#4.-Evaluate-the-model
# https://github.com/106368015AlvinYang/Taiwanese-Food-101/blob/master/Taiwanese-Food-101.ipynb
# https://www.kaggle.com/code/theimgclist/multiclass-food-classification-using-tensorflow
# https://www.kaggle.com/code/karan842/pneumonia-detection-transfer-learning-94-acc
# https://www.kaggle.com/code/theeyeschico/food-classification-using-tensorflow
# https://www.kaggle.com/code/abhijeetbhilare/food-classification-using-resnet
# https://www.kaggle.com/code/niharika41298/food-nutrition-analysis-eda
# https://www.kaggle.com/code/artgor/food-recognition-challenge-eda
# https://www.kaggle.com/datasets/kmader/food41
# https://keras.io/api/applications/

import os
import glob
import math
import matplotlib.pyplot as plt

from keras.models import Model
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from keras.applications import Xception
from keras.applications import ResNet50V2
from keras.applications import MobileNetV2
from keras.applications import InceptionV3
from keras.applications import DenseNet121
from keras.applications import DenseNet201
from keras.applications import ResNet152V2
from keras.applications import InceptionResNetV2
# https://github.com/keras-team/keras/issues/17199
# from keras.applications import EfficientNetB3 have bugs
from networks.efficientnet import EfficientNetB3
from vit_keras.vit import vit_l32
from keras.optimizers import Adam


DATASET_DIRS = glob.glob("dataset/*")
TRAIN_PATH = "dataset/train/"
VAL_PATH = "dataset/val/"
TEST_PATH = "dataset/test/"
TRAIN_DIRS = glob.glob("dataset/train/*")
VAL_DIRS = glob.glob("dataset/val/*")
TEST_DIRS = glob.glob("dataset/test/*")

BATCH_SIZE = 32
IMAGE_SIZE = 512
MODEL = "VitL32"
CHECKPOINT_PATH = "checkpoints/" + MODEL + ".h5"
FIGURE_PATH = "figures/" + MODEL + ".png"
MODEL_PATH = "models/" + MODEL + ".h5"
LOG_PATH = "logs/" + MODEL + ".log"


def show_food(name):
    food_path = TRAIN_PATH + name
    food = os.listdir(food_path)

    plt.figure(figsize=(15, 5))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        img = plt.imread(os.path.join(food_path, food[i]))
        plt.title(name)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def run_data_augmentation():
    train_datagen = ImageDataGenerator(
        rescale=1/255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.5, 3.0],
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=20,
        fill_mode="nearest"
    )
    val_test_datagen = ImageDataGenerator(rescale=1/255)
    train_data = train_datagen.flow_from_directory(TRAIN_PATH,
                                                   shuffle=True,
                                                   batch_size=BATCH_SIZE,
                                                   target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                   class_mode="categorical")
    val_data = val_test_datagen.flow_from_directory(VAL_PATH,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                    class_mode="categorical")
    test_data = val_test_datagen.flow_from_directory(TEST_PATH,
                                                     batch_size=1,
                                                     shuffle=False,
                                                     target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                     class_mode="categorical")
    return train_data, val_data, test_data


def acc_top5(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


def compile_model():
    if not os.path.exists(CHECKPOINT_PATH):
        # net = DenseNet121(
        #     input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        #     weights="imagenet",
        #     include_top=False,
        #     # classes=len(TRAIN_DIRS)
        # )
        net = vit_l32(
            image_size=IMAGE_SIZE,
            activation="softmax",
            pretrained=True,
            pretrained_top=False,
            include_top=False)
        for layer in net.layers:
            layer.trainable = False
        x = net.output
        # x = GlobalAveragePooling2D()(x)
        x = Dense(4096, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        x = Dense(4096, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        x = Dense(4096, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        x = Dense(4096, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        x = Dense(4096, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        predictions = Dense(len(TRAIN_DIRS), activation="softmax")(x)
        model = Model(inputs=net.input, outputs=predictions)
        optimizer = Adam(learning_rate=1e-5)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy", acc_top5])
    else:
        model = load_model(CHECKPOINT_PATH, custom_objects={"acc_top5": acc_top5})
        print("Checkpoint Model Loaded")
    early_stopping = EarlyStopping(monitor="val_accuracy", mode="max", patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(CHECKPOINT_PATH, monitor="val_accuracy", save_best_only=True,
                                 verbose=1, save_weights_only=False)
    lr = ReduceLROnPlateau(monitor="val_accuracy", mode="max", min_lr=1e-7, patience=10)
    csv_logger = CSVLogger(LOG_PATH)
    print(model.summary())
    return model, early_stopping, checkpoint, lr, csv_logger


def train():
    train_data, val_data, test_data = run_data_augmentation()
    model, early_stopping, checkpoint, lr, csv_logger = compile_model()
    history = model.fit(train_data, epochs=300,
                        validation_data=val_data,
                        callbacks=[early_stopping, checkpoint, lr, csv_logger],
                        steps_per_epoch=math.ceil(train_data.samples / BATCH_SIZE),
                        batch_size=BATCH_SIZE,
                        validation_steps=math.ceil(val_data.samples / BATCH_SIZE),
                        validation_batch_size=BATCH_SIZE)

    train_score = model.evaluate(train_data, batch_size=BATCH_SIZE, steps=math.ceil(train_data.samples / BATCH_SIZE))
    print("Train Loss: ", train_score[0])
    print("Train Accuracy: ", train_score[1])

    val_score = model.evaluate(val_data)
    print("Validation Loss: ", val_score[0])
    print("Validation Accuracy: ", val_score[1])

    test_score = model.evaluate(test_data)
    print("Test Loss: ", test_score[0])
    print("Test Accuracy: ", test_score[1])

    model.save(MODEL_PATH)

    plt.figure(figsize=(12, 8))
    plt.title("EVALUATION")
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Loss")
    plt.plot(history.history["val_loss"], label="Val_Loss")
    plt.legend()
    plt.title("Loss Evaluation")
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val_Accuracy")
    plt.legend()
    plt.title("Accuracy Evaluation")
    # plt.show()
    plt.savefig(FIGURE_PATH)


if __name__ == "__main__":
    # show_food("Apple")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train()
