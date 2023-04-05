# https://www.kaggle.com/code/karan842/pneumonia-detection-transfer-learning-94-acc
# https://www.kaggle.com/code/elcaiseri/mnist-simple-cnn-keras-accuracy-0-99-top-1#4.-Evaluate-the-model
# https://www.kaggle.com/code/theimgclist/multiclass-food-classification-using-tensorflow
# https://www.kaggle.com/code/theeyeschico/food-classification-using-tensorflow
# https://www.kaggle.com/code/abhijeetbhilare/food-classification-using-resnet
# https://www.kaggle.com/code/niharika41298/food-nutrition-analysis-eda
# https://www.kaggle.com/code/artgor/food-recognition-challenge-eda
# https://www.kaggle.com/datasets/kmader/food41
# https://keras.io/api/applications/

import os
import glob
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from keras.applications import ResNet50V2
from keras.applications import MobileNetV2
from keras.applications import InceptionV3
from keras.applications import EfficientNetB0
from keras.regularizers import l2
from keras.optimizers import Adam


DATASET_DIRS = glob.glob("dataset/*")
TRAIN_PATH = "dataset/train/"
VAL_PATH = "dataset/val/"
TEST_PATH = "dataset/test/"
TRAIN_DIRS = glob.glob("dataset/train/*")
VAL_DIRS = glob.glob("dataset/val/*")
TEST_DIRS = glob.glob("dataset/test/*")

BATCH_SIZE = 32
MODEL = "EfficientNetB0.h5"  # ResNet50V2.h5, MobileNetV2.h5, InceptionV3.h5, EfficientNetB0.h5
CHECKPOINT_PATH = "checkpoints/" + MODEL
MODEL_PATH = "models/" + MODEL
LOG_PATH = "logs/" + MODEL


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
    img_datagen = ImageDataGenerator(
        rescale=1 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.5, 2.0],
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=20,
        fill_mode="nearest"
    )
    val_test_datagen = ImageDataGenerator(rescale=1 / 255)
    train_data = img_datagen.flow_from_directory(TRAIN_PATH, batch_size=BATCH_SIZE, class_mode="categorical")
    val_data = val_test_datagen.flow_from_directory(VAL_PATH, batch_size=BATCH_SIZE, class_mode="categorical")
    test_data = val_test_datagen.flow_from_directory(TEST_PATH, batch_size=BATCH_SIZE, class_mode="categorical")
    return train_data, val_data, test_data


def compile_model():
    net = EfficientNetB0(
        weights="imagenet",
        include_top=False,
    )
    for layer in net.layers:
        layer.trainable = False
    x = net.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    predictions = Dense(len(TRAIN_DIRS), kernel_regularizer=l2(0.005), activation="softmax")(x)
    model = Model(inputs=net.input, outputs=predictions)
    early_stopping = EarlyStopping(monitor="val_accuracy", mode="max", patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(CHECKPOINT_PATH, monitor="val_accuracy", save_best_only=True, verbose=1)
    lr = ReduceLROnPlateau(monitor="val_accuracy", mode="max", patience=10)
    csv_logger = CSVLogger(LOG_PATH)
    optimizer = Adam(lr=0.00001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    print(model.summary())
    return model, early_stopping, checkpoint, lr, csv_logger


def train():
    train_data, val_data, test_data = run_data_augmentation()
    model, early_stopping, checkpoint, lr, csv_logger = compile_model()
    history = model.fit(train_data, epochs=200,
                        validation_data=val_data,
                        callbacks=[early_stopping, checkpoint, lr, csv_logger],
                        batch_size=BATCH_SIZE)

    train_score = model.evaluate(train_data)
    print("Train Loss: ", train_score[0])
    print("Train Accuracy: ", train_score[1])

    test_score = model.evaluate(test_data)
    print("Test Loss: ", test_score[0])
    print("Test Accuracy: ", test_score[1])

    model.save_weights(MODEL_PATH)

    plt.figure(figsize=(12, 8))
    plt.title("EVALUATION OF VGG19")
    plt.subplot(2, 2, 1)
    plt.plot(history.history["loss"], label="Loss")
    plt.plot(history.history["val_loss"], label="Val_Loss")
    plt.legend()
    plt.title("Loss Evolution")
    plt.subplot(2, 2, 2)
    plt.plot(history.history["accuracy"], label="Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val_Accuracy")
    plt.legend()
    plt.title("Accuracy Evolution")
    plt.show()


if __name__ == '__main__':
    # show_food("Apple")
    train()
