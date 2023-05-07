import os
import glob
import random
import shutil


RAW_DATA_DIRS = glob.glob("foodsg-233/*")
DATASET_PATH = "dataset/"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.15
TEST_RATIO = 0.05


def create_dataset():
    os.makedirs(DATASET_PATH)

    for raw_data_dir in RAW_DATA_DIRS:
        os.makedirs(DATASET_PATH + "train/" + os.path.basename(raw_data_dir))
        os.makedirs(DATASET_PATH + "val/" + os.path.basename(raw_data_dir))
        os.makedirs(DATASET_PATH + "test/" + os.path.basename(raw_data_dir))

    for raw_data_dir in RAW_DATA_DIRS:
        food = glob.glob(f"{raw_data_dir}/*")
        full_indices = range(len(food))
        train_indices = random.sample(full_indices, int(len(full_indices) * TRAIN_RATIO))
        val_test_indices = list(set(full_indices) - set(train_indices))
        val_indices = random.sample(val_test_indices, int(len(full_indices) * VAL_RATIO))
        test_indices = list(set(val_test_indices) - set(val_indices))
        print(raw_data_dir)
        print(len(full_indices), (len(train_indices) + len(val_indices) + len(test_indices)))
        for idx in train_indices:
            shutil.copyfile(food[idx], DATASET_PATH + "train/" + os.path.basename(raw_data_dir) + "/" +
                            os.path.basename(food[idx]))
        for idx in val_indices:
            shutil.copyfile(food[idx], DATASET_PATH + "val/" + os.path.basename(raw_data_dir) + "/" +
                            os.path.basename(food[idx]))
        for idx in test_indices:
            shutil.copyfile(food[idx], DATASET_PATH + "test/" + os.path.basename(raw_data_dir) + "/" +
                            os.path.basename(food[idx]))


if __name__ == "__main__":
    if not os.path.isdir(DATASET_PATH):
        create_dataset()
