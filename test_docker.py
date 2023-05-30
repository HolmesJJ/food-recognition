import cv2
import glob

from time import time
from model_docker import ModelDocker


FOOD_DIR = glob.glob("food/*")

IMAGE_SIZE = 512


if __name__ == "__main__":
    start_time = time()
    with ModelDocker() as sess:
        image = cv2.cvtColor(cv2.imread(FOOD_DIR[0]), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        result = sess.run(image=image, top_n=5)
        print("result:", result)
    end_time = time()
    print("runtime:", (end_time - start_time))
