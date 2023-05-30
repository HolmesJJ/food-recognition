# https://zhuanlan.zhihu.com/p/42809515

import grpc
import json
import docker
import numpy as np

from predict_pb2 import PredictionRequest
from predict_pb2_grpc import PredictionServiceStub


IMAGE = "food-recognition:v1.0"
PORT = 50051


class ModelDocker(object):

    def __init__(self):
        self.client = docker.from_env()
        self.container = None
        self.is_grpc_start = False

    @staticmethod
    def get_container(client):
        container = client.containers.run(image=IMAGE,
                                          command="python predict.py",
                                          # runtime="nvidia",
                                          # environment=["CUDA_VISIBLE_DEVICES=0"],
                                          ports={"{0}/tcp".format(PORT): PORT},
                                          detach=True,
                                          auto_remove=True)
        return container

    def __enter__(self):
        self.container = self.get_container(self.client)
        print("container has started...")
        messages = []
        for line in self.container.logs(stream=True):
            messages.append(line.decode().strip())
            if line.strip().find(b"grpc_server_start") >= 0:
                self.is_grpc_start = True
                print("grpc server has started...")
                break
        if not self.is_grpc_start:
            print("\n".join(messages))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.container.stop()
        print("container has stopped...")

    def run(self, image, top_n):
        if not self.container:
            return {"code": -1, "message": "container error"}
        if not self.is_grpc_start:
            return {"code": -1, "message": "grpc error"}
        if not isinstance(image, np.ndarray):
            return {"code": 0, "message": "image must be numpy array"}
        shape = json.dumps(image.shape)
        image = image.tobytes()
        stub = PredictionServiceStub(grpc.insecure_channel("localhost:{0}".format(PORT)))
        response = stub.predict(PredictionRequest(image=image, shape=shape, top_n=top_n))
        result = {"code": 1, "message": "success"}
        result = {**result, **json.loads(response.message)}
        return result
