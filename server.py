from flask import Flask
from flask import request
from flask_cors import CORS
from flask_restful import Api
from flask_restful import Resource
from inference import load_models
from inference import ensemble_predict


application = Flask(__name__)
application.config["CORS_HEADERS"] = "Content-Type"
application.config["CORS_RESOURCES"] = {r"/api/*": {"origins": "*"}}
application.config["PROPAGATE_EXCEPTIONS"] = True

cors = CORS(application)
api = Api(application)


class Index(Resource):
    def get(self):
        return "Hello World"


class Prediction(Resource):
    def post(self):
        content = request.json
        response = {
            "code": 0,
            "message": ""
        }
        if "image" not in content:
            response["message"] = "image missing"
            return response
        image = content["image"]
        top_n_predictions = 1 if "top_n_predictions" not in content else content["top_n_predictions"]
        top_n_matches = 1 if "top_n_matches" not in content else content["top_n_matches"]
        ensemble_type = 2 if "ensemble_type" not in content else content["ensemble_type"]
        reduction = "mean" if "reduction" not in content else content["reduction"]
        name = False if "name" not in content else bool(content["name"])
        similarity = False if "similarity" not in content else bool(content["similarity"])
        try:
            p = ensemble_predict(cats, eps, mods, encoded_string=image,
                                 top_n_predictions=top_n_predictions, top_n_matches=top_n_matches,
                                 ensemble_type=ensemble_type, reduction=reduction,
                                 name=name, similarity=similarity)
            response["code"] = 1
            response["message"] = "success"
            response["prediction"] = [p]
        except (Exception, ):
            response["message"] = "predict error"
        return response


api.add_resource(Index, "/", endpoint="index")
api.add_resource(Prediction, "/api/predict", endpoint="predict")


if __name__ == "__main__":
    cats, eps, mods = load_models()
    # application.debug = True
    application.run(host="0.0.0.0", port=5000)
