
# example inputs

smiles = ["CCCOCCC", "CCCCNCCC"]

# prediction with model in onnx format

model_file = "model.onnx"

from src.predictor import OnnxPredictor

mdl = OnnxPredictor(model_file)
print(mdl.predict(smiles))

# prediction with model in tflite format

model_file = "model.tflite"

from src.predictor import TflitePredictor

mdl = TflitePredictor(model_file)
print(mdl.predict(smiles))