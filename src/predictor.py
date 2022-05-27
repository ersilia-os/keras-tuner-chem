import os
import numpy as np
import onnxruntime as rt
import tensorflow as tf

from .descriptor import featurizer


class OnnxPredictor(object):
    def __init__(self, onnx_file):
        self.model_file = onnx_file

    def predict(self, smiles):
        X = featurizer(smiles)
        X = X.astype(np.float32)
        sess = rt.InferenceSession(self.model_file)
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        output_data = sess.run([output_name], {input_name: X})
        y = np.array(output_data[0])
        return y


class TflitePredictor(object):
    def __init__(self, tflite_file):
        self.model_file = tflite_file

    def predict(self, smiles):
        X = featurizer(smiles)
        X = X.astype(np.float32)
        interpreter = tf.lite.Interpreter(self.model_file)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        output_data = []
        for x in X:
            interpreter.set_tensor(input_details[0]["index"], [x])
            interpreter.invoke()
            output_data += [interpreter.get_tensor(output_details[0]["index"])[0]]
        y = np.array(output_data)
        return y
